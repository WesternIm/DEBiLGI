import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from debilgi.discord.delivery import DiscordReplyTransport, ReplyDeliveryService, delivery_nonce
from debilgi.discord.delivery_repository import DeliveryRepository, decode_reply, encode_reply
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.language.research_wait import LexicalResearchPending
from debilgi.language.semantic_reply_plan import ProcessingOutcome, ProcessingOutcomeType
from debilgi.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / 'migrations'
NOW = datetime.now(UTC)


def event(message_id=10, *, bot=False):
    return normalize_message_created(
        message_id=message_id, guild_id=1, channel_id=2, author_id=3,
        created_at=NOW, content='hello', author_is_bot=bot, bot_mentioned=True,
    )


def reply():
    return PreparedReply('hello there', 'intent:1', ('hello', 'there'), 'genome', 2,
                         fallback=True, pending_concept='hello')


@pytest.fixture
async def environment(tmp_path):
    database = Database(tmp_path / 'delivery.sqlite3', MIGRATIONS)
    await database.initialize()
    events = EventRepository(database)
    repository = DeliveryRepository(database)
    workflow = SimpleNamespace(prepare=AsyncMock(return_value=reply()), record_sent=AsyncMock())
    workflow.validate = AsyncMock(side_effect=lambda _event, prepared: prepared)
    sent = SimpleNamespace(id=100, created_at=NOW)
    transport = SimpleNamespace(channel=AsyncMock(return_value=object()),
                                send=AsyncMock(return_value=sent),
                                find_sent=AsyncMock(return_value=None))
    service = ReplyDeliveryService(repository, events, workflow, transport)
    try:
        yield SimpleNamespace(database=database, events=events, repository=repository,
                              workflow=workflow, transport=transport, service=service, sent=sent)
    finally:
        await service.stop()
        await database.close()


async def ready(env):
    await env.database.execute(
        "UPDATE reply_delivery_jobs SET next_attempt_at = CURRENT_TIMESTAMP"
    )


async def test_event_and_job_are_atomic_and_duplicate_delivery_is_once(environment):
    env = environment
    incoming = event()
    assert await env.events.append(incoming)
    assert not await env.events.append(incoming)
    await env.events.append(event(11, bot=True))
    await env.events.append(replace(event(12), source='archive'))
    rows = await env.database.fetchall('SELECT * FROM reply_delivery_jobs')
    assert len(rows) == 1
    await asyncio.gather(env.service.run_once(), env.service.run_once())
    assert (await env.repository.get(incoming.event_id))['status'] == 'completed'
    env.transport.send.assert_awaited_once()
    env.workflow.record_sent.assert_awaited_once()


async def test_job_insert_failure_rolls_back_event(environment):
    env = environment
    await env.database.execute(
        """CREATE TRIGGER fail_job BEFORE INSERT ON reply_delivery_jobs
           BEGIN SELECT RAISE(ABORT, 'job failure'); END"""
    )
    with pytest.raises(Exception, match='job failure'):
        await env.events.append(event())
    assert await env.events.get(event().event_id) is None


@pytest.mark.parametrize('stage', ['pending', 'preparing', 'prepared'])
async def test_restart_resumes_before_send(environment, stage):
    env = environment
    incoming = event()
    await env.events.append(incoming)
    if stage == 'prepared':
        await env.repository.save_reply(incoming.event_id, reply())
    else:
        await env.repository.state(incoming.event_id, stage)
    await env.database.close()
    await env.database.initialize()
    await env.repository.recover()
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'completed'
    env.transport.send.assert_awaited_once()
    assert env.workflow.prepare.await_count == (0 if stage == 'prepared' else 1)


async def test_ambiguous_send_never_retries_without_receipt(environment):
    env = environment
    incoming = event()
    await env.events.append(incoming)
    env.transport.send.side_effect = TimeoutError('receipt lost')
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'uncertain'
    await env.repository.recover()
    await ready(env)
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'uncertain'
    env.transport.send.assert_awaited_once()
    env.workflow.record_sent.assert_not_awaited()
    env.transport.find_sent.return_value = env.sent
    await ready(env)
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'completed'
    env.transport.send.assert_awaited_once()
    env.workflow.record_sent.assert_awaited_once()


async def test_uncertain_receipt_does_not_block_new_delivery(environment):
    env = environment
    uncertain, newer = event(10), event(12)
    await env.events.append(uncertain)
    await env.repository.save_reply(uncertain.event_id, reply())
    await env.repository.state(uncertain.event_id, 'uncertain')
    await env.events.append(newer)

    await env.service.run_once()

    assert (await env.repository.get(newer.event_id))['status'] == 'completed'
    assert (await env.repository.get(uncertain.event_id))['status'] == 'uncertain'
    env.transport.send.assert_awaited_once()
    env.transport.find_sent.assert_not_awaited()


async def test_stale_uncertain_receipt_becomes_terminal_without_resending(environment):
    env = environment
    incoming = replace(event(), timestamp=NOW - timedelta(days=2))
    await env.events.append(incoming)
    await env.repository.save_reply(incoming.event_id, reply())
    await env.repository.state(incoming.event_id, 'uncertain')

    await env.service.run_once()

    job = await env.repository.get(incoming.event_id)
    assert job['status'] == 'failed'
    assert job['last_error'] == 'receipt_not_found_after_reconciliation_window'
    env.transport.find_sent.assert_awaited_once()
    env.transport.send.assert_not_awaited()


async def test_restart_during_send_requires_reconciliation(environment):
    env = environment
    incoming = event()
    await env.events.append(incoming)
    await env.repository.save_reply(incoming.event_id, reply())
    await env.repository.state(incoming.event_id, 'sending')
    await env.repository.recover()
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'uncertain'
    env.transport.send.assert_not_awaited()


async def test_bookkeeping_failure_retries_without_resending(environment):
    env = environment
    incoming = event()
    await env.events.append(incoming)
    env.workflow.record_sent.side_effect = RuntimeError('bookkeeping unavailable')
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'sent'
    env.workflow.record_sent.side_effect = None
    await ready(env)
    await env.repository.recover()
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'completed'
    env.transport.send.assert_awaited_once()
    assert env.workflow.record_sent.await_count == 2


async def test_preparation_failure_is_bounded(environment):
    env = environment
    await env.events.append(event())
    env.workflow.prepare.side_effect = RuntimeError('prepare failed')
    for _ in range(4):
        await ready(env)
        await env.service.run_once()
    assert (await env.repository.get(event().event_id))['status'] == 'failed'
    assert env.workflow.prepare.await_count == 3
    env.transport.send.assert_not_awaited()


async def test_dictionary_wait_survives_restart_and_does_not_block_other_replies(environment):
    env = environment
    waiting, other = event(10), event(12)
    await env.events.append(waiting)
    await env.events.append(other)
    available = False

    async def prepare(incoming):
        if incoming.event_id == waiting.event_id and not available:
            raise LexicalResearchPending(("термин",), 300)
        return reply()

    env.workflow.prepare.side_effect = prepare
    await env.service.run_once()
    job = await env.repository.get(waiting.event_id)
    assert job['status'] == 'pending'
    assert job['prepared_json'] is None
    env.transport.send.assert_not_awaited()
    await env.service.run_once()
    assert (await env.repository.get(other.event_id))['status'] == 'completed'
    await env.database.close()
    await env.database.initialize()
    await env.repository.recover()
    assert (await env.repository.get(waiting.event_id))['status'] == 'pending'
    available = True
    await ready(env)
    await env.service.run_once()
    assert (await env.repository.get(waiting.event_id))['status'] == 'completed'
    assert env.transport.send.await_count == 2  # one send per original event


async def test_no_reply_is_terminal(environment):
    env = environment
    await env.events.append(event())
    env.workflow.prepare.return_value = None
    await env.service.run_once()
    await env.repository.recover()
    assert not await env.service.run_once()
    assert (await env.repository.get(event().event_id))['status'] == 'no_reply'
    env.transport.send.assert_not_awaited()


def test_prepared_reply_metadata_round_trip():
    prepared = replace(reply(), processing_outcome=ProcessingOutcome(
        ProcessingOutcomeType.REPLIED, 'intention', 'plan', 'hello there',
        None, None, False, {'confidence': 0.8},
    ))
    assert decode_reply(encode_reply(prepared)) == prepared


async def test_restored_reply_is_withheld_when_context_changed(environment):
    env = environment
    incoming = event()
    await env.events.append(incoming)
    await env.repository.save_reply(incoming.event_id, reply())
    env.workflow.validate.side_effect = None
    env.workflow.validate.return_value = None
    await env.repository.recover()
    await env.service.run_once()
    assert (await env.repository.get(incoming.event_id))['status'] == 'no_reply'
    env.transport.send.assert_not_awaited()


async def test_transport_uses_stable_nonce_and_correct_reply_reference():
    channel = SimpleNamespace(guild=SimpleNamespace(id=1), send=AsyncMock())
    bot = SimpleNamespace(get_channel=lambda _id: channel)
    transport = DiscordReplyTransport(bot)
    incoming = event()
    assert await transport.channel(incoming) is channel
    await transport.send(channel, incoming, reply())
    options = channel.send.call_args.kwargs
    assert options['nonce'] == delivery_nonce(incoming.event_id)
    assert options['reference'].message_id == 10
    assert options['reference'].channel_id == 2
    assert options['allowed_mentions'].everyone is False
    channel.guild.id = 99
    with pytest.raises(ValueError, match='guild'):
        await transport.channel(incoming)


async def test_reconciliation_requires_nonce_and_bot_author():
    incoming = event()
    messages = [
        SimpleNamespace(id=100, nonce='unrelated', content=reply().text,
                        author=SimpleNamespace(id=99)),
        SimpleNamespace(id=101, nonce=delivery_nonce(incoming.event_id), content=reply().text,
                        author=SimpleNamespace(id=3)),
        SimpleNamespace(id=102, nonce=delivery_nonce(incoming.event_id), content=reply().text,
                        author=SimpleNamespace(id=99)),
    ]

    async def history(**_kwargs):
        for message in messages:
            yield message

    channel = SimpleNamespace(guild=SimpleNamespace(id=1), history=history)
    bot = SimpleNamespace(user=SimpleNamespace(id=99), get_channel=lambda _id: channel)
    transport = DiscordReplyTransport(bot)
    assert (await transport.find_sent(incoming, reply())).id == 102
    messages.pop()
    assert await transport.find_sent(incoming, reply()) is None
