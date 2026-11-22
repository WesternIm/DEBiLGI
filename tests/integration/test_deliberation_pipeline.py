"""Reasoning retries a supported plan after a failed surface, without Discord."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.config import Settings
from debilgi.discord.bot import create_bot
from debilgi.discord.reply_workflow import ReplyWorkflow
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.language.communication import ResponseComposer
from debilgi.storage.database import Database


async def test_rejected_primary_surface_returns_to_next_supported_plan(tmp_path, monkeypatch):
    database = Database(tmp_path / 'deliberation.sqlite3', Path('migrations'))
    bot = create_bot(Settings(discord_token=None, migrations_path=Path('migrations'),
                              database_path=database.path, test_replies_enabled=True,
                              ), database, EventRepository(database))
    await database.initialize()
    await bot.pipeline.start()
    now = datetime(2026, 11, 14, 14, tzinfo=UTC)

    def event(n, text, actor, parent=None):
        return normalize_message_created(
            message_id=n, guild_id=1, channel_id=2, author_id=actor,
            created_at=now + timedelta(seconds=n * 40), content=text,
            bot_user_id=99, author_is_bot=False, bot_mentioned=actor == 3,
            reply_to_message_id=parent,
        )

    try:
        for turn in (event(1, 'ты крутой', 10), event(2, 'Ого!', 11, 1),
                     event(3, 'ты крутой', 12), event(4, 'Ну а то)', 13, 3)):
            await bot.pipeline.publish(turn)
        await bot.pipeline.join()
        query = event(10, 'ну ты бля крутой', 3)
        forms = await bot.semantic_assimilation.dialogue_learning.forms_for(
            query, query.meaning.text)
        cores = [f for f in forms if f.component == 'core']
        assert {'agree', 'acknowledge'} <= {f.meaning for f in cores}
        first = max(cores, key=lambda f: (f.activation, f.support, f.rule_id or '')).meaning
        attempted = []
        original = ResponseComposer._realize_reaction

        def realize(composer, request, record):
            meaning = request.plan.claim.predicate.lemma
            attempted.append(meaning)
            if meaning == first:
                record['status'] = 'unexpressed'
                record['surface_checks'].append({'text': '', 'issues': ('test_reader_rejection',)})
                return ()
            return original(composer, request, record)

        monkeypatch.setattr(ResponseComposer, '_realize_reaction', realize)
        workflow = ReplyWorkflow(SimpleNamespace(user=SimpleNamespace(id=99)), bot.pipeline,
                                 bot.reply_service, bot.context_gate, bot.response_traces)
        await bot.pipeline.publish(query)
        await bot.pipeline.join()
        exchanges_before = (await database.fetchone(
            "SELECT COUNT(*) FROM language_observed_exchanges"))[0]
        prepared = await workflow.prepare(query)
        assert prepared is not None, attempted
        assert attempted[0] == first and any(a != first for a in attempted)
        assert len(attempted) <= 4
        assert prepared.deliberation_steps > 0
        assert await workflow.validate(query, prepared)
        # Failed wording cannot accidentally become an outgoing fact or training pair.
        assert (await database.fetchone(
            "SELECT COUNT(*) FROM language_observed_exchanges"))[0] == exchanges_before
    finally:
        await bot.pipeline.stop()
        await database.close()
        await bot.close()


async def test_causal_ellipsis_survives_full_dialogue_task_and_arbitration(tmp_path):
    database = Database(tmp_path / 'causal.sqlite3', Path('migrations'))
    bot = create_bot(Settings(discord_token=None, migrations_path=Path('migrations'),
                              database_path=database.path, test_replies_enabled=True,
                              ), database, EventRepository(database))
    await database.initialize()
    await bot.pipeline.start()
    now = datetime(2026, 11, 14, 14, tzinfo=UTC)
    try:
        for n, text, actor, addressed in (
            (1, 'О чём ты спрашиваешь?', 10, False),
            (10, 'потому что ты?', 3, True),
        ):
            incoming = normalize_message_created(
                message_id=n, guild_id=1, channel_id=2, author_id=actor,
                created_at=now + timedelta(seconds=n * 40), content=text,
                bot_user_id=99, author_is_bot=False, bot_mentioned=addressed,
            )
            await bot.pipeline.publish(incoming)
        workflow = ReplyWorkflow(SimpleNamespace(user=SimpleNamespace(id=99)), bot.pipeline,
                                 bot.reply_service, bot.context_gate, bot.response_traces)
        prepared = await workflow.prepare(incoming)
        steps = await bot.response_traces.steps_for_event(incoming.event_id)
        assert prepared is not None, steps
        assert prepared.text.endswith('?') and 'спрашиваешь' in prepared.text
    finally:
        await bot.pipeline.stop()
        await database.close()
        await bot.close()


async def _recovery_case(tmp_path, monkeypatch, fallback):
    database = Database(tmp_path / 'reconsideration.sqlite3', Path('migrations'))
    bot = create_bot(Settings(discord_token=None, migrations_path=Path('migrations'),
                              database_path=database.path, test_replies_enabled=True,
                              ), database, EventRepository(database))
    await database.initialize()
    await bot.pipeline.start()
    now = datetime(2026, 11, 14, 14, tzinfo=UTC)

    def turn(n, text, actor, parent=None):
        return normalize_message_created(
            message_id=n, guild_id=1, channel_id=2, author_id=actor,
            created_at=now + timedelta(seconds=n * 40), content=text,
            bot_user_id=99, author_is_bot=False, bot_mentioned=actor == 3,
            reply_to_message_id=parent)

    try:
        for e in (turn(1, 'Привет', 10), turn(2, 'Ну привет!', 11, 1),
                  turn(3, 'Я инженер', 12), turn(4, 'Ого', 13, 3),
                  turn(5, 'Ты любишь читать?', 14), turn(6, 'Ого', 15, 5),
                  turn(7, 'Я врач', 16), turn(8, 'Ого', 17, 7),
                  turn(9, 'Я читаю книгу', 18), turn(10, 'Книгу?', 19, 9)):
            await bot.pipeline.publish(e)
        await bot.pipeline.join()
        original = ResponseComposer._realize_reaction

        def reject_supported(composer, request, record):
            if request.plan.review.get('phase') != fallback:
                record['status'] = 'unexpressed'
                record['surface_checks'].append({
                    'text': '', 'issues': ('controlled_reader_failure',)})
                return ()
            return original(composer, request, record)

        monkeypatch.setattr(ResponseComposer, '_realize_reaction', reject_supported)
        workflow = ReplyWorkflow(SimpleNamespace(user=SimpleNamespace(id=99)), bot.pipeline,
                                 bot.reply_service, bot.context_gate, bot.response_traces)
        query = turn(12, 'эй' if fallback == 'tone' else 'лошара', 3)
        await bot.pipeline.publish(query)
        await bot.pipeline.join()
        prepared = await workflow.prepare(query)
        assert prepared, await bot.response_traces.steps_for_event(query.event_id)
        assert await workflow.validate(query, prepared)
        diagnostic = await bot.response_traces.diagnostics_for_event(
            query.event_id, status='prepared', final_text=prepared.text,
            selected_route=prepared.response_source)
        content = diagnostic['content']
        review = content['reconsideration']
        assert review['reactivation']['source'] == 'reader_failure'
        assert review['failures'] and review['questions']
        assert content['reaction_meaning'] == (
            'acknowledge' if fallback == 'tone' else 'slot_question')
        if fallback == 'clarification':
            assert 'лошара' in prepared.text and prepared.text.endswith('?')
            assert review['reactivation']['level'] == 'clarification'
        assert content['reverse_check']['passed'] and content['hypothetical_assertions'] == 0
        # Recovering is not a new human demonstration or a source of world claims.
        assert not await database.fetchone(
            "SELECT 1 FROM language_observed_exchanges WHERE prompt_event_id=?",
            (query.event_id,))
        return {'input': query.meaning.text, 'answer': prepared.text,
                'meaning': content['reaction_meaning'], 'reconsideration': review,
                'reverse_check': content['reverse_check'],
                'hypothetical_assertions': content['hypothetical_assertions']}
    finally:
        await bot.pipeline.stop()
        await database.close()
        await bot.close()


@pytest.mark.parametrize('fallback', ['tone', 'clarification'])
async def test_all_supported_cores_rejected_reactivate_human_tone(tmp_path, monkeypatch, fallback):
    assert (await _recovery_case(tmp_path, monkeypatch, fallback))['answer']
