"""Sleep rhythm, durable work boundaries, isolation and daylight competition."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.discord.delivery import ReplyDeliveryService
from debilgi.discord.delivery_repository import DeliveryRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.language.arbitration import ArbitrationProfile, ResponseArbitrator, ResponseCandidate
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 17, 10, tzinfo=UTC)


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "sleep.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def event(n=1, text="а ты предположи", **kwargs):
    return Event(f"sleep:{n}", 1, 2, 3, 9, EventType.MESSAGE_CREATED,
                 NOW + timedelta(seconds=n),
                 {"message_id": n, "bot_user_id": 9, "bot_mentioned": True,
                  "content": text, "author_is_bot": False, **kwargs}, "discord")


def candidate(text="Я не знаю."):
    return {"renderer": "sleep_replay:query_unobserved_relation", "text": text,
            "speech_act": "brief_statement", "source_confidence": 0.5,
            "metadata": {"communicative_status": "express_limit",
                         "locally_composed": True, "authored_reply_template": False}}


async def hole(db, item, *, answer=None, kind="withheld"):
    await EventRepository(db).append(item)
    trace = ResponseTraceRepository(db)
    await trace.begin(item, prompt=str(item.payload["content"]))
    await trace.finish(item.event_id, status=kind, final_text=answer)


async def sleeping(db, replay=None):
    clock = [NOW]
    presence = AsyncMock()
    service = ContinuousCognitionService(db, clock=lambda: clock[0], presence=presence,
                                         replay=replay)
    await service.restore_sleep()
    clock[0] += timedelta(minutes=30)
    await service.rhythm_once()
    assert service.sleeping
    return service, clock, presence


async def test_thirty_minutes_cycles_channel_activity_and_restart(db):
    clock = [NOW]
    presence = AsyncMock()
    service = ContinuousCognitionService(db, clock=lambda: clock[0], presence=presence,
                                         activity_filter=AsyncMock(return_value=True))
    await service.restore_sleep()
    clock[0] += timedelta(minutes=29, seconds=59)
    assert await service.rhythm_once() == "awake"
    clock[0] += timedelta(seconds=1)
    assert await service.rhythm_once() == "sleeping"
    end = clock[0] + timedelta(minutes=90)
    clock[0] = end
    assert await service.rhythm_once() == "sleeping"
    assert (await db.fetchone("SELECT sleep_cycles FROM continuous_cognition_runtime"))[0] == 2
    await service.note_activity(event(bot_mentioned=False))
    assert service.sleeping
    restarted = ContinuousCognitionService(db, clock=lambda: clock[0], presence=presence)
    await restarted.restore_sleep()
    assert restarted.sleeping
    clock[0] += timedelta(minutes=90)
    assert await restarted.rhythm_once() == "awake"
    assert [c.args[0] for c in presence.await_args_list] == [
        "online", "idle", "idle", "idle", "online"]


async def test_irrelevant_import_bot_and_other_channel_do_not_reset_idle(db):
    clock = [NOW]
    service = ContinuousCognitionService(db, clock=lambda: clock[0],
                                         activity_filter=AsyncMock(return_value=False))
    await service.restore_sleep()
    clock[0] += timedelta(minutes=25)
    for item in (event(bot_mentioned=False), event(author_is_bot=True),
                 replace(event(), source="seed_corpus_toloka")):
        await service.note_activity(item)
    clock[0] += timedelta(minutes=5)
    assert await service.rhythm_once() == "sleeping"


async def test_tag_finishes_exactly_one_unit_and_delivers_persistent_job(db):
    entered, release = asyncio.Event(), asyncio.Event()

    async def replay(_event):
        entered.set()
        await release.wait()
        return (candidate(),)

    service, _, presence = await sleeping(db, replay)
    old = event()
    await hole(db, old)
    assert await service.capture_sleep_gap(old.event_id) == 1
    unit = asyncio.create_task(service.sleep_once())
    await entered.wait()
    incoming = event(2, "привет")
    await EventRepository(db).append(incoming)
    # Old withheld job has already been handled during the day.
    jobs = DeliveryRepository(db)
    await jobs.state(old.event_id, "no_reply")
    prepared = PreparedReply("Привет!", "sleep:test", ("Привет",), "genome", 1)

    async def prepare(item):
        await service.wait_until_awake(item)
        return prepared

    workflow = SimpleNamespace(prepare=prepare, validate=AsyncMock(return_value=prepared),
                               record_sent=AsyncMock())
    transport = SimpleNamespace(channel=AsyncMock(return_value=object()),
                                send=AsyncMock(return_value=SimpleNamespace(
                                    id=900, created_at=NOW)))
    delivery = ReplyDeliveryService(jobs, EventRepository(db), workflow, transport)
    answer = asyncio.create_task(delivery.run_once())
    await service.note_activity(incoming)
    await asyncio.sleep(0.02)
    assert service.sleeping and not answer.done()
    # Observation and a different async task remain live during the unit.
    assert await EventRepository(db).get(incoming.event_id)
    release.set()
    assert await unit == 1
    assert await answer
    transport.send.assert_awaited_once()
    assert (await jobs.get(incoming.event_id))["status"] == "completed"
    assert not service.sleeping
    assert presence.await_args_list[-1].args == ("online",)
    assert (await db.fetchone("SELECT COUNT(*) FROM continuous_cognition_hypotheses"))[0] == 1


async def test_restart_with_pending_tag_preserves_sleep_and_wakes_at_boundary(db):
    service, clock, _ = await sleeping(db)
    await service.note_activity(event())
    path = db.path
    await db.close()
    await db.initialize()
    assert db.path == path
    restarted = ContinuousCognitionService(db, clock=lambda: clock[0])
    await restarted.restore_sleep()
    assert restarted.sleeping
    await restarted.rhythm_once()
    assert not restarted.sleeping


async def test_holes_are_idempotent_prioritized_and_experiments_wait(db):
    replay = AsyncMock(return_value=(candidate(),))
    service, _, _ = await sleeping(db, replay)
    for n, text in ((1, "что не хочет?"), (2, "а что ты хочешь?"),
                    (3, "а ты предположи"), (4, "а ты предположи")):
        item = event(n, text)
        await hole(db, item, answer="Я не знаю, что хочу.", kind="sent")
        assert await service.capture_sleep_gap(item.event_id) == 1
        assert await service.capture_sleep_gap(item.event_id) == 0
    await service.sleep_once()
    assert replay.call_args.args[0].event_id == "sleep:4"
    assert not await db.fetchone("SELECT 1 FROM meaning_claims")
    assert not await db.fetchone("SELECT 1 FROM continuous_cognition_evidence")
    assert (await db.fetchone(
        "SELECT MAX(occurrence_count) FROM continuous_cognition_gaps"))[0] == 2


async def test_hypothesis_competes_is_scoped_and_rejection_prevents_sleep_retry(db):
    service, _, _ = await sleeping(db, AsyncMock(return_value=(candidate(),)))
    item = event()
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    assert await service.sleep_once() == 1
    recalled = await service.sleep_candidates(item, "а ты предположи")
    assert len(recalled) == 1
    for other in (replace(item, actor_id=4), replace(item, guild_id=8),
                  replace(item, channel_id=5)):
        assert not await service.sleep_candidates(other, "а ты предположи")
    old = ResponseCandidate.create(event_id=item.event_id, source="semantic_reasoning",
                                   renderer="old", text="Я не знаю, что я хочу.",
                                   speech_act="brief_statement", source_confidence=0.7)
    decision = ResponseArbitrator().decide(
        event_id=item.event_id, prompt="а ты предположи", focus_concepts=(),
        candidates=(old, *recalled), recent_replies=(), profile=ArbitrationProfile.baseline(1))
    assert len(decision.scores) == 2
    assert any(s.eligible and s.candidate.candidate_id == recalled[0].candidate_id
               for s in decision.scores)
    identifier = recalled[0].metadata["sleep_hypothesis_id"]
    await ResponseTraceRepository(db).step(
        item.event_id, stage="selection", component="response_arbitrator", status="selected",
        summary="offline selected sleep competitor", details={"sleep_hypothesis_id": identifier})
    sent = PreparedReply(recalled[0].text, "trial", (), "genome", 2,
                         winner_candidate_id=recalled[0].candidate_id)
    await service.record_sleep_trial(item, sent, 700)
    row = await db.fetchone("SELECT roles_json FROM continuous_cognition_hypotheses")
    assert json.loads(row[0])["receipts"] == [700]
    rejection = event(5, "не, ты чё", reply_to_message_id=700)
    await EventRepository(db).append(rejection)
    assert await service.observe_sleep_feedback(rejection) == 1
    assert await service.observe_sleep_feedback(rejection) == 0
    row = await db.fetchone("SELECT confidence,status,epistemic_status FROM "
                            "continuous_cognition_hypotheses WHERE hypothesis_id=?", (identifier,))
    assert row[0] < 0.5 and row[1] == "rejected" and row[2] == "hypothesis"
    assert not await service.sleep_candidates(item, "а ты предположи")
    await db.execute("UPDATE continuous_cognition_gaps SET occurrence_count=occurrence_count+1")
    assert await service.sleep_once() == 0
    assert (await db.fetchone(
        "SELECT status FROM continuous_cognition_hypotheses"))[0] == "rejected"
    assert not await db.fetchone("SELECT 1 FROM meaning_claims")


async def test_new_gap_preempts_smaller_experiment_budget_and_maintenance(db):
    maintenance = AsyncMock(return_value=True)
    service, _, _ = await sleeping(db, AsyncMock(return_value=(candidate(),)))
    service._maintenance = maintenance
    await service.sleep_once()
    assert (await db.fetchone("SELECT experiment_units FROM continuous_cognition_runtime"))[0] == 1
    item = event()
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    assert await service.sleep_once() == 1
    maintenance.assert_not_awaited()
    await service.sleep_once()
    await service.sleep_once()
    maintenance.assert_awaited_once()


async def test_repeated_day_approval_is_one_person_and_never_promotes_fact(db):
    service, _, _ = await sleeping(db, AsyncMock(return_value=(candidate(),)))
    item = event()
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    await service.sleep_once()
    stored = await db.fetchone("SELECT roles_json FROM continuous_cognition_hypotheses")
    data = json.loads(stored[0])
    data["receipts"] = [700]
    await db.execute("UPDATE continuous_cognition_hypotheses SET roles_json=?", (json.dumps(data),))
    for n in range(2, 8):
        approval = event(n, "да", reply_to_message_id=700)
        await EventRepository(db).append(approval)
        await service.observe_sleep_feedback(approval)
    row = await db.fetchone("SELECT positive_evidence,confidence,epistemic_status,status FROM "
                            "continuous_cognition_hypotheses")
    assert tuple(row) == (1, 0.65, "hypothesis", "working")
    assert not await db.fetchone("SELECT 1 FROM meaning_claims")


async def test_sleep_calls_existing_synonym_veto_and_keeps_zero_human_votes(db):
    from debilgi.language.lexicon import LexiconService

    item = event(1, "Вижу мишку")
    await EventRepository(db).append(item)
    for governor in ("видеть", "кормить"):
        for word in ("мишка", "медведь"):
            await db.execute("INSERT INTO language_generation_word_relations VALUES "
                             "(1,2,?,?,'direct_object','noun','accusative',3)", (governor, word))
    # No pending in-memory work: this also proves recovery after restart.
    service, _, _ = await sleeping(db)
    await service.sleep_once()
    row = await db.fetchone("SELECT * FROM continuous_cognition_hypotheses WHERE "
                            "predicate_key='synonym'")
    assert row and row["positive_evidence"] == row["negative_evidence"] == 0
    lexicon = LexiconService.for_meaning(db)
    assert not await lexicon.canonical_lemmas(1)
    assert not await lexicon.observe_synonym(item, "лыжа", "ложь", distributional=True)
    assert not await db.fetchone("SELECT 1 FROM meaning_claims")


async def test_forms_warmup_runs_in_worker_while_live_reader_keeps_projection(db, monkeypatch):
    import threading

    from debilgi.language import dialogue_learning as learning_module

    started, release = threading.Event(), threading.Event()
    learning = learning_module.DialogueStructureLearning(db, None)
    fetchall = db.fetchall

    async def rows(sql, parameters=()):
        if sql.startswith("SELECT DISTINCT guild_id FROM language_observed_exchanges"):
            return [(1,)]
        return await fetchall(sql, parameters)

    def slow_update(_self, _rows):
        started.set()
        assert release.wait(2)

    monkeypatch.setattr(db, "fetchall", rows)
    monkeypatch.setattr(learning_module._ExchangeRows, "update", slow_update)
    task = asyncio.create_task(learning.warm_analysis_cache())
    while not started.is_set():
        await asyncio.sleep(0.001)
    # This is a concurrent live reader publishing a newer projection.
    active = ["new live projection"]
    learning._exchange_rows[1] = active
    assert await db.fetchone("SELECT 1")
    release.set()
    assert await task
    assert learning._exchange_rows[1] is active


def test_sleep_vetoes_unbound_learned_parts_and_asserted_question_prefaces():
    item = event(1, "а что ты хочешь?")
    unbound = candidate("кто их не хочет?")
    unbound["metadata"]["communicative_status"] = "social"
    assert not ContinuousCognitionService.sleep_surface_allowed(item, unbound)
    asserted = candidate("Я хочу бегать. А ты?")
    asserted["metadata"]["communicative_status"] = "social"
    assert not ContinuousCognitionService.sleep_surface_allowed(item, asserted)
    prefaced = candidate("ну да, хочу) а ты?)")
    prefaced["metadata"]["communicative_status"] = "social"
    assert not ContinuousCognitionService.sleep_surface_allowed(item, prefaced)
    assert ContinuousCognitionService.sleep_surface_allowed(item, candidate())


async def test_restart_prepared_job_uses_real_workflow_wake_barrier(db):
    from debilgi.discord.reply_workflow import ReplyWorkflow

    service, clock, _ = await sleeping(db)
    item = event(1, "привет")
    await hole(db, item)
    prepared = PreparedReply("Привет!", "sleep:restart", (), "genome", 1)
    jobs = DeliveryRepository(db)
    await jobs.save_reply(item.event_id, prepared)
    await db.close()
    await db.initialize()
    restored = ContinuousCognitionService(db, clock=lambda: clock[0])
    await restored.restore_sleep()
    assert restored.sleeping
    context = SimpleNamespace(
        evaluate=AsyncMock(return_value=SimpleNamespace(
            allowed=True, text=prepared.text, reason="same_topic")),
        commit_outgoing=AsyncMock())
    bot = SimpleNamespace(continuous_cognition=restored, user=SimpleNamespace(id=9))
    workflow = ReplyWorkflow(bot, SimpleNamespace(), SimpleNamespace(record_sent=AsyncMock()),
                             context, ResponseTraceRepository(db))
    transport = SimpleNamespace(channel=AsyncMock(return_value=object()),
                                send=AsyncMock(return_value=SimpleNamespace(
                                    id=900, created_at=NOW)))
    delivery = ReplyDeliveryService(jobs, EventRepository(db), workflow, transport)
    assert await delivery.run_once()
    assert not restored.sleeping
    transport.send.assert_awaited_once()
    assert (await jobs.get(item.event_id))["status"] == "completed"


async def test_sleep_retires_only_obsolete_parser_cache_in_bounded_batches(db):
    from debilgi.language.dialogue_learning import DialogueStructureLearning

    item = event()
    await EventRepository(db).append(item)
    async with db.transaction() as connection:
        await connection.executemany(
            "INSERT INTO language_exchange_analysis VALUES (?,'obsolete','reaction',?,'null')",
            [(str(n), f"old:{n}") for n in range(300)])
    learning = DialogueStructureLearning(db, None)
    assert not await learning.warm_analysis_cache(max_exchanges=8)
    assert (await db.fetchone("SELECT COUNT(*) FROM language_exchange_analysis"))[0] == 44
    await learning.warm_analysis_cache(max_exchanges=8)
    assert await learning.warm_analysis_cache(max_exchanges=8)
    assert not await db.fetchone("SELECT 1 FROM language_exchange_analysis")
    assert await EventRepository(db).get(item.event_id) == item


async def test_sleep_analogy_can_use_vetoed_synonym_hypotheses_without_confirming(db):
    from debilgi.language.lexicon import LexiconService

    lexicon = LexiconService.for_meaning(db)
    item = event(70, "похожие слова")
    await EventRepository(db).append(item)
    await lexicon.observe_synonym(item, "мишка", "медведь", distributional=True)
    assert not await lexicon.canonical_lemmas(1)
    analogy = await lexicon.canonical_lemmas(1, for_hypothesis=True)
    assert analogy['мишка'] == analogy['медведь']
    # Even a stale, externally populated proposal must pass the same veto.
    for n, pair in enumerate((('читать', 'прочитать'), ('левый', 'правый'))):
        await db.execute(
            "INSERT INTO continuous_cognition_hypotheses "
            "(hypothesis_id,guild_id,signature,predicate_key,roles_json,epistemic_status,"
            "confidence,first_seen_at,last_reviewed_at) "
            "VALUES (?,1,?,'synonym',?,'hypothesis',.5,?,?)",
            (f'veto:{n}', f'veto:{n}', json.dumps(pair), NOW.isoformat(), NOW.isoformat()))
    await db.execute(
        "INSERT INTO language_synonym_links(guild_id,lemma,synonym_lemma,sense_id,source) "
        "VALUES (1,'левый','иной','sense-a','test'),(1,'правый','другой','sense-b','test')")
    analogy = await lexicon.canonical_lemmas(1, for_hypothesis=True)
    assert analogy.get('читать', 'читать') != analogy.get('прочитать', 'прочитать')
    assert analogy.get('левый', 'левый') != analogy.get('правый', 'правый')
    assert not await db.fetchone('SELECT 1 FROM continuous_cognition_evidence')
    assert not await db.fetchone('SELECT 1 FROM meaning_claims')


async def test_real_sleep_replay_does_not_write_agent_facts(db):
    from debilgi.config import Settings
    from debilgi.discord.bot import create_bot
    from debilgi.events.normalizer import normalize_message_created

    settings = Settings(discord_token=None, database_path=db.path,
                        migrations_path=Path('migrations'), test_replies_enabled=True)
    bot = create_bot(settings, db, EventRepository(db))
    try:
        await bot.pipeline.start()
        previous = None
        for n, text in enumerate(('Ты хочешь читать?', 'Хочу!', 'Наверное.')):
            item = normalize_message_created(
                message_id=400+n, guild_id=1, channel_id=2, author_id=30+n % 2,
                created_at=NOW-timedelta(minutes=10-n), content=text, author_is_bot=False,
                reply_to_message_id=previous, reply_to_author_id=30+(n-1) % 2 if n else None,
                bot_user_id=9)
            await bot.pipeline.publish(item)
            await bot.pipeline.join()
            previous = 400+n
        target = event(80, 'Ты хочешь плавать?')
        await hole(db, target, answer='Я не знаю, хочу ли я плавать.', kind='sent')
        before = [dict(r) for r in await db.fetchall(
            'SELECT * FROM meaning_claims ORDER BY claim_id')]
        variants = await bot.reply_service.replay_sleep_episode(target)
        assert any(v['metadata'].get('sleep_opinion') for v in variants)
        after = [dict(r) for r in await db.fetchall(
            'SELECT * FROM meaning_claims ORDER BY claim_id')]
        assert before == after
    finally:
        await bot.close()


async def test_rejected_meaning_cannot_return_with_another_surface(db):
    first, second = candidate('Я не знаю.'), candidate('Не знаю я.')
    first['metadata']['message_plan_key'] = second['metadata']['message_plan_key'] = 'same-query'
    replay = AsyncMock(return_value=(first, second))
    service, _, _ = await sleeping(db, replay)
    item = event(90)
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    assert await service.sleep_once() == 1
    await db.execute("UPDATE continuous_cognition_hypotheses SET status='rejected',confidence=.1")
    newer = event(91)
    await hole(db, newer)
    await service.capture_sleep_gap(newer.event_id)
    replay.return_value = (second,)
    assert await service.sleep_once() == 0
    assert not await service.sleep_candidates(newer, 'а ты предположи')
