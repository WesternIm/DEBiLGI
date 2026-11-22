"""Receipt/source grounded self memory; no imported persona or recursive claims."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.experience.repository import ExperienceRepository
from debilgi.language.canonical_claims import Known, Unknown, activity_thought
from debilgi.language.generative_planner import plan_from_evidence
from debilgi.language.planning import ReplyContext
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)


def message(index, text="я читаю книгу.", **changes):
    event = normalize_message_created(
        message_id=index,
        guild_id=1,
        channel_id=2,
        author_id=3,
        created_at=NOW + timedelta(seconds=index),
        content=text,
        author_is_bot=False,
        bot_user_id=99,
        bot_mentioned=True,
    )
    return replace(event, **changes)


@pytest.fixture
async def memory(tmp_path):
    database = Database(tmp_path / "self.sqlite3", Path(__file__).parents[2] / "migrations")
    await database.initialize()
    yield PropositionMemory(database), database
    await database.close()


async def append(db, event):
    await EventRepository(db).append(event)
    return event


async def test_twenty_operations_sources_statuses_and_replay(memory):
    memory, db = memory
    # Exactly twenty input operations: five delivery states, four runtime states,
    # four search states, two processing states, two corpus operations, two
    # independent receipts without meaning intentions, and a generated assertion.
    sources = [await append(db, message(i)) for i in range(1, 21)]
    statuses = ("prepared", "sending", "uncertain", "sent", "completed")
    for event, status in zip(sources[:5], statuses, strict=True):
        await db.execute(
            """UPDATE reply_delivery_jobs SET status = ?, response_message_id = ?, sent_at = ?
               WHERE event_id = ?""",
            (status, 1000 + event.payload["message_id"], NOW.isoformat(), event.event_id),
        )
    for event, status in zip(
        sources[5:9],
        ("queued", "processing", "completed", "failed"),
        strict=True,
    ):
        await db.execute(
            """INSERT INTO continuous_cognition_work_items
               (work_id, source_event_id, guild_id, channel_id, status,
                next_attempt_at, created_at, updated_at, completed_at)
               VALUES (?, ?, 1, 2, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.event_id,
                status,
                NOW.isoformat(),
                NOW.isoformat(),
                (NOW + timedelta(seconds=100)).isoformat(),
                (NOW + timedelta(seconds=100)).isoformat() if status == "completed" else None,
            ),
        )
    for event, status in zip(
        sources[9:13],
        ("pending", "resolved", "no_result", "failed"),
        strict=True,
    ):
        await db.execute(
            """INSERT INTO knowledge_queries
               (query_id, guild_id, channel_id, event_id, source, language, query_text,
                query_key, reason, uncertainty, status, created_at, completed_at)
               VALUES (?, 1, 2, ?, 'test', 'ru', 'грамматика', ?, 'explicit', 0, ?, ?, ?)""",
            (
                event.event_id,
                event.event_id,
                event.event_id,
                status,
                NOW.isoformat(),
                (NOW + timedelta(seconds=100)).isoformat() if status != "pending" else None,
            ),
        )
    for event, status in zip(sources[13:15], ("processed", "processing"), strict=True):
        await db.execute(
            "UPDATE event_processing SET status = ?, completed_at = ? WHERE event_id = ?",
            (status, (NOW + timedelta(seconds=100)).isoformat(), event.event_id),
        )
    for event in sources[15:17]:
        await db.execute(
            "UPDATE events SET source = 'seed_corpus_toloka' WHERE event_id = ?", (event.event_id,)
        )
        await db.execute(
            "UPDATE event_processing SET status = 'processed', completed_at = ? WHERE event_id = ?",
            (NOW.isoformat(), event.event_id),
        )
    for event in sources[17:19]:
        # The same entry point as greetings and arbitrary non-meaning sends.
        await memory.record_agent_activity(
            event,
            source_key=f"receipt:2:{2000 + event.payload['message_id']}",
            operation="send",
            thought=activity_thought("отправить", "сообщение"),
            occurred_at=NOW,
            receipt_id=2000 + event.payload["message_id"],
        )
    await memory.observe(sources[19])  # Generated/report content is not an operation.
    question = message(120, "что ты сегодня делал?")
    await memory.project_agent_sources(question)
    activities = await db.fetchall("SELECT * FROM agent_activities ORDER BY activity_id")
    assert len(activities) == 9  # 4 sends + 2 work + 2 search + 1 processing
    assert {r["source_event_id"] for r in activities} == {
        sources[i - 1].event_id for i in (4, 5, 7, 8, 11, 12, 14, 18, 19)
    }
    result = await memory.retrieve_claims(question)
    assert isinstance(result, Known) and len(result.claims) == 8
    operation_evidence = await db.fetchall(
        "SELECT x.meaning_key,x.actor_id FROM meaning_evidence x JOIN meaning_claims c "
        "ON c.claim_id=x.claim_id WHERE c.source_kind='agent_observed'"
    )
    assert operation_evidence and all(
        r["meaning_key"] and r["actor_id"] is None for r in operation_evidence
    )
    for claim in result.claims:
        assert claim.subject_ref == "discord:agent:99" and claim.source_kind == "agent_observed"
        evidence = await db.fetchone(
            "SELECT * FROM meaning_evidence WHERE claim_id = ?", (claim.claim_id,)
        )
        assert evidence["source_event_id"] == claim.source_event_id
        assert evidence["source_key"] in claim.thought.evidence
        assert claim.proof.source_event_ids == (claim.source_event_id,)
    active = await memory.retrieve_claims(message(120, "что ты делаешь?"))
    assert isinstance(active, Known) and len(active.claims) == 1
    expired = await memory.retrieve_claims(message(200, "что ты делаешь?"))
    assert isinstance(expired, Unknown)
    assert isinstance(await memory.retrieve_claims(message(120, "какие книги ты читал?")), Unknown)
    assert isinstance(await memory.retrieve_claims(message(120, "ты где?")), Unknown)
    model = await ExperienceRepository(db).self_model(1)
    assert set(model.claim_ids) == {r["claim_id"] for r in activities}
    before = [tuple(r) for r in activities]
    for _ in range(2):
        await PropositionMemory(db).project_agent_sources(question)
    await db.close()
    await db.initialize()
    await PropositionMemory(db).project_agent_sources(question)
    assert [
        tuple(r) for r in await db.fetchall("SELECT * FROM agent_activities ORDER BY activity_id")
    ] == before
    # Completion closes the active interval. Replaying processing cannot reopen it.
    await db.execute(
        """UPDATE continuous_cognition_work_items SET status = 'completed', completed_at = ?
           WHERE source_event_id = ?""",
        ((NOW + timedelta(seconds=125)).isoformat(), sources[6].event_id),
    )
    assert isinstance(await memory.retrieve_claims(message(130, "что ты делаешь?")), Unknown)
    await memory.record_agent_activity(
        sources[6],
        source_key=f"continuous_cognition_work_items:{sources[6].event_id}",
        operation="work",
        thought=activity_thought("обрабатывать", "сообщение"),
        occurred_at=NOW + timedelta(seconds=129),
        status="active",
    )
    assert isinstance(await memory.retrieve_claims(message(130, "что ты делаешь?")), Unknown)


async def test_identity_is_canonical_and_corpus_never_supplies_self(memory):
    memory, db = memory
    corpus = message(1, "я на море.", source="seed_corpus_toloka")
    await append(db, corpus)
    await memory.observe(corpus)
    query = message(3, "ты где?")
    result = await memory.retrieve_claims(query)
    assert isinstance(result, Unknown)
    plan = plan_from_evidence(
        ReplyContext(
            prompt=query.payload["content"],
            input_meaning=query.meaning,
            claim_result=result,
            corpus_utterances=((corpus.event_id, "Я на море."),),
            knowledge_facts=("Я на море.",),
        )
    )
    assert len(plan.proposals) == 1
    assert plan.proposals[0].operation == "express_query_unknown"
    assert plan.proposals[0].plan.knowledge_limit.query == result.query
    assert not await db.fetchall("SELECT * FROM agent_activities")
    assert not await db.fetchall("SELECT * FROM agent_preferences")
    named = message(4, "как тебя зовут?")
    named = replace(
        named,
        payload={
            **named.payload,
            "bot_identity": {"id": 99, "username": "Гоша"},
        },
    )
    await append(db, named)
    result = await memory.retrieve_claims(named)
    assert isinstance(result, Known) and result.claims[0].predicate == "называться"
    assert result.claims[0].thought.dependencies[-1].argument.head.lemma == "Гоша"
    plans = plan_from_evidence(
        ReplyContext(
            prompt=named.payload["content"],
            input_meaning=named.meaning,
            claim_result=result,
        )
    )
    assert plans.proposals[0].operation == "retrieve_self_claim"
    assert result.claims[0].claim_id in plans.proposals[0].plan.claim.evidence


async def test_send_entry_point_records_greeting_without_old_intention(memory):
    memory, db = memory
    from unittest.mock import AsyncMock

    from debilgi.discord.replies import SafeMentionReplyService

    event = await append(db, message(1, "Привет!"))
    service = SafeMentionReplyService.__new__(SafeMentionReplyService)
    service._context_character_limit = 1000
    service._proposition_memory = memory
    service._dialogue_tasks = service._meaning_intents = service._assertion_experiences = None
    service._processing_outcomes = service._open_questions = service._self_dataset = None
    service._pending_positions = {}
    service._experience_consolidator = AsyncMock()
    reply = PreparedReply("Привет.", "greet", (), "", 1)
    for _ in range(2):
        await service.record_sent(event, response_message_id=777, reply=reply, created_at=NOW)
    assert reply.meaning_intention_id is None
    activities = await db.fetchall("SELECT * FROM agent_activities")
    assert len(activities) == 1 and activities[0]["receipt_id"] == 777


async def test_form_learning_requires_actual_live_evidence(memory):
    memory, db = memory
    from debilgi.dataset.self_dataset import SelfDatasetRepository
    from debilgi.language.semantic_assimilation import SemanticAssimilationService

    service = SemanticAssimilationService(db, SelfDatasetRepository(db))
    for index in range(1, 4):
        event = await append(db, message(index))
        await service.observe_local_structure(
            guild_id=1,
            structure=event.meaning,
            source_event_id=event.event_id,
            observed_at=event.utc_timestamp,
        )
    await memory.project_agent_sources(message(10, "что ты сегодня делал?"))
    rows = await db.fetchall("SELECT * FROM agent_activities WHERE operation = 'learn'")
    assert rows
    for row in rows:
        evidence = await db.fetchone(
            "SELECT * FROM language_semantic_lesson_evidence WHERE source_event_id = ?",
            (row["source_event_id"],),
        )
        assert evidence and evidence["input_hash"] in row["source_key"]
    assert not await db.fetchall("SELECT * FROM agent_preferences")


async def test_preferences_require_repeated_same_action_receipts_and_owned_feedback(memory):
    memory, db = memory
    from debilgi.experience.meaning_intents import MeaningIntentRepository
    from debilgi.language.generative_planner import objective_for

    repo = MeaningIntentRepository(db)
    # A concrete operation about one subject. Other subjects, people and unsent
    # plans must never be pooled into the estimate for this action.
    from debilgi.language.communication import MessagePlan, reader_goal
    from debilgi.language.generative_planner import DerivedPlan

    async def experience(index, *, target="сообщение", user=3, sent=True, corpus=False):
        prompt = await append(db, message(index, "расскажи", actor_id=user))
        thought = activity_thought("отправить", target)
        plan = DerivedPlan(MessagePlan(reader_goal(prompt.meaning), thought), "send", ())
        selection = await repo.choose(
            prompt, (plan,), objective_for(prompt.meaning), parent_intent_id="test"
        )
        if sent:
            await repo.record_sent(
                prompt,
                intention_id=selection.intention_id,
                response_message_id=1000 + index,
                created_at=prompt.utc_timestamp + timedelta(seconds=1),
            )
            await memory.record_agent_activity(
                prompt,
                source_key=f"receipt:2:{1000 + index}",
                operation="send",
                thought=thought,
                occurred_at=prompt.utc_timestamp + timedelta(seconds=1),
                receipt_id=1000 + index,
            )
        feedback = message(index + 1, "Отлично.", actor_id=user)
        feedback = replace(
            feedback,
            payload={**feedback.payload, "reply_to_message_id": 1000 + index},
            timestamp=prompt.utc_timestamp + timedelta(seconds=2),
            source="seed_corpus_toloka" if corpus else "discord",
        )
        await append(db, feedback)
        await repo.observe(feedback)

    await experience(1)
    assert not await db.fetchall("SELECT * FROM agent_preferences")
    await experience(4, target="журнал")
    await experience(7, user=4)
    await experience(10, sent=False)
    await experience(13, corpus=True)
    assert not await db.fetchall("SELECT * FROM agent_preferences")
    await experience(16)
    rows = await db.fetchall("SELECT * FROM agent_preferences")
    assert len(rows) == 1 and rows[0]["support"] == 2 and rows[0]["estimate"] == 1
    assert set(rows[0]["evidence_ids"].split(",")) == {message(2).event_id, message(17).event_id}
    assert len(await repo.learned_goals(message(20))) == 1
    assert not await repo.learned_goals(message(20, actor_id=5))


@pytest.mark.parametrize("status", ["queued", "failed"])
async def test_runtime_retry_or_failure_closes_active_claim(memory, status):
    memory, db = memory
    source = await append(db, message(1))
    await db.execute(
        """INSERT INTO continuous_cognition_work_items
           (work_id, source_event_id, guild_id, channel_id, status,
            next_attempt_at, created_at, updated_at)
           VALUES ('work', ?, 1, 2, 'processing', ?, ?, ?)""",
        (source.event_id, NOW.isoformat(), NOW.isoformat(), NOW.isoformat()),
    )
    assert isinstance(await memory.retrieve_claims(message(2, "что ты делаешь?")), Known)
    await db.execute(
        "UPDATE continuous_cognition_work_items SET status = ?, updated_at = ?",
        (status, (NOW + timedelta(seconds=3)).isoformat()),
    )
    assert isinstance(await memory.retrieve_claims(message(4, "что ты делаешь?")), Unknown)
    await db.execute(
        "UPDATE continuous_cognition_work_items SET status = 'processing', updated_at = ?",
        ((NOW + timedelta(seconds=5)).isoformat(),),
    )
    assert isinstance(await memory.retrieve_claims(message(6, "что ты делаешь?")), Known)
    assert len(await db.fetchall("SELECT * FROM agent_activities")) == 1


async def test_received_book_extract_is_not_reading_a_book(memory):
    memory, db = memory
    source = await append(db, message(1))
    await db.execute(
        """INSERT INTO knowledge_concepts
           (concept_id, source, language, external_id, title, normalized_title, definition,
            canonical_url, confidence, fetched_at, updated_at)
           VALUES ('book', 'test', 'ru', '1', 'Книга', 'книга', 'отрывок', 'test:1', 1, ?, ?)""",
        (NOW.isoformat(), NOW.isoformat()),
    )
    await db.execute(
        """INSERT INTO knowledge_queries
           (query_id, guild_id, channel_id, event_id, source, language, query_text,
            query_key, reason, uncertainty, status, concept_id, created_at, completed_at)
           VALUES ('query', 1, 2, ?, 'test', 'ru', 'книга', 'книга', 'explicit', 0,
                   'resolved', 'book', ?, ?)""",
        (source.event_id, NOW.isoformat(), NOW.isoformat()),
    )
    received = await memory.retrieve_claims(message(3, "что ты получил?"))
    assert isinstance(received, Known) and received.claims[0].predicate == "получить"
    assert "knowledge_queries:query:receive" in received.claims[0].thought.evidence
    assert isinstance(await memory.retrieve_claims(message(3, "какие книги ты читал?")), Unknown)


async def test_identity_changes_and_replay_do_not_leave_two_current_names(memory):
    memory, db = memory
    for index, name in ((1, "Гоша"), (3, "Саша"), (5, "Гоша")):
        event = message(index, "как тебя зовут?")
        event = replace(
            event,
            payload={
                **event.payload,
                "bot_identity": {"id": 99, "username": name},
            },
        )
        await append(db, event)
        for _ in range(2):
            result = await memory.retrieve_claims(event)
            assert len(result.claims) == 1
            assert result.claims[0].thought.dependencies[-1].argument.head.lemma == name
    assert len(await db.fetchall("SELECT * FROM agent_activities")) == 3
    result = await memory.retrieve_claims(message(4, "как тебя зовут?"))
    assert len(result.claims) == 1
    assert result.claims[0].thought.dependencies[-1].argument.head.lemma == "Саша"
    # A name supplied in another channel needs its own scoped proof; it cannot
    # disappear because the same agent already had a name elsewhere.
    other = message(10, "как тебя зовут?", channel_id=4)
    other = replace(
        other,
        payload={
            **other.payload,
            "bot_identity": {"id": 99, "username": "Гоша"},
        },
    )
    await append(db, other)
    assert len((await memory.retrieve_claims(other)).claims) == 1
    assert len((await memory.retrieve_claims(message(11, "как тебя зовут?"))).claims) == 1


def test_second_person_object_is_not_agent_subject():
    from debilgi.language.canonical_claims import query_from_event, self_query

    query = query_from_event(message(1, "что я тебе отправил?"))
    assert not self_query(query)
    assert ("subject", "discord:user:3") in query.arguments


async def test_projection_reaches_receipts_older_than_first_source_batch(memory):
    memory, db = memory
    source = await append(db, message(1))
    await db.execute(
        """UPDATE reply_delivery_jobs SET status = 'sent', response_message_id = 777,
           sent_at = ? WHERE event_id = ?""",
        (NOW.isoformat(), source.event_id),
    )
    for index in range(2, 515):
        await append(db, message(index))
    result = await memory.retrieve_claims(message(600, "что ты сегодня делал?"))
    assert len(result.claims) == 1
    assert result.claims[0].thought.evidence == ("receipt:2:777",)
