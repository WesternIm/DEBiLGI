from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.discord.delivery_repository import decode_reply, encode_reply
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.language.assertion_experience import (
    AssertionExperienceRepository,
    ExperienceGuidance,
    experience_proposal,
    explicit_feedback,
)
from debilgi.language.assertion_memory import build_assertion_memory, plan_assertion_continuation
from debilgi.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
NOW = datetime(2026, 9, 9, tzinfo=UTC)


def message(text, index=0, *, bot=False, reference=None):
    return Event(
        event_id=f"event:{index}",
        guild_id=1,
        channel_id=10,
        actor_id=123 if bot else 100,
        target_id=100 if bot else 123,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW + timedelta(seconds=index),
        payload={
            "content": text,
            "author_is_bot": bot,
            "bot_user_id": 123,
            "reply_to_message_id": reference,
        },
        source="discord",
    )


def scenario(index=0, *, noun="дверь"):
    anchor = message(f"Я пока не открыл {noun}.", index, bot=True)
    prompt = message("Ну всё, время настало", index + 1)
    history = (anchor,)
    plan = plan_assertion_continuation(
        prompt.payload["content"], build_assertion_memory(history, prompt), history, prompt
    )
    assert plan is not None
    return anchor, prompt, plan, experience_proposal(prompt.payload["content"], plan)


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "experience.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


async def save_episode(db, index=0):
    anchor, prompt, plan, proposal = scenario(index)
    events = EventRepository(db)
    await events.append(anchor)
    await events.append(prompt)
    repository = AssertionExperienceRepository(db)
    await repository.record_sent(
        prompt,
        response_message_id=1000 + index,
        proposal=proposal,
        created_at=prompt.utc_timestamp + timedelta(seconds=1),
    )
    return repository, prompt, plan, proposal


async def feedback(db, text, index=3, *, reference=1000, **changes):
    event = replace(message(text, index, reference=reference), **changes)
    await EventRepository(db).append(event)
    await AssertionExperienceRepository(db).observe_feedback(event)
    return event


async def test_votes_are_once_per_sent_question_and_survive_restart(db):
    repository, prompt, plan, proposal = await save_episode(db)
    vote = await feedback(db, "Да, именно это")
    await repository.observe_feedback(vote)
    await feedback(db, "да", 4)
    await repository.record_sent(
        prompt,
        response_message_id=1000,
        proposal=proposal,
        created_at=prompt.utc_timestamp + timedelta(seconds=1),
    )
    future = message("Ну всё, время настало", 100)
    assert await repository.guidance(future, proposal) == ExperienceGuidance(1, 0)
    adapted, _ = await repository.adapt(future, future.payload["content"], plan)
    assert adapted.confidence > plan.confidence
    assert adapted.thought.clause_kind == "question"
    await db.close()
    await db.initialize()
    assert await AssertionExperienceRepository(db).guidance(future, proposal) == ExperienceGuidance(
        1, 0
    )


async def test_rejected_interpretation_changes_future_selection_and_expires(db):
    repository, _, plan, proposal = await save_episode(db)
    await feedback(db, "нет")
    await save_episode(db, 10)
    await feedback(db, "нет, не это", 13, reference=1010)
    future = message("Ну всё, время настало", 100)
    adapted, guidance = await repository.adapt(future, future.payload["content"], plan)
    assert adapted is None and guidance.suppressed
    assert guidance == ExperienceGuidance(0, 2)
    later = replace(future, timestamp=NOW + timedelta(days=31))
    restored, guidance = await repository.adapt(later, later.payload["content"], plan)
    assert restored is not None and guidance == ExperienceGuidance()


@pytest.mark.parametrize(
    "changes",
    [
        {"guild_id": 2},
        {"channel_id": 11},
        {"actor_id": 999},
        {"source": "archive"},
        {"timestamp": NOW + timedelta(minutes=6)},
        {"timestamp": NOW - timedelta(seconds=1)},
        {"payload": {"content": "да", "reply_to_message_id": 1000, "author_is_bot": True}},
    ],
)
async def test_feedback_requires_the_original_user_scope_and_time(db, changes):
    repository, _, _, proposal = await save_episode(db)
    await feedback(db, "да", **changes)
    assert await repository.guidance(message("А теперь?", 100), proposal) == ExperienceGuidance()


@pytest.mark.parametrize(
    "text,reference",
    [("да", None), ("да", 9999), ("да?", 1000), ("да, но я имел в виду другое", 1000)],
)
async def test_ambiguous_unbound_feedback_does_not_train(db, text, reference):
    repository, _, _, proposal = await save_episode(db)
    await feedback(db, text, reference=reference)
    assert await repository.guidance(message("А теперь?", 100), proposal) == ExperienceGuidance()


async def test_feedback_arriving_before_send_bookkeeping_is_reconciled(db):
    anchor, prompt, _, proposal = scenario()
    events = EventRepository(db)
    await events.append(anchor)
    await events.append(prompt)
    await feedback(db, "да")
    assert not await db.fetchall("SELECT * FROM assertion_feedback")
    repository = AssertionExperienceRepository(db)
    await repository.record_sent(
        prompt, response_message_id=1000, proposal=proposal, created_at=NOW + timedelta(seconds=2)
    )
    assert await repository.guidance(message("А теперь?", 100), proposal) == ExperienceGuidance(
        1, 0
    )


async def test_experience_generalizes_objects_but_not_other_users_or_cues(db):
    repository, _, _, proposal = await save_episode(db)
    await feedback(db, "да")
    _, _, window_plan, window_proposal = scenario(10, noun="окно")
    assert proposal["learning_key"] == window_proposal["learning_key"]
    assert await repository.guidance(message("...", 100), window_proposal) == ExperienceGuidance(
        1, 0
    )
    other_user = replace(message("...", 100), actor_id=555)
    assert await repository.guidance(other_user, proposal) == ExperienceGuidance()
    other_cue = experience_proposal("А теперь?", window_plan)
    assert await repository.guidance(message("...", 100), other_cue) == ExperienceGuidance()
    # Future observations must not influence historical/offline replay.
    assert await repository.guidance(message("...", 2), proposal) == ExperienceGuidance()


def test_negative_question_requires_explicit_interpretation_feedback():
    assert explicit_feedback("да", negated_question=True) is None
    assert explicit_feedback("нет", negated_question=True) is None
    assert explicit_feedback("да, именно это", negated_question=True) is True
    assert explicit_feedback("нет, не это", negated_question=True) is False
    assert explicit_feedback("да 😂", negated_question=False) is None


def test_prepared_hypothesis_metadata_survives_outbox_serialization():
    _, _, _, proposal = scenario()
    prepared = PreparedReply("Я открыл дверь?", "i", (), "g", 1, assertion_experience=proposal)
    assert decode_reply(encode_reply(prepared)) == prepared
    assert "дверь" not in proposal["learning_key"]


def test_generalization_preserves_who_performs_the_action():
    _, prompt, plan, original = scenario()
    subject = plan.thought.dependencies[0]
    changed_subject = replace(
        subject,
        argument=replace(
            subject.argument,
            head=replace(subject.argument.head, lemma="ты", surface="ты", entity_kind="user:100"),
        ),
    )
    other = replace(
        plan,
        thought=replace(
            plan.thought, dependencies=(changed_subject, *plan.thought.dependencies[1:])
        ),
    )
    assert (
        experience_proposal(prompt.payload["content"], other)["learning_key"]
        != original["learning_key"]
    )
