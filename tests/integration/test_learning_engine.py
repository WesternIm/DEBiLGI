from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.expectations import (
    ExpectationPredictor,
    ExpectationStatus,
    OutcomeType,
)
from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.cognition.learning import LearningConfig, LearningEngine
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
BASE = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class ManualClock:
    def __init__(self) -> None:
        self.now = BASE

    def __call__(self) -> datetime:
        return self.now


def message(
    name: str,
    seconds: int,
    *,
    actor_id: int = 100,
    message_id: int = 1,
    reply_to: int | None = None,
) -> Event:
    event_type = EventType.MESSAGE_REPLIED if reply_to is not None else EventType.MESSAGE_CREATED
    return Event(
        event_id=name,
        guild_id=1,
        channel_id=10,
        actor_id=actor_id,
        target_id=None,
        event_type=event_type,
        timestamp=BASE + timedelta(seconds=seconds),
        payload={
            "message_id": message_id,
            "reply_to_message_id": reply_to,
            "content": "тест",
            "author_is_bot": False,
            "mentioned_user_ids": [],
        },
        source="test",
    )


def reaction(name: str, seconds: int, *, message_id: int = 1) -> Event:
    return Event(
        event_id=name,
        guild_id=1,
        channel_id=10,
        actor_id=200,
        target_id=100,
        event_type=EventType.REACTION_ADDED,
        timestamp=BASE + timedelta(seconds=seconds),
        payload={"message_id": message_id, "emoji": "💩"},
        source="test",
    )


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "learning.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def build_engine(database: Database, clock: ManualClock):
    event_repository = EventRepository(database)
    world_model = WorldModelRepository(database)
    memory = ShortTermMemory(MemoryLimits(retention=timedelta(hours=1)), clock=clock)
    extractor = BehaviorFeatureExtractor(FeatureConfig(lookback=timedelta(hours=1)))
    engine = LearningEngine(
        world_model,
        memory,
        extractor,
        ExpectationPredictor(),
        LearningConfig(horizon=timedelta(seconds=30), poll_interval=timedelta(seconds=1)),
        clock=clock,
    )
    return event_repository, world_model, memory, engine


async def process(
    event_repository: EventRepository,
    memory: ShortTermMemory,
    engine: LearningEngine,
    clock: ManualClock,
    item: Event,
) -> None:
    clock.now = item.utc_timestamp
    assert await event_repository.append(item)
    await memory.observe(item)
    await engine.observe(item)


async def test_message_creates_persisted_low_confidence_expectation(
    database: Database,
) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    trigger = message("trigger", 0)

    await process(event_repository, memory, engine, clock, trigger)

    expectation = await world_model.by_trigger_event(trigger.event_id)
    assert expectation is not None
    assert expectation.status is ExpectationStatus.PENDING
    assert expectation.prediction.outcome is OutcomeType.NO_RESPONSE
    assert expectation.prediction.probability == pytest.approx(0.2)


async def test_reply_resolves_expectation_and_records_difference(database: Database) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    trigger = message("trigger", 0, message_id=1)
    reply = message("reply", 10, actor_id=200, message_id=2, reply_to=1)

    await process(event_repository, memory, engine, clock, trigger)
    await process(event_repository, memory, engine, clock, reply)

    expectation = await world_model.by_trigger_event(trigger.event_id)
    assert expectation is not None
    assert expectation.status is ExpectationStatus.RESOLVED
    assert expectation.actual_outcome is OutcomeType.REPLY
    assert expectation.difference == pytest.approx(0.8)
    assert expectation.resolved_by_event_id == reply.event_id


async def test_reaction_resolves_matching_message_expectation(database: Database) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    trigger = message("trigger", 0, message_id=1)
    observed_reaction = reaction("reaction", 5, message_id=1)

    await process(event_repository, memory, engine, clock, trigger)
    await process(event_repository, memory, engine, clock, observed_reaction)

    expectation = await world_model.by_trigger_event(trigger.event_id)
    assert expectation is not None
    assert expectation.actual_outcome is OutcomeType.REACTION


async def test_same_actor_message_is_a_distinct_outcome(database: Database) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    first = message("first", 0, actor_id=100, message_id=1)
    second = message("second", 5, actor_id=100, message_id=2)

    await process(event_repository, memory, engine, clock, first)
    await process(event_repository, memory, engine, clock, second)

    expectation = await world_model.by_trigger_event(first.event_id)
    assert expectation is not None
    assert expectation.actual_outcome is OutcomeType.SAME_ACTOR_MESSAGE


async def test_other_participant_message_is_a_distinct_outcome(database: Database) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    first = message("first", 0, actor_id=100, message_id=1)
    second = message("second", 5, actor_id=200, message_id=2)

    await process(event_repository, memory, engine, clock, first)
    await process(event_repository, memory, engine, clock, second)

    expectation = await world_model.by_trigger_event(first.event_id)
    assert expectation is not None
    assert expectation.actual_outcome is OutcomeType.OTHER_MESSAGE


async def test_timeout_records_no_response_and_learns_pattern(database: Database) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    trigger = message("trigger", 0)
    await process(event_repository, memory, engine, clock, trigger)

    resolved_count = await engine.expire_overdue(BASE + timedelta(seconds=31))

    expectation = await world_model.by_trigger_event(trigger.event_id)
    stats = await world_model.stats(1)
    assert resolved_count == 1
    assert expectation is not None
    assert expectation.actual_outcome is OutcomeType.NO_RESPONSE
    assert stats.learned_patterns == 1


async def test_duplicate_observation_does_not_create_or_learn_twice(
    database: Database,
) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    trigger = message("trigger", 0)
    await process(event_repository, memory, engine, clock, trigger)

    assert not await engine.observe(trigger)
    stats = await world_model.stats(1)
    assert stats.pending_expectations == 1
    assert stats.resolved_expectations == 0


async def test_observed_outcome_changes_future_similar_prediction(
    database: Database,
) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    first = message("first", 0, message_id=1)
    observed_reaction = reaction("reaction", 5, message_id=1)
    similar_next = message("similar-next", 10, message_id=2)

    await process(event_repository, memory, engine, clock, first)
    await process(event_repository, memory, engine, clock, observed_reaction)
    await process(event_repository, memory, engine, clock, similar_next)

    expectation = await world_model.by_trigger_event(similar_next.event_id)
    assert expectation is not None
    assert expectation.prediction.outcome is OutcomeType.REACTION
    assert expectation.prediction.probability > 0.2
    assert expectation.prediction.evidence_support > 0


async def test_persisted_pending_expectation_expires_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "restart-learning.sqlite3"
    first_database = Database(db_path, MIGRATIONS)
    await first_database.initialize()
    first_clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(first_database, first_clock)
    trigger = message("trigger", 0)
    await process(event_repository, memory, engine, first_clock, trigger)
    await first_database.close()

    second_database = Database(db_path, MIGRATIONS)
    await second_database.initialize()
    second_clock = ManualClock()
    second_clock.now = BASE + timedelta(seconds=31)
    _, restarted_model, _, restarted_engine = build_engine(second_database, second_clock)
    await restarted_engine.start()
    try:
        expectation = await restarted_model.by_trigger_event(trigger.event_id)
        assert expectation is not None
        assert expectation.actual_outcome is OutcomeType.NO_RESPONSE
    finally:
        await restarted_engine.stop()
        await second_database.close()


async def test_fast_chat_can_hold_and_resolve_multiple_expectations(
    database: Database,
) -> None:
    clock = ManualClock()
    event_repository, world_model, memory, engine = build_engine(database, clock)
    first = message("first", 0, actor_id=100, message_id=1)
    simultaneous = message("simultaneous", 0, actor_id=200, message_id=2)
    later = message("later", 1, actor_id=300, message_id=3)

    await process(event_repository, memory, engine, clock, first)
    await process(event_repository, memory, engine, clock, simultaneous)
    assert (await world_model.stats(1)).pending_expectations == 2

    await process(event_repository, memory, engine, clock, later)
    first_result = await world_model.by_trigger_event(first.event_id)
    second_result = await world_model.by_trigger_event(simultaneous.event_id)
    assert first_result is not None and first_result.status is ExpectationStatus.RESOLVED
    assert second_result is not None and second_result.status is ExpectationStatus.RESOLVED
    assert (await world_model.stats(1)).pending_expectations == 1
