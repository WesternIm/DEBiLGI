from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.core import CognitiveCore, CognitiveCoreConfig
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.expectations import ExpectationPredictor
from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.cognition.learning import LearningConfig, LearningEngine
from debilgi.cognition.models import EntityType
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.simulation import replay_events
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
    guild_id: int = 1,
    actor_id: int = 100,
    message_id: int | None = None,
    reply_to: int | None = None,
) -> Event:
    return Event(
        event_id=name,
        guild_id=guild_id,
        channel_id=10,
        actor_id=actor_id,
        target_id=None,
        event_type=(
            EventType.MESSAGE_REPLIED if reply_to is not None else EventType.MESSAGE_CREATED
        ),
        timestamp=BASE + timedelta(seconds=seconds),
        payload={
            "message_id": message_id if message_id is not None else seconds + 1,
            "reply_to_message_id": reply_to,
            "content": f"message {name}",
            "author_is_bot": False,
            "mentioned_user_ids": [],
        },
        source="test",
    )


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "cognitive.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def build_stack(database: Database, clock: ManualClock):
    events = EventRepository(database)
    memory = ShortTermMemory(MemoryLimits(retention=timedelta(days=1)), clock=clock)
    world = WorldModelRepository(database)
    learning = LearningEngine(
        world,
        memory,
        BehaviorFeatureExtractor(FeatureConfig(lookback=timedelta(days=1))),
        ExpectationPredictor(),
        LearningConfig(horizon=timedelta(seconds=30), poll_interval=timedelta(seconds=1)),
        clock=clock,
    )
    cognitive_repository = CognitiveRepository(database)
    core = CognitiveCore(
        cognitive_repository,
        world,
        config=CognitiveCoreConfig(
            episode_gap=timedelta(seconds=10),
            maintenance_interval=timedelta(seconds=1),
            raw_text_retention=timedelta(seconds=1),
            consolidation_batch_size=100,
        ),
        clock=clock,
    )
    return events, memory, world, learning, cognitive_repository, core


async def process(stack: tuple, clock: ManualClock, event: Event) -> None:
    events, memory, _, learning, _, core = stack
    clock.now = event.utc_timestamp
    assert await events.append(event)
    await memory.observe(event)
    await learning.observe(event)
    await core.observe(event)


async def test_core_processing_is_idempotent_and_persistent(tmp_path: Path) -> None:
    path = tmp_path / "restart.sqlite3"
    first_db = Database(path, MIGRATIONS)
    await first_db.initialize()
    clock = ManualClock()
    stack = build_stack(first_db, clock)
    event = message("one", 0)
    await process(stack, clock, event)
    repository = stack[4]
    core = stack[5]

    assert not await core.observe(event)
    model = await repository.entity_model(1, EntityType.USER, 100)
    assert model is not None and model.observations == 1
    assert len(await repository.recent_intents(1)) == 1
    await first_db.close()

    second_db = Database(path, MIGRATIONS)
    await second_db.initialize()
    restored = CognitiveRepository(second_db)
    try:
        model = await restored.entity_model(1, EntityType.USER, 100)
        assert model is not None and model.observations == 1
        assert len(await restored.recent_episodes(1)) == 1
        assert await restored.internal_state(1, 100) is not None
        assert (await restored.recent_intents(1))[0].shadow_only
    finally:
        await second_db.close()


async def test_difference_episode_causal_trace_and_raw_retention(
    database: Database,
) -> None:
    clock = ManualClock()
    stack = build_stack(database, clock)
    first = message("first", 0, message_id=1)
    reply = message("reply", 5, actor_id=200, message_id=2, reply_to=1)

    await process(stack, clock, first)
    await process(stack, clock, reply)

    events, _, _, _, repository, core = stack
    differences = await repository.recent_differences(1)
    episodes = await repository.recent_episodes(1)
    intents = await repository.recent_intents(1)
    intent = intents[0]
    assert differences[0]["scalar_d"] == pytest.approx(0.8)
    assert differences[0]["outcome_mismatch"] == 1.0
    assert episodes[0].event_ids == (first.event_id, reply.event_id)
    assert episodes[0].observed_outcome == "reply"
    assert episodes[0].importance > 0.1
    assert [step["stage"] for step in intent.reasoning_trace] == [
        "observation",
        "genome",
        "expectation_context",
        "mismatch_state",
        "decision",
    ]
    assert intent.executed_at is None and intent.shadow_only
    assert intents[1].actual_outcome is None
    assert intents[1].evaluated_at is None

    await core.maintenance_once(reply.utc_timestamp + timedelta(seconds=2))
    restored = await events.get(first.event_id)
    assert restored is not None
    assert "content" not in restored.payload
    assert restored.payload["message_id"] == 1


async def test_repeated_episodes_strengthen_then_contradiction_weakens_memory(
    database: Database,
) -> None:
    clock = ManualClock()
    stack = build_stack(database, clock)
    repository = stack[4]
    first_batch = [
        message("a1", 0, message_id=1),
        message("a2", 1, message_id=2),
        message("b1", 20, message_id=3),
        message("b2", 21, message_id=4),
    ]
    for event in first_batch:
        await process(stack, clock, event)
    await repository.close_stale_episodes(BASE + timedelta(seconds=35), timedelta(seconds=10))
    assert await repository.consolidate(guild_id=1) == 2
    strengthened = (
        await repository.semantic_memories(1, scope_type="user", scope_id=100, limit=1)
    )[0]
    assert strengthened.confirmations == 2

    await process(stack, clock, message("contradiction", 40, message_id=5))
    await repository.close_stale_episodes(BASE + timedelta(seconds=60), timedelta(seconds=10))
    assert await repository.consolidate(guild_id=1) == 1
    weakened = (await repository.semantic_memories(1, scope_type="user", scope_id=100, limit=1))[0]
    assert weakened.contradictions == 1
    assert weakened.confidence < strengthened.confidence

    await repository.decay_semantic(BASE + timedelta(days=30))
    decayed = (await repository.semantic_memories(1, scope_type="user", scope_id=100, limit=1))[0]
    assert decayed.confidence < weakened.confidence


async def test_guild_models_and_semantic_memory_are_isolated(database: Database) -> None:
    clock = ManualClock()
    stack = build_stack(database, clock)
    for event in (
        message("g1-a", 0, guild_id=1, message_id=1),
        message("g1-b", 1, guild_id=1, message_id=2),
        message("g2-a", 20, guild_id=2, message_id=3),
    ):
        await process(stack, clock, event)
    repository = stack[4]
    await repository.close_stale_episodes(BASE + timedelta(seconds=40), timedelta(seconds=10))
    await repository.consolidate()

    assert await repository.entity_model(1, EntityType.USER, 100) is not None
    assert await repository.entity_model(2, EntityType.USER, 100) is not None
    assert all(item.guild_id == 1 for item in await repository.semantic_memories(1))
    assert all(item.guild_id == 2 for item in await repository.semantic_memories(2))


async def test_offline_replay_exposes_each_cognitive_stage(tmp_path: Path) -> None:
    database = Database(tmp_path / "replay.sqlite3", MIGRATIONS)
    try:
        frames = await replay_events(
            [
                message("start", 0, message_id=1),
                message("answer", 5, actor_id=200, message_id=2, reply_to=1),
            ],
            database,
        )
        assert len(frames) == 2
        assert frames[-1].resolved_expectations >= 1
        assert frames[-1].last_difference is not None
        assert frames[-1].episode_id is not None
        assert frames[-1].semantic_memories >= 1
        assert frames[-1].tension is not None
        assert frames[-1].shadow_intent is not None
    finally:
        await database.close()
