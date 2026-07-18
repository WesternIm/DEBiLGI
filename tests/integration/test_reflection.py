from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.models import PreparedReply
from debilgi.experience.repository import ExperienceRepository
from debilgi.reflection.engine import ReflectionEngine
from debilgi.reflection.repository import ReflectionRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 18, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "reflection.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(event_id: str, *, at: datetime, reply_to: int | None = None) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED if reply_to else EventType.MESSAGE_CREATED,
        timestamp=at,
        payload={
            "message_id": int(event_id.rsplit(":", 1)[-1]),
            "reply_to_message_id": reply_to,
            "author_is_bot": False,
            "content": "feedback",
        },
        source="test",
    )


def prepared() -> PreparedReply:
    return PreparedReply(
        text="why did the lamp stop working?",
        intent_id="intent:reflection",
        tokens=("why", "did", "the", "lamp", "stop", "working", "?"),
        genome_signature="genome-signature",
        candidate_count=64,
    )


async def test_outcome_creates_one_reflection_and_strategy_beliefs(database: Database) -> None:
    experiences = ExperienceRepository(database)
    reflections = ReflectionRepository(database)
    engine = ReflectionEngine(reflections, experiences)
    consolidator = ExperienceConsolidator(experiences, reflection=engine)
    prompt = event("message:1", at=NOW)
    await consolidator.register_sent(
        prompt,
        response_message_id=999,
        reply=prepared(),
        created_at=NOW + timedelta(seconds=1),
    )

    pending = await experiences.pending_for_response_message(1, 999)
    assert pending is not None
    assert pending.expected_reward == 0.0

    feedback = event("message:2", at=NOW + timedelta(seconds=5), reply_to=999)
    await consolidator.observe(feedback)
    await consolidator.observe(feedback)

    reflection = await reflections.reflection_for_experience(pending.experience_id)
    stats = await reflections.stats(1)
    beliefs = await reflections.beliefs(1)
    assert reflection is not None
    assert reflection.lesson_kind == "better_than_expected"
    assert reflection.actual_reward == pytest.approx(0.90)
    assert reflection.prediction_error == pytest.approx(0.90)
    assert "act:question" in reflection.strategies
    assert "search:wide" in reflection.strategies
    assert stats.reflections == 1
    assert stats.strategy_beliefs == 4
    assert all(belief.observations == 1 for belief in beliefs)


async def test_learned_strategy_reward_becomes_next_prediction(database: Database) -> None:
    experiences = ExperienceRepository(database)
    reflections = ReflectionRepository(database)
    engine = ReflectionEngine(reflections, experiences)
    consolidator = ExperienceConsolidator(experiences, reflection=engine)

    first = event("message:10", at=NOW)
    await consolidator.register_sent(
        first,
        response_message_id=1000,
        reply=prepared(),
        created_at=NOW,
    )
    await consolidator.observe(
        event("message:11", at=NOW + timedelta(seconds=2), reply_to=1000)
    )

    second = event("message:12", at=NOW + timedelta(seconds=4))
    await consolidator.register_sent(
        second,
        response_message_id=1001,
        reply=prepared(),
        created_at=NOW + timedelta(seconds=5),
    )
    pending = await experiences.pending_for_response_message(1, 1001)
    assert pending is not None
    assert pending.expected_reward == pytest.approx(0.90)

    await consolidator.observe(
        event("message:13", at=NOW + timedelta(seconds=8), reply_to=1001)
    )
    reflection = await reflections.reflection_for_experience(pending.experience_id)
    assert reflection is not None
    assert reflection.lesson_kind == "expectation_confirmed"
    assert reflection.surprise == pytest.approx(0.0)
