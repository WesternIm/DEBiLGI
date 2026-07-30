from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.experience.consolidation import ExperienceConsolidator
from debilgi.experience.models import PreparedReply
from debilgi.experience.repository import ExperienceRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "experience.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(
    event_id: str,
    *,
    at: datetime,
    reply_to: int | None = None,
    actor_id: int = 100,
) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=actor_id,
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


def prepared(*, initiative: bool = False) -> PreparedReply:
    return PreparedReply(
        text="server again broken",
        intent_id="intent:test",
        tokens=("server", "again", "broken"),
        genome_signature="genome-signature",
        candidate_count=32,
        initiative=initiative,
        goal_id="goal:test" if initiative else None,
        initiative_drive=0.8 if initiative else 0.0,
    )


async def test_plain_direct_reply_is_neutral_for_language_transitions(database: Database) -> None:
    repository = ExperienceRepository(database)
    consolidator = ExperienceConsolidator(repository)
    prompt = event("message:1", at=NOW)
    await consolidator.register_sent(
        prompt,
        response_message_id=999,
        reply=prepared(),
        created_at=NOW + timedelta(seconds=1),
    )

    await consolidator.observe(event("message:2", at=NOW + timedelta(seconds=5), reply_to=999))

    stats = await repository.stats(1)
    adjustments = dict(await repository.transition_adjustments(1))
    assert stats.resolved == 1
    assert stats.average_reward == pytest.approx(0.0)
    assert stats.learned_transitions == 3
    assert adjustments == {}


async def test_ignored_reply_expires_and_weakens_transitions(database: Database) -> None:
    repository = ExperienceRepository(database)
    consolidator = ExperienceConsolidator(repository)
    prompt = event("message:3", at=NOW)
    await consolidator.register_sent(
        prompt,
        response_message_id=1000,
        reply=prepared(),
        created_at=NOW,
    )

    await consolidator.observe(event("message:4", at=NOW + timedelta(minutes=11), actor_id=200))

    stats = await repository.stats(1)
    adjustments = dict(await repository.transition_adjustments(1))
    assert stats.expired == 1
    assert stats.average_reward == pytest.approx(-0.18)
    assert adjustments[("server", "again")] < 0.0


async def test_negative_reaction_weakens_the_generated_transitions(database: Database) -> None:
    repository = ExperienceRepository(database)
    consolidator = ExperienceConsolidator(repository)
    prompt = event("message:5", at=NOW)
    await consolidator.register_sent(
        prompt,
        response_message_id=1001,
        reply=prepared(),
        created_at=NOW,
    )
    reaction = Event(
        event_id="reaction:negative",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.REACTION_ADDED,
        timestamp=NOW + timedelta(seconds=3),
        payload={"message_id": 1001, "emoji": "👎"},
        source="test",
    )

    await consolidator.observe(reaction)

    stats = await repository.stats(1)
    adjustments = dict(await repository.transition_adjustments(1))
    assert stats.resolved == 1
    assert stats.average_reward == pytest.approx(-0.8)
    assert adjustments[("server", "again")] < 0.0


async def test_neutral_reply_does_not_claim_initiative_success(database: Database) -> None:
    repository = ExperienceRepository(database)
    consolidator = ExperienceConsolidator(repository)
    prompt = event("message:6", at=NOW)
    await consolidator.register_sent(
        prompt,
        response_message_id=1002,
        reply=prepared(initiative=True),
        created_at=NOW,
    )

    await consolidator.observe(event("message:7", at=NOW + timedelta(seconds=3), reply_to=1002))

    self_model = await repository.self_model(1)
    assert self_model.utterances_observed == 1
    assert self_model.initiative_attempts == 1
    assert self_model.initiative_successes == 0
    assert self_model.efficacy == pytest.approx(0.5)
