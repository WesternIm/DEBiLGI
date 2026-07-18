from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.autonomy.policy import InitiativePolicy
from debilgi.autonomy.repository import AutonomyRepository
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.events.models import Event, EventType
from debilgi.experience.models import PreparedReply
from debilgi.experience.repository import ExperienceRepository
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "autonomy.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(
    event_id: str,
    *,
    at: datetime = NOW,
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
            "bot_mentioned": False,
            "content": "server behaves strangely again",
        },
        source="test",
    )


def prepared(goal_id: str = "goal:test") -> PreparedReply:
    return PreparedReply(
        text="server behaves strangely?",
        intent_id="intent:test",
        tokens=("server", "behaves", "strangely", "?"),
        genome_signature=DEBILGI_GENOME.signature,
        candidate_count=32,
        initiative=True,
        goal_id=goal_id,
        initiative_drive=0.8,
        initiative_reasoning=({"stage": "test"},),
    )


def intent() -> ActionIntent:
    return ActionIntent(
        intent_id="intent:test",
        guild_id=1,
        action_type=ActionType.SAVE_OBSERVATION,
        target_id=100,
        channel_id=10,
        urgency=0.7,
        confidence=0.8,
        expected_effect={"hypothesis": "observe"},
        reasoning_trace=(),
        evidence=("message:1",),
        decision_context={
            "state_uncertainty": 1.0,
            "state_interest": 1.0,
            "state_unexpectedness": 1.0,
            "familiarity": 0.8,
        },
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )


async def test_open_goal_blocks_another_and_direct_reply_resolves_it(
    database: Database,
) -> None:
    autonomy = AutonomyRepository(database)
    trigger = event("message:1")
    await autonomy.observe(trigger)

    assert await autonomy.start_goal(
        trigger,
        response_message_id=999,
        reply=prepared(),
        created_at=NOW,
    )
    assert not await autonomy.start_goal(
        trigger,
        response_message_id=1000,
        reply=prepared("goal:second"),
        created_at=NOW,
    )
    assert (await autonomy.state(1, 10)).evidence_balance == pytest.approx(0.28)

    await autonomy.observe(event("message:2", reply_to=999, at=NOW + timedelta(hours=4)))

    assert await autonomy.open_goal(1, 10) is None
    assert await autonomy.goal_counts(1) == {"answered": 1}


async def test_balance_changes_from_events_not_elapsed_time(database: Database) -> None:
    autonomy = AutonomyRepository(database)
    trigger = event("message:3")
    await autonomy.observe(trigger)
    await autonomy.start_goal(
        trigger,
        response_message_id=1001,
        reply=prepared(),
        created_at=NOW,
    )
    before = await autonomy.state(1, 10)

    unchanged = await autonomy.state(1, 10)
    assert unchanged.evidence_balance == before.evidence_balance

    await autonomy.observe(event("message:4", actor_id=200))
    await autonomy.observe(event("message:5", actor_id=200))
    after = await autonomy.state(1, 10)
    assert after.evidence_balance == pytest.approx(before.evidence_balance + 0.28)


async def test_policy_uses_expected_information_and_respects_open_goal(
    database: Database,
) -> None:
    autonomy = AutonomyRepository(database)
    policy = InitiativePolicy(
        DEBILGI_GENOME,
        autonomy,
        ExperienceRepository(database),
        enabled=True,
    )
    trigger = event("message:6")
    await autonomy.observe(trigger)

    decision = await policy.evaluate(trigger, intent())

    assert decision.should_initiate
    assert decision.goal_id is not None
    assert {step["stage"] for step in decision.reasoning} == {
        "initiative_state",
        "expected_information_gain",
        "self_model",
        "initiative_decision",
    }

    await autonomy.start_goal(
        trigger,
        response_message_id=1002,
        reply=prepared(decision.goal_id),
        created_at=NOW,
    )
    blocked = await policy.evaluate(event("message:7"), intent())
    assert not blocked.should_initiate
