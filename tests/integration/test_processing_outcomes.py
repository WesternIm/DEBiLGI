from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.processing_outcomes import ProcessingOutcomeRepository
from debilgi.language.semantic_reply_plan import (
    ProcessingOutcomeType,
    build_clarification_plan,
    realize_clarification_plan,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "processing-outcomes.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_processing_outcome_survives_process_memory(database: Database) -> None:
    event = Event(
        event_id="discord:message:semantic-plan",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "это самое", "author_is_bot": False},
        source="test",
    )
    await EventRepository(database).append(event)
    result = realize_clarification_plan(
        build_clarification_plan(
            intention_id="intent:semantic",
            target_user_id=100,
            prompt="это самое",
            confidence=0.8,
        )
    )

    await ProcessingOutcomeRepository(database).save(
        event,
        result.outcome,
        response_message_id=999,
        created_at=NOW,
    )

    row = await database.fetchone(
        "SELECT * FROM language_processing_outcomes WHERE source_event_id = ?",
        (event.event_id,),
    )
    assert row is not None
    assert row["outcome_type"] == ProcessingOutcomeType.REPLIED.value
    assert row["reply_plan_id"] == "semantic-reply:intent:semantic"
    assert row["response_message_id"] == 999
