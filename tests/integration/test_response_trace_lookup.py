from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 29, 6, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "response-trace-lookup.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_trace_lookup_is_scoped_to_the_discord_guild(database: Database) -> None:
    event = Event(
        event_id="discord:message:guild-trace",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "А почему так?", "author_is_bot": False},
        source="test",
    )
    await EventRepository(database).append(event)
    traces = ResponseTraceRepository(database)
    await traces.begin(event, prompt="А почему так?")
    await traces.step(
        event.event_id,
        stage="selection",
        component="response_arbitrator",
        status="selected",
        summary="contextual explanation selected",
    )

    latest = await traces.trace_for_guild(1)
    selected = await traces.trace_for_guild(1, event_id=event.event_id)

    assert latest is not None
    assert latest[0]["source_event_id"] == event.event_id
    assert selected is not None
    assert selected[1][0]["stage"] == "selection"
    assert await traces.trace_for_guild(2, event_id=event.event_id) is None
