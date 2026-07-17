from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.literacy.engine import LiteracyEngine
from debilgi.literacy.repository import LiteracyRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 20, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "literacy.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event() -> Event:
    return Event(
        event_id="event:literacy:1",
        guild_id=1,
        channel_id=2,
        actor_id=3,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "сколько будет 12 + 8?", "author_is_bot": False},
        source="test",
    )


async def test_observation_is_idempotent_and_updates_aggregate_once(database: Database) -> None:
    repository = LiteracyRepository(database)
    engine = LiteracyEngine(repository)
    current = event()

    await engine.observe(current)
    await engine.observe(current)

    stats = await repository.stats(1)
    assert stats.messages_observed == 1
    assert stats.letters_observed == 12
    assert stats.digits_observed == 3
    assert stats.arithmetic_queries == 1
