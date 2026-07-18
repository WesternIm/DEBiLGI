from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.social.engine import assess_reaction
from debilgi.social.repository import SocialReactionRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 15, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "social.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event() -> Event:
    return Event(
        event_id="event:social:1",
        guild_id=1,
        channel_id=2,
        actor_id=3,
        target_id=4,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "да, это отлично", "author_is_bot": False},
        source="test",
    )


async def test_reaction_is_idempotent_and_updates_relation_once(database: Database) -> None:
    current = event()
    await EventRepository(database).append(current)
    repository = SocialReactionRepository(database)
    reaction = assess_reaction(current, None, DEBILGI_GENOME)

    assert await repository.save(reaction)
    assert not await repository.save(reaction)

    restored = await repository.reaction_for_event(current.event_id)
    relation = await repository.relation_state(1, 3)
    stats = await repository.stats(1)
    assert restored is not None
    assert restored.blend_id == reaction.blend_id
    assert relation is not None
    assert relation.observations == 1
    assert stats.reactions == 1
    assert stats.relation_models == 1
