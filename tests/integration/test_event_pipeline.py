from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.normalizer import normalize_reaction
from debilgi.events.pipeline import EventPipeline
from debilgi.events.repository import EventRepository, ProcessingStatus
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"


def make_event(event_id: str = "pipeline:1", guild_id: int = 1) -> Event:
    return Event(
        event_id=event_id,
        guild_id=guild_id,
        channel_id=2,
        actor_id=3,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
        payload={"message_id": 4},
        source="test",
    )


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "pipeline.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_duplicate_delivery_is_stored_and_processed_once(database: Database) -> None:
    repository = EventRepository(database)
    processed: list[str] = []

    async def handler(item: Event) -> None:
        processed.append(item.event_id)

    pipeline = EventPipeline(repository, [handler])
    await pipeline.start()
    try:
        assert await pipeline.publish(make_event()) is True
        assert await pipeline.publish(make_event()) is False
        assert await pipeline.wait_until_processed("pipeline:1")
        await pipeline.join()

        assert processed == ["pipeline:1"]
        assert await repository.count() == 1
        assert await repository.processing_status("pipeline:1") is ProcessingStatus.PROCESSED
    finally:
        await pipeline.stop()


async def test_pending_event_is_recovered_after_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "recovery.sqlite3"
    first_database = Database(db_path, MIGRATIONS)
    await first_database.initialize()
    first_repository = EventRepository(first_database)
    await first_repository.append(make_event("recovered"))
    await first_database.close()

    second_database = Database(db_path, MIGRATIONS)
    await second_database.initialize()
    second_repository = EventRepository(second_database)
    processed: list[str] = []

    async def handler(item: Event) -> None:
        processed.append(item.event_id)

    pipeline = EventPipeline(second_repository, [handler])
    await pipeline.start()
    try:
        await pipeline.join()
        assert processed == ["recovered"]
        assert await second_repository.processing_status("recovered") is ProcessingStatus.PROCESSED
    finally:
        await pipeline.stop()
        await second_database.close()


async def test_explicit_replay_reprocesses_without_duplicate_storage(database: Database) -> None:
    repository = EventRepository(database)
    processed: list[str] = []

    async def handler(item: Event) -> None:
        processed.append(item.event_id)

    pipeline = EventPipeline(repository, [handler])
    await pipeline.start()
    try:
        await pipeline.publish(make_event("replay-me", guild_id=42))
        await pipeline.join()
        assert await pipeline.replay(42, limit=1) == 1
        await pipeline.join()

        assert processed == ["replay-me", "replay-me"]
        assert await repository.count(42) == 1
    finally:
        await pipeline.stop()


async def test_handler_failure_is_persisted(database: Database) -> None:
    repository = EventRepository(database)

    async def broken_handler(_item: Event) -> None:
        raise RuntimeError("boom")

    pipeline = EventPipeline(repository, [broken_handler], max_attempts=1)
    await pipeline.start()
    try:
        await pipeline.publish(make_event("broken"))
        assert not await pipeline.wait_until_processed("broken")
        await pipeline.join()

        stats = await repository.processing_stats(1)
        assert stats.failed == 1
        assert await repository.processing_status("broken") is ProcessingStatus.FAILED
    finally:
        await pipeline.stop()


async def test_reaction_redelivery_is_once_but_add_and_remove_are_distinct(
    database: Database,
) -> None:
    repository = EventRepository(database)
    processed: list[str] = []

    async def handler(item: Event) -> None:
        processed.append(item.event_id)

    arguments = {
        "guild_id": 1,
        "channel_id": 2,
        "message_id": 3,
        "actor_id": 4,
        "target_id": 5,
        "emoji": "💅",
        "observed_at": datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    }
    added = normalize_reaction(added=True, **arguments)
    removed = normalize_reaction(added=False, **arguments)
    pipeline = EventPipeline(repository, [handler])
    await pipeline.start()
    try:
        assert await pipeline.publish(added)
        assert not await pipeline.publish(normalize_reaction(added=True, **arguments))
        assert await pipeline.publish(removed)
        await pipeline.join()

        assert processed == [added.event_id, removed.event_id]
        assert await repository.count(1) == 2
    finally:
        await pipeline.stop()
