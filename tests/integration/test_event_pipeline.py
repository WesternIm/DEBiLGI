import asyncio
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


async def test_retry_resumes_failed_stage_without_restart(database: Database) -> None:
    repository = EventRepository(database)
    calls = [0, 0]

    async def first(_event: Event) -> None:
        calls[0] += 1

    async def flaky(_event: Event) -> None:
        calls[1] += 1
        if calls[1] == 1:
            raise RuntimeError("temporary failure")

    pipeline = EventPipeline(repository, [first, flaky], retry_delay=0)
    await pipeline.start()
    try:
        await pipeline.publish(make_event())
        assert await pipeline.wait_until_processed("pipeline:1")
        await pipeline.join()
        assert calls == [1, 2]
        row = await database.fetchone("SELECT attempts FROM event_processing")
        assert row["attempts"] == 2
    finally:
        await pipeline.stop()


async def test_retry_budget_is_bounded(database: Database) -> None:
    repository = EventRepository(database)
    calls = 0

    async def broken(_event: Event) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("permanent failure")

    pipeline = EventPipeline(repository, [broken], max_attempts=3, retry_delay=0)
    await pipeline.start()
    try:
        await pipeline.publish(make_event())
        assert not await pipeline.wait_until_processed("pipeline:1")
        await pipeline.join()
        assert calls == 3
        assert pipeline.running
    finally:
        await pipeline.stop()


async def test_failure_recording_error_does_not_kill_worker(
    database: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = EventRepository(database)

    async def broken_mark_failed(*_args) -> None:
        raise RuntimeError("database temporarily unavailable")

    async def handler(event: Event) -> None:
        if event.event_id == "broken":
            raise RuntimeError("handler failed")

    monkeypatch.setattr(repository, "mark_failed", broken_mark_failed)
    pipeline = EventPipeline(repository, [handler], max_attempts=1)
    await pipeline.start()
    try:
        await pipeline.publish(make_event("broken"))
        await pipeline.publish(make_event("healthy"))
        await asyncio.wait_for(pipeline.join(), timeout=2)
        assert pipeline.running
        assert await repository.processing_status("healthy") is ProcessingStatus.PROCESSED
        assert await repository.processing_status("broken") is ProcessingStatus.PROCESSING
    finally:
        await pipeline.stop()


async def test_stop_timeout_preserves_events_for_recovery(database: Database) -> None:
    repository = EventRepository(database)
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = []

    async def handler(event: Event) -> None:
        entered.set()
        await release.wait()
        completed.append(event.event_id)

    pipeline = EventPipeline(repository, [handler], queue_size=1)
    await pipeline.start()
    await pipeline.publish(make_event("first"))
    await asyncio.wait_for(entered.wait(), timeout=2)
    await pipeline.publish(make_event("second"))
    await asyncio.wait_for(pipeline.stop(timeout=0.01), timeout=2)
    assert not pipeline.running
    assert await repository.processing_status("first") is ProcessingStatus.PROCESSING
    assert await repository.processing_status("second") is ProcessingStatus.PENDING
    release.set()
    await pipeline.start()
    try:
        await asyncio.wait_for(pipeline.join(), timeout=2)
        assert completed == ["first", "second"]
    finally:
        await pipeline.stop()


async def test_running_reports_dead_workers(database: Database) -> None:
    pipeline = EventPipeline(EventRepository(database))
    await pipeline.start()
    for worker in pipeline._workers:
        worker.cancel()
    await asyncio.gather(*pipeline._workers, return_exceptions=True)
    assert not pipeline.running
    await pipeline.stop()


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


async def test_answer_waits_only_for_required_observers(database):
    repository = EventRepository(database)
    before, after = asyncio.Event(), asyncio.Event()

    async def required(_event):
        before.set()

    async def learning(_event):
        after.set()

    pipeline = EventPipeline(repository, [required], post_handlers=[learning])
    await pipeline.start()
    try:
        waiter = asyncio.create_task(pipeline.wait_until_ready("pipeline:1"))
        await asyncio.sleep(0)
        await pipeline.publish(make_event())
        assert await asyncio.wait_for(waiter, 1)
        assert before.is_set() and not after.is_set()
        pipeline.finish_response("pipeline:1")
        await asyncio.wait_for(pipeline.join(), 1)
        assert after.is_set()
        assert await pipeline.wait_until_processed("pipeline:1")
    finally:
        pipeline.finish_response("pipeline:1")
        await pipeline.stop()


async def test_timed_out_waiter_does_not_disconnect_another_waiter(database):
    release = asyncio.Event()

    async def required(_event):
        await release.wait()

    pipeline = EventPipeline(EventRepository(database), [required])
    await pipeline.start()
    try:
        await pipeline.publish(make_event())
        first = asyncio.create_task(pipeline.wait_until_processed("pipeline:1", timeout=.01))
        second = asyncio.create_task(pipeline.wait_until_processed("pipeline:1", timeout=1))
        assert not await first
        release.set()
        assert await second
    finally:
        release.set()
        await pipeline.stop()


async def test_recovered_event_does_not_reuse_cancelled_wait_result(database):
    entered, release = asyncio.Event(), asyncio.Event()

    async def required(_event):
        entered.set()
        await release.wait()

    pipeline = EventPipeline(EventRepository(database), [required])
    await pipeline.start()
    try:
        await pipeline.publish(make_event())
        await asyncio.wait_for(entered.wait(), 1)
        await pipeline.stop(drain=False)
        assert not await pipeline.wait_until_processed("pipeline:1")
        await pipeline.start()
        waiter = asyncio.create_task(pipeline.wait_until_processed("pipeline:1"))
        await asyncio.sleep(0)
        release.set()
        assert await asyncio.wait_for(waiter, 1)
    finally:
        release.set()
        await pipeline.stop()


async def test_previous_turn_learning_does_not_block_next_turn_readiness(database):
    learning_entered, release = asyncio.Event(), asyncio.Event()

    async def learning(event):
        if event.event_id == "first":
            learning_entered.set()
            await release.wait()

    pipeline = EventPipeline(EventRepository(database), post_handlers=[learning])
    await pipeline.start()
    try:
        await pipeline.publish(make_event("first"))
        await asyncio.wait_for(learning_entered.wait(), 1)
        await pipeline.publish(make_event("second"))
        assert await pipeline.wait_until_ready("second", timeout=.5)
        pipeline.finish_response("second")
        assert not release.is_set()
    finally:
        release.set()
        pipeline.finish_response("second")
        await pipeline.stop()
