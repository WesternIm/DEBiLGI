import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from debilgi.storage.database import Database

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "isolation.sqlite3", MIGRATIONS)
    await database.initialize()
    await database.execute("CREATE TABLE isolation_probe (value INTEGER)")
    await database.execute("INSERT INTO isolation_probe VALUES (1)")
    try:
        yield database
    finally:
        await database.close()


@pytest.mark.parametrize("rollback", [False, True])
async def test_other_task_cannot_read_uncommitted_changes(
    database: Database, rollback: bool
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    reading = asyncio.Event()

    async def writer() -> None:
        try:
            async with database.transaction() as connection:
                await connection.execute("UPDATE isolation_probe SET value = 2")
                row = await database.fetchone("SELECT value FROM isolation_probe")
                assert row["value"] == 2
                entered.set()
                await release.wait()
                if rollback:
                    raise ValueError("rollback requested")
        except ValueError:
            if not rollback:
                raise

    async def reader():
        reading.set()
        return await database.fetchall("SELECT value FROM isolation_probe")

    write_task = asyncio.create_task(writer())
    read_task = None
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        read_task = asyncio.create_task(reader())
        await asyncio.wait_for(reading.wait(), timeout=2)
        # The WAL reader may complete before the writer commits. Establish
        # that ordering explicitly, then check a fresh post-commit snapshot.
        rows = await asyncio.wait_for(read_task, timeout=2)
        assert rows[0]["value"] == 1
        release.set()
        await asyncio.wait_for(write_task, timeout=2)
        current = await database.fetchone("SELECT value FROM isolation_probe")
        assert current["value"] == (1 if rollback else 2)
    finally:
        release.set()
        tasks = [write_task] + ([read_task] if read_task is not None else [])
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_invalid_nested_writes_fail_without_deadlock(database: Database) -> None:
    async with database.transaction():
        with pytest.raises(RuntimeError, match="transaction connection"):
            await database.execute("UPDATE isolation_probe SET value = 3")
        with pytest.raises(RuntimeError, match="Nested transactions"):
            async with database.transaction():
                pass
    row = await database.fetchone("SELECT value FROM isolation_probe")
    assert row["value"] == 1


async def test_failed_initialization_can_be_retried(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    migration = migrations / "001_probe.sql"
    migration.write_text("INVALID SQL;", encoding="utf-8")
    database = Database(tmp_path / "retry.sqlite3", migrations)
    try:
        with pytest.raises(sqlite3.OperationalError):
            await database.initialize()
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = database.connection
        migration.write_text("CREATE TABLE probe (value INTEGER);", encoding="utf-8")
        await database.initialize()
        assert await database.schema_version() == 1
    finally:
        await database.close()


async def test_reader_uses_committed_snapshot_without_waiting_for_writer(database):
    async with database.transaction() as connection:
        await connection.execute("UPDATE isolation_probe SET value = 7")
        async def committed_read():
            return await database.fetchone("SELECT value FROM isolation_probe")
        row = await asyncio.wait_for(asyncio.create_task(committed_read()), timeout=1)
        assert row["value"] == 1
        assert (await database.fetchone("SELECT value FROM isolation_probe"))["value"] == 7
    assert (await database.fetchone("SELECT value FROM isolation_probe"))["value"] == 7


async def test_overlapping_readers_do_not_reuse_a_pre_commit_cursor_snapshot(database):
    entered, release = asyncio.Event(), asyncio.Event()

    async def old_reader():
        async with (
            database._read_connection() as connection,
            connection.execute("SELECT value FROM isolation_probe") as cursor,
        ):
            assert (await cursor.fetchone())["value"] == 1
            entered.set()
            await release.wait()

    first = asyncio.create_task(old_reader())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await database.execute("UPDATE isolation_probe SET value = 8")
        second = asyncio.create_task(database.fetchone("SELECT value FROM isolation_probe"))
        await asyncio.sleep(0)
        release.set()
        await first
        assert (await asyncio.wait_for(second, 1))["value"] == 8
    finally:
        release.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
