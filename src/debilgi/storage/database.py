"""Small asynchronous SQLite boundary."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite

from debilgi.storage.migrations import apply_migrations, current_schema_version


class Database:
    def __init__(self, path: Path, migrations_path: Path) -> None:
        self.path = path
        self.migrations_path = migrations_path
        self._connection: aiosqlite.Connection | None = None
        self._reader: aiosqlite.Connection | None = None
        self.read_queries = 0
        self.rows_read = 0
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._transaction_owner: asyncio.Task[Any] | None = None
        self._maintenance_tasks: set[asyncio.Task[Any]] = set()

    @property
    def connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise RuntimeError("Database is not initialized")
        return self._connection

    async def initialize(self) -> None:
        if self._connection is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(self.path)
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute("PRAGMA foreign_keys = ON")
            await connection.execute("PRAGMA journal_mode = WAL")
            await connection.execute("PRAGMA busy_timeout = 5000")
            await apply_migrations(connection, self.migrations_path)
        except BaseException:
            await connection.close()
            raise
        reader = None
        try:
            reader = await aiosqlite.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
            reader.row_factory = aiosqlite.Row
            await reader.execute("PRAGMA query_only = ON")
            await reader.execute("PRAGMA busy_timeout = 5000")
        except BaseException:
            if reader is not None:
                await reader.close()
            await connection.close()
            raise
        self._connection = connection
        self._reader = reader

    async def close(self) -> None:
        pending = tuple(t for t in self._maintenance_tasks if t is not asyncio.current_task())
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._maintenance_tasks.clear()
        if self._reader is not None:
            await self._reader.close()
            self._reader = None
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def execute(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Cursor:
        if self._transaction_owner is asyncio.current_task():
            raise RuntimeError("Use the transaction connection for writes inside a transaction")
        async with self._write_lock:
            try:
                cursor = await self.connection.execute(sql, tuple(parameters))
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise
            return cursor

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialize a group of writes and commit or roll it back atomically."""
        if self._transaction_owner is asyncio.current_task():
            raise RuntimeError("Nested transactions are not supported")
        async with self._write_lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            self._transaction_owner = asyncio.current_task()
            try:
                yield self.connection
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise
            finally:
                self._transaction_owner = None

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[aiosqlite.Connection]:
        # The owner sees its writes; concurrent readers see the committed WAL snapshot.
        if self._transaction_owner is asyncio.current_task():
            yield self.connection
        else:
            if self._reader is None:
                raise RuntimeError("Database is not initialized")
            # One complete cursor per connection avoids retaining another
            # reader's old snapshot across a newly committed write.
            async with self._read_lock:
                yield self._reader

    async def fetchone(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with (
            self._read_connection() as connection,
            connection.execute(sql, tuple(parameters)) as cursor,
        ):
            row = await cursor.fetchone()
            self.read_queries += 1
            self.rows_read += int(row is not None)
            return row

    async def fetchall(self, sql: str, parameters: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with (
            self._read_connection() as connection,
            connection.execute(sql, tuple(parameters)) as cursor,
        ):
            rows = list(await cursor.fetchall())
            self.read_queries += 1
            self.rows_read += len(rows)
            return rows

    async def schema_version(self) -> int:
        async with self._read_connection() as connection:
            return await current_schema_version(connection)
