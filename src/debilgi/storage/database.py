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
        self._write_lock = asyncio.Lock()
        self._transaction_owner: asyncio.Task[Any] | None = None

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
        self._connection = connection

    async def close(self) -> None:
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
        # The owner may read its own writes. Other tasks wait for commit/rollback.
        if self._transaction_owner is asyncio.current_task():
            yield self.connection
        else:
            async with self._write_lock:
                yield self.connection

    async def fetchone(self, sql: str, parameters: Iterable[Any] = ()) -> aiosqlite.Row | None:
        async with (
            self._read_connection() as connection,
            connection.execute(sql, tuple(parameters)) as cursor,
        ):
            return await cursor.fetchone()

    async def fetchall(self, sql: str, parameters: Iterable[Any] = ()) -> list[aiosqlite.Row]:
        async with (
            self._read_connection() as connection,
            connection.execute(sql, tuple(parameters)) as cursor,
        ):
            return list(await cursor.fetchall())

    async def schema_version(self) -> int:
        async with self._read_connection() as connection:
            return await current_schema_version(connection)
