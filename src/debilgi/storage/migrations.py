"""Ordered, checksum-verified SQL migrations."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import aiosqlite

MIGRATION_PATTERN = re.compile(r"^(?P<version>\d+)_(?P<name>[a-z0-9_]+)\.sql$")


class MigrationError(RuntimeError):
    """Raised when migration files are missing, malformed, or changed after use."""


async def apply_migrations(connection: aiosqlite.Connection, directory: Path) -> None:
    if not directory.is_dir():
        raise MigrationError(f"Migrations directory does not exist: {directory}")

    await connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await connection.commit()

    applied_rows = await (
        await connection.execute("SELECT version, checksum FROM schema_migrations")
    ).fetchall()
    applied = {int(row[0]): str(row[1]) for row in applied_rows}

    migrations: list[tuple[int, str, Path, str]] = []
    for path in directory.glob("*.sql"):
        match = MIGRATION_PATTERN.fullmatch(path.name)
        if not match:
            raise MigrationError(f"Invalid migration filename: {path.name}")
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()
        migrations.append((int(match["version"]), match["name"], path, checksum))

    versions = [version for version, _, _, _ in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("Duplicate migration version")

    for version, name, path, checksum in sorted(migrations):
        if version in applied:
            if applied[version] != checksum:
                raise MigrationError(f"Applied migration {path.name} was modified")
            continue

        sql = path.read_text(encoding="utf-8")
        escaped_name = name.replace("'", "''")
        script = (
            "BEGIN IMMEDIATE;\n"
            f"{sql}\n"
            "INSERT INTO schema_migrations(version, name, checksum) "
            f"VALUES ({version}, '{escaped_name}', '{checksum}');\n"
            "COMMIT;"
        )
        try:
            await connection.executescript(script)
        except Exception:
            await connection.rollback()
            raise


async def current_schema_version(connection: aiosqlite.Connection) -> int:
    row = await (
        await connection.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migrations")
    ).fetchone()
    return int(row[0]) if row else 0
