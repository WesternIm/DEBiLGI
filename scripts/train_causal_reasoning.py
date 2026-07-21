"""Warm-start the causal graph from retained local conversations."""

from __future__ import annotations

import asyncio

from debilgi.config import Settings
from debilgi.language.causal_reasoning import CausalReasoningRepository
from debilgi.storage.database import Database


async def train() -> tuple[int, int, int, int]:
    settings = Settings.from_env()
    database = Database(settings.database_path, settings.migrations_path)
    await database.initialize()
    try:
        repository = CausalReasoningRepository(database)
        guild_rows = await database.fetchall(
            """SELECT DISTINCT guild_id FROM events
               UNION SELECT DISTINCT guild_id FROM archive_messages"""
        )
        event_links = 0
        archive_links = 0
        for row in guild_rows:
            guild_id = int(row["guild_id"])
            event_links += await repository.rebuild_events(guild_id)
            archive_links += await repository.rebuild_archive(guild_id)
        knowledge_links = await repository.rebuild_knowledge()
        stored = await database.fetchone(
            "SELECT COUNT(*) AS count FROM language_causal_links"
        )
        return (
            event_links,
            archive_links,
            knowledge_links,
            int(stored["count"]) if stored is not None else 0,
        )
    finally:
        await database.close()


def main() -> None:
    event_links, archive_links, knowledge_links, stored = asyncio.run(train())
    print(
        f"event_observations={event_links} archive_observations={archive_links} "
        f"knowledge_observations={knowledge_links} "
        f"stored_links={stored}"
    )


if __name__ == "__main__":
    main()
