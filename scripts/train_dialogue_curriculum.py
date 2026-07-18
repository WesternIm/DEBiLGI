"""Install the multi-turn conversational curriculum into the active local archive."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.dialogue_curriculum import SCENES, total_turns
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.models import ArchivedMessage
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1371151602559225978
CHANNEL_ID = 1371151715184676878
BASE_MESSAGE_ID = 8_200_000_000_000_000_000
AUTHOR_IDS = {
    "bot": 1_527_681_283_470_135_466,
    "human_a": 9_200_000_000_000_001,
    "human_b": 9_200_000_000_000_002,
}


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup_path = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        archive = ArchiveRepository(database)
        discourse = DiscourseMemoryRepository(database)
        semantic = SemanticMemoryRepository(database)
        last_message_id = BASE_MESSAGE_ID + total_turns() - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_message_id,)
        )
        if complete is None and not evaluate_only:
            inserted = await _ingest(archive, guild_id)
            print(f"dialogue_messages={inserted}")
        elif complete is None:
            print("dialogue_messages=not_present")
        else:
            print("dialogue_messages=already_present")

        if not evaluate_only:
            print(f"semantic_rebuilt={await semantic.rebuild_archive(guild_id)}")
            print(f"discourse_rebuilt={await discourse.rebuild_archive(guild_id)}")

        stats = await archive.language_stats(guild_id)
        discourse_stats = await discourse.stats(guild_id)
        skill_counts = Counter(skill for scene in SCENES for skill in scene.skills)
        print(f"backup={backup_path}")
        print(f"scenes={len(SCENES)}")
        print(f"curriculum_turns={total_turns()}")
        print(f"skills={len(skill_counts)}")
        print(f"raw_messages={stats.raw_messages}")
        print(f"reply_patterns={stats.reply_patterns}")
        print(f"discourse_turns={discourse_stats.turns}")
        print(f"open_questions={discourse_stats.open_questions}")
        print(f"answered_questions={discourse_stats.answered_questions}")
    finally:
        await database.close()


async def _ingest(repository: ArchiveRepository, guild_id: int) -> int:
    started = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)
    job = await repository.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=AUTHOR_IDS["human_a"],
        after_at=started - timedelta(days=1),
        before_at=started + timedelta(days=2),
        max_messages=total_turns(),
        delete_raw_after=False,
    )
    await repository.mark_running(job.job_id)
    inserted = 0
    message_offset = 0
    for scene_index, scene in enumerate(SCENES):
        parent_message_id: int | None = None
        scene_start = started + timedelta(hours=scene_index)
        for turn_index, turn in enumerate(scene.turns):
            message_id = BASE_MESSAGE_ID + message_offset
            inserted += await repository.ingest_message(
                job.job_id,
                ArchivedMessage(
                    message_id=message_id,
                    guild_id=guild_id,
                    channel_id=CHANNEL_ID,
                    author_id=AUTHOR_IDS[turn.speaker],
                    created_at=scene_start + timedelta(seconds=turn_index * 25),
                    content=turn.text,
                    reply_to_message_id=parent_message_id,
                    author_is_bot=turn.speaker == "bot",
                ),
            )
            parent_message_id = message_id
            message_offset += 1
    await repository.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + total_turns() - 1,
        seen=inserted,
        processed=inserted,
    )
    await repository.complete_job(job.job_id)
    return inserted


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(f"{database_path.stem}-before-dialogues-{stamp}.sqlite3")
    with sqlite3.connect(database_path) as source, sqlite3.connect(backup) as target:
        source.backup(target)
    return backup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args()
    asyncio.run(train(evaluate_only=arguments.evaluate_only))


if __name__ == "__main__":
    main()
