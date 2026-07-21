"""Install natural self-dialogue pairs into the active local language archive."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.models import ArchivedMessage
from debilgi.language.self_dialogue_curriculum import SELF_DIALOGUE_PAIRS
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978
CHANNEL_ID = 1_371_151_715_184_676_880
BOT_ID = 1_527_681_283_470_135_466
TEACHER_ID = 9_300_000_000_000_003
BASE_MESSAGE_ID = 8_500_000_000_000_000_000
STARTED_AT = datetime(2026, 7, 18, 20, 0, tzinfo=UTC)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        archive = ArchiveRepository(database)
        if not evaluate_only:
            await _fail_abandoned_training_jobs(database)
        last_id = BASE_MESSAGE_ID + len(SELF_DIALOGUE_PAIRS) * 2 - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_id,)
        )
        if not evaluate_only and complete is None:
            print(f"self_dialogue_messages={await _ingest(archive, guild_id)}")
        else:
            print(f"self_dialogue_messages={'already_present' if complete else 'not_present'}")
        print(f"backup={backup}")
        print(f"schema={await database.schema_version()}")
        print(f"self_dialogue_pairs={len(SELF_DIALOGUE_PAIRS)}")
        print("reply_probes:")
        for prompt in ("ты разумный?", "а почему ты разумный?", "ты радуешься?"):
            exact = await archive.exact_learned_reply(guild_id, tokenize(prompt))
            print(f"- {prompt} -> {exact}")
        unrelated = await archive.matching_replies(guild_id, tokenize("ты квадратный?"))
        print(f"unrelated_self_reply_count={len(unrelated)}")
    finally:
        await database.close()


async def _ingest(archive: ArchiveRepository, guild_id: int) -> int:
    total = len(SELF_DIALOGUE_PAIRS) * 2
    job = await archive.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=TEACHER_ID,
        after_at=STARTED_AT - timedelta(days=1),
        before_at=STARTED_AT + timedelta(days=1),
        max_messages=total,
        delete_raw_after=False,
    )
    await archive.mark_running(job.job_id)
    inserted = 0
    offset = 0
    for prompt, response in SELF_DIALOGUE_PAIRS:
        prompt_id = BASE_MESSAGE_ID + offset
        response_id = prompt_id + 1
        moment = STARTED_AT + timedelta(minutes=offset)
        inserted += await archive.ingest_message(
            job.job_id,
            ArchivedMessage(
                message_id=prompt_id,
                guild_id=guild_id,
                channel_id=CHANNEL_ID,
                author_id=TEACHER_ID,
                created_at=moment,
                content=prompt,
                reply_to_message_id=None,
                author_is_bot=False,
            ),
        )
        inserted += await archive.ingest_message(
            job.job_id,
            ArchivedMessage(
                message_id=response_id,
                guild_id=guild_id,
                channel_id=CHANNEL_ID,
                author_id=BOT_ID,
                created_at=moment + timedelta(seconds=5),
                content=response,
                reply_to_message_id=prompt_id,
                author_is_bot=True,
            ),
        )
        offset += 2
    await archive.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + total - 1,
        seen=total,
        processed=inserted,
    )
    await archive.complete_job(job.job_id)
    return inserted


async def _fail_abandoned_training_jobs(database: Database) -> None:
    await database.execute(
        """UPDATE archive_import_jobs SET
            status = 'failed', last_error = 'abandoned self-dialogue training attempt',
            updated_at = CURRENT_TIMESTAMP
           WHERE channel_id = ? AND requested_by = ? AND status = 'running'
             AND messages_seen = 0 AND messages_processed = 0""",
        (CHANNEL_ID, TEACHER_ID),
    )


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(
        f"{database_path.stem}-before-self-dialogue-{stamp}.sqlite3"
    )
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
