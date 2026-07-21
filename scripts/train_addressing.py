"""Train the active archive and grammar statistics on Russian direct address."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.events.normalizer import normalize_message_created
from debilgi.language.addressing import DEFAULT_ADDRESSING
from debilgi.language.addressing_curriculum import ADDRESSING_PAIRS, ADDRESSING_SKILLS
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.models import ArchivedMessage
from debilgi.language.morphology import MorphologyRepository
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978
CHANNEL_ID = 1_371_151_715_184_676_881
BOT_ID = 1_527_681_283_470_135_466
TEACHER_ID = 9_300_000_000_000_004
BASE_MESSAGE_ID = 8_600_000_000_000_000_000
STARTED_AT = datetime(2026, 7, 18, 21, 0, tzinfo=UTC)


async def train(*, evaluate_only: bool = False) -> None:
    settings = Settings.from_env()
    database_path = settings.database_path.resolve()
    guild_id = settings.dev_guild_id or DEFAULT_GUILD_ID
    backup = None if evaluate_only else _backup(database_path)
    database = Database(database_path, settings.migrations_path)
    await database.initialize()
    try:
        archive = ArchiveRepository(database)
        morphology = MorphologyRepository(database)
        semantic = SemanticMemoryRepository(database)
        discourse = DiscourseMemoryRepository(database)
        if not evaluate_only:
            await _fail_abandoned_training_jobs(database)
        last_id = BASE_MESSAGE_ID + len(ADDRESSING_PAIRS) * 2 - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_id,)
        )
        if not evaluate_only and complete is None:
            print(f"addressing_messages={await _ingest(archive, morphology, guild_id)}")
            print(f"semantic_rebuilt={await semantic.rebuild_archive(guild_id)}")
            print(f"discourse_rebuilt={await discourse.rebuild_archive(guild_id)}")
        else:
            print(f"addressing_messages={'already_present' if complete else 'not_present'}")
        await _report(database, archive, guild_id, backup)
    finally:
        await database.close()


async def _ingest(
    archive: ArchiveRepository, morphology: MorphologyRepository, guild_id: int
) -> int:
    total = len(ADDRESSING_PAIRS) * 2
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
    for prompt, response in ADDRESSING_PAIRS:
        prompt_id = BASE_MESSAGE_ID + offset
        response_id = prompt_id + 1
        moment = STARTED_AT + timedelta(minutes=offset)
        for message_id, author_id, text, parent_id, is_bot, observed_at in (
            (prompt_id, TEACHER_ID, prompt, None, False, moment),
            (
                response_id,
                BOT_ID,
                response,
                prompt_id,
                True,
                moment + timedelta(seconds=5),
            ),
        ):
            added = await archive.ingest_message(
                job.job_id,
                ArchivedMessage(
                    message_id=message_id,
                    guild_id=guild_id,
                    channel_id=CHANNEL_ID,
                    author_id=author_id,
                    created_at=observed_at,
                    content=text,
                    reply_to_message_id=parent_id,
                    author_is_bot=is_bot,
                ),
            )
            inserted += int(added)
            if added:
                await morphology.observe(
                    normalize_message_created(
                        message_id=message_id,
                        guild_id=guild_id,
                        channel_id=CHANNEL_ID,
                        author_id=TEACHER_ID,
                        created_at=observed_at,
                        content=text,
                        author_is_bot=False,
                        reply_to_message_id=parent_id,
                    )
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


async def _report(
    database: Database,
    archive: ArchiveRepository,
    guild_id: int,
    backup: Path | None,
) -> None:
    row = await database.fetchone(
        """SELECT COALESCE(SUM(occurrence_count), 0) AS count
           FROM language_grammar_patterns WHERE guild_id = ? AND roles_json LIKE '%address%'""",
        (guild_id,),
    )
    print(f"backup={backup}")
    print(f"schema={await database.schema_version()}")
    print(f"addressing_pairs={len(ADDRESSING_PAIRS)}")
    print(f"addressing_skills={len(ADDRESSING_SKILLS)}")
    print(f"address_pattern_observations={int(row['count']) if row else 0}")
    print("addressing_probes:")
    for phrase in (
        "Узнай, умник",
        "Узнай умник",
        "Найди умника",
        "Что значит «умник»?",
        "Маша, где мороженое?",
        "Умник узнай что такое мороженое",
        "Скажи умник где ключ",
        "Маша ты знаешь ответ",
        "Ты умник расскажи",
    ):
        analysis = DEFAULT_ADDRESSING.analyze(phrase)
        addresses = tuple(span.text for span in analysis.addresses)
        print(f"- {phrase} -> content={analysis.content_text!r}, addresses={addresses}")
    print("learned_reply_probes:")
    for prompt in ("Что такое обращение?", "Узнай, умник.", "Найди умника."):
        answer = await archive.exact_learned_reply(guild_id, tokenize(prompt))
        print(f"- {prompt} -> {answer}")


async def _fail_abandoned_training_jobs(database: Database) -> None:
    await database.execute(
        """UPDATE archive_import_jobs SET
            status = 'failed', last_error = 'abandoned addressing training attempt',
            updated_at = CURRENT_TIMESTAMP
           WHERE channel_id = ? AND requested_by = ? AND status = 'running'
             AND messages_seen = 0 AND messages_processed = 0""",
        (CHANNEL_ID, TEACHER_ID),
    )


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(f"{database_path.stem}-before-addressing-{stamp}.sqlite3")
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
