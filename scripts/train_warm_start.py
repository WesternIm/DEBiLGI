"""Install a compact everyday dialogue seed into the active local archive."""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from debilgi.config import Settings
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.experience.models import PreparedReply
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.models import ArchivedMessage
from debilgi.language.morphology import MorphologyRepository
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.language.warm_start_curriculum import WARM_START_PAIRS, WARM_START_SKILLS
from debilgi.storage.database import Database

DEFAULT_GUILD_ID = 1_371_151_602_559_225_978
CHANNEL_ID = 1_371_151_715_184_676_883
BOT_ID = 1_527_681_283_470_135_466
TEACHER_ID = 9_300_000_000_000_005
BASE_MESSAGE_ID = 8_700_000_000_000_000_000
ROUTING_BASE_MESSAGE_ID = 8_710_000_000_000_000_000
ROUTING_PAIR_COUNT = 6
STARTED_AT = datetime(2026, 7, 18, 22, 0, tzinfo=UTC)


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
        dataset = SelfDatasetRepository(database)
        if not evaluate_only:
            await _fail_abandoned_training_jobs(database)
        last_id = BASE_MESSAGE_ID + len(WARM_START_PAIRS) * 2 - 1
        complete = await database.fetchone(
            "SELECT 1 FROM archive_messages WHERE message_id = ?", (last_id,)
        )
        messages_added = 0
        if not evaluate_only and complete is None:
            messages_added += await _ingest(archive, morphology, guild_id)
        if not evaluate_only:
            messages_added += await _ingest_routing_extension(
                archive, morphology, database, guild_id
            )
        if not evaluate_only and messages_added:
            print(f"warm_start_messages={messages_added}")
            print(f"semantic_rebuilt={await semantic.rebuild_archive(guild_id)}")
            print(f"discourse_rebuilt={await discourse.rebuild_archive(guild_id)}")
        elif evaluate_only or complete is not None:
            print(f"warm_start_messages={'already_present' if complete else 'not_present'}")
        if not evaluate_only:
            examples_added = await _install_dataset(dataset, database, guild_id)
            print(f"instruction_examples_added={examples_added}")
            examples_confirmed = await _confirm_dataset(dataset, database, guild_id)
            print(f"instruction_examples_confirmed={examples_confirmed}")
        await _report(database, archive, guild_id, backup)
    finally:
        await database.close()


async def _ingest(
    archive: ArchiveRepository, morphology: MorphologyRepository, guild_id: int
) -> int:
    total = len(WARM_START_PAIRS) * 2
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
    for pair_index, (prompt, response) in enumerate(WARM_START_PAIRS):
        offset = pair_index * 2
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
    await archive.advance_job(
        job.job_id,
        last_message_id=BASE_MESSAGE_ID + total - 1,
        seen=total,
        processed=inserted,
    )
    await archive.complete_job(job.job_id)
    return inserted


async def _ingest_routing_extension(
    archive: ArchiveRepository,
    morphology: MorphologyRepository,
    database: Database,
    guild_id: int,
) -> int:
    routing_pairs = WARM_START_PAIRS[-ROUTING_PAIR_COUNT:]
    missing: list[tuple[int, str, str]] = []
    for pair_index, (prompt, response) in enumerate(routing_pairs):
        present = await database.fetchone(
            """SELECT 1
               FROM archive_messages AS prompt
               JOIN archive_messages AS response
                 ON response.guild_id = prompt.guild_id
                AND response.channel_id = prompt.channel_id
                AND response.reply_to_message_id = prompt.message_id
               WHERE prompt.guild_id = ? AND prompt.channel_id = ?
                 AND prompt.content = ? AND response.content = ?
               LIMIT 1""",
            (guild_id, CHANNEL_ID, prompt, response),
        )
        if present is None:
            missing.append((pair_index, prompt, response))
    if not missing:
        return 0
    job = await archive.create_job(
        guild_id=guild_id,
        channel_id=CHANNEL_ID,
        requested_by=TEACHER_ID,
        after_at=STARTED_AT - timedelta(days=1),
        before_at=STARTED_AT + timedelta(days=2),
        max_messages=len(missing) * 2,
        delete_raw_after=False,
    )
    await archive.mark_running(job.job_id)
    inserted = 0
    latest_id = ROUTING_BASE_MESSAGE_ID
    for pair_index, prompt, response in missing:
        prompt_id = ROUTING_BASE_MESSAGE_ID + pair_index * 2
        response_id = prompt_id + 1
        latest_id = response_id
        moment = STARTED_AT + timedelta(hours=3, minutes=pair_index * 2)
        for message_id, author_id, text, parent_id, is_bot, observed_at in (
            (prompt_id, TEACHER_ID, prompt, None, False, moment),
            (response_id, BOT_ID, response, prompt_id, True, moment + timedelta(seconds=5)),
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
                        author_id=author_id,
                        created_at=observed_at,
                        content=text,
                        author_is_bot=False,
                        reply_to_message_id=parent_id,
                    )
                )
    await archive.advance_job(
        job.job_id,
        last_message_id=latest_id,
        seen=len(missing) * 2,
        processed=inserted,
    )
    await archive.complete_job(job.job_id)
    return inserted


async def _install_dataset(
    dataset: SelfDatasetRepository,
    database: Database,
    guild_id: int,
) -> int:
    added = 0
    for pair_index, (prompt, response) in enumerate(WARM_START_PAIRS):
        source_event_id = f"curriculum:warm-start:{pair_index}"
        present = await database.fetchone(
            """SELECT 1 FROM learning_example_sources
               WHERE guild_id = ? AND source_event_id = ?""",
            (guild_id, source_event_id),
        )
        if present is not None:
            continue
        prompt_id = BASE_MESSAGE_ID + pair_index * 2
        response_id = prompt_id + 1
        moment = STARTED_AT + timedelta(minutes=pair_index * 2)
        event = normalize_message_created(
            message_id=prompt_id,
            guild_id=guild_id,
            channel_id=CHANNEL_ID,
            author_id=TEACHER_ID,
            created_at=moment,
            content=prompt,
            author_is_bot=False,
            bot_mentioned=True,
        )
        event = replace(
            event,
            event_id=source_event_id,
            source="warm_start_curriculum",
        )
        example_id = await dataset.capture_exchange(
            event,
            response_message_id=response_id,
            reply=PreparedReply(
                text=response,
                intent_id=f"curriculum:warm-start:{pair_index}",
                tokens=tokenize(response),
                genome_signature="curriculum:warm-start",
                candidate_count=1,
            ),
            created_at=moment + timedelta(seconds=5),
        )
        if example_id is None:
            continue
        await dataset.record_feedback(
            example_id=example_id,
            feedback_event_id=f"curriculum-feedback:warm-start:{pair_index}",
            kind="curriculum_approval",
            score=0.9,
            observed_at=moment + timedelta(seconds=6),
        )
        added += 1
    return added


async def _confirm_dataset(
    dataset: SelfDatasetRepository,
    database: Database,
    guild_id: int,
) -> int:
    rows = await database.fetchall(
        """SELECT source.source_event_id, example.example_id
           FROM learning_example_sources AS source
           JOIN learning_instruction_examples AS example
             ON example.example_id = source.example_id
           WHERE source.guild_id = ?
             AND source.source_event_id LIKE 'curriculum:warm-start:%'
             AND example.status <> 'accepted'""",
        (guild_id,),
    )
    confirmed = 0
    for row in rows:
        source_event_id = str(row["source_event_id"])
        confirmed += int(
            await dataset.record_feedback(
                example_id=str(row["example_id"]),
                feedback_event_id=f"curriculum-confirmation:{source_event_id}",
                kind="curriculum_confirmation",
                score=0.95,
                observed_at=STARTED_AT + timedelta(days=1),
            )
        )
    return confirmed


async def _report(
    database: Database,
    archive: ArchiveRepository,
    guild_id: int,
    backup: Path | None,
) -> None:
    raw = await database.fetchone("SELECT COUNT(*) AS count FROM archive_messages")
    print(f"backup={backup}")
    print(f"schema={await database.schema_version()}")
    print(f"warm_start_pairs={len(WARM_START_PAIRS)}")
    print(f"warm_start_skills={len(WARM_START_SKILLS)}")
    print(f"raw_messages={int(raw['count']) if raw else 0}")
    print("warm_start_probes:")
    for prompt in (
        "Привет",
        "Ответь коротко",
        "Предположи",
        "Я не это имела в виду",
        "Не повторяйся",
        "Как ты решаешь, где искать информацию?",
        "Что значит слово «кринж»?",
        "Расскажи про озеро Байкал",
        "Фырбуль — это маленький складной фонарь",
        "Что означает ключ в фразе «я открыл дверь ключом»?",
        "Словарь и Википедия объясняют это по-разному",
    ):
        answer = await archive.exact_learned_reply(guild_id, tokenize(prompt))
        print(f"- {prompt} -> {answer}")


async def _fail_abandoned_training_jobs(database: Database) -> None:
    await database.execute(
        """UPDATE archive_import_jobs SET
            status = 'failed', last_error = 'abandoned warm-start training attempt',
            updated_at = CURRENT_TIMESTAMP
           WHERE channel_id = ? AND requested_by = ? AND status = 'running'
             AND messages_seen = 0 AND messages_processed = 0""",
        (CHANNEL_ID, TEACHER_ID),
    )


def _backup(database_path: Path) -> Path | None:
    if not database_path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = database_path.with_name(f"{database_path.stem}-before-warm-start-{stamp}.sqlite3")
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
