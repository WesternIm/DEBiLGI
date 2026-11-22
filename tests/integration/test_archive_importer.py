import asyncio
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.language.archive_importer import (
    ArchiveImporter,
    ArchiveImportManager,
)
from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.models import ArchivedMessage, ArchiveImportJob, ImportStatus
from debilgi.language.semantics import SemanticMemoryRepository
from debilgi.language.tokenizer import tokenize
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
AFTER = datetime(2026, 1, 1, tzinfo=UTC)
BEFORE = datetime(2026, 2, 1, tzinfo=UTC)


def message(
    message_id: int,
    content: str,
    *,
    reply_to: int | None = None,
    guild_id: int = 1,
    channel_id: int = 2,
) -> ArchivedMessage:
    return ArchivedMessage(
        message_id=message_id,
        guild_id=guild_id,
        channel_id=channel_id,
        author_id=message_id + 100,
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        content=content,
        reply_to_message_id=reply_to,
        author_is_bot=False,
    )


class FakeSource:
    def __init__(self, messages: Sequence[ArchivedMessage]) -> None:
        self.messages = list(messages)
        self.cursors: list[int | None] = []

    async def fetch_batch(self, job: ArchiveImportJob, limit: int) -> Sequence[ArchivedMessage]:
        self.cursors.append(job.last_message_id)
        return [
            item
            for item in self.messages
            if job.last_message_id is None or item.message_id > job.last_message_id
        ][:limit]


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "archive.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def create_job(
    repository: ArchiveRepository,
    *,
    max_messages: int = 100,
    delete_raw_after: bool = False,
) -> ArchiveImportJob:
    return await repository.create_job(
        guild_id=1,
        channel_id=2,
        requested_by=3,
        after_at=AFTER,
        before_at=BEFORE,
        max_messages=max_messages,
        delete_raw_after=delete_raw_after,
    )


async def test_ingestion_is_idempotent_and_extracts_reply_patterns(
    database: Database,
) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository)
    await repository.mark_running(job.job_id)

    assert await repository.ingest_message(job.job_id, message(10, "ну привет сука"))
    assert await repository.ingest_message(job.job_id, message(11, "привет блять 😈", reply_to=10))
    assert not await repository.ingest_message(
        job.job_id, message(11, "привет блять 😈", reply_to=10)
    )

    stats = await repository.language_stats(1)
    assert stats.raw_messages == 2
    assert stats.reply_patterns == 1
    assert stats.semantic_frames == 0  # greeting/interjection is not a factual proposition
    assert await repository.token_count(1, "привет") == 2
    assert await repository.token_count(1, "блять") == 1
    assert await repository.recent_texts(1, 2) == (
        "ну привет сука",
        "привет блять 😈",
    )
    assert (await repository.matching_reply_heads(1, tokenize("ну привет сука")))[0] == (
        "привет",
        "блять",
        "😈",
    )
    assert (await repository.matching_replies(1, tokenize("ну привет сука")))[0] == (
        "привет блять 😈"
    )
    assert await repository.exact_learned_reply(1, tokenize("ну привет сука")) == (
        "привет блять 😈"
    )
    assert (("ну", "привет"), 1) in await repository.common_bigrams(1)
    character_rows = await database.fetchall(
        "SELECT name FROM sqlite_master WHERE name = ?", ("language_character_transitions",)
    )
    assert character_rows == []  # Dead table is removed by migration 063.


async def test_reply_retrieval_ignores_pronoun_only_overlap(database: Database) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository)
    await repository.mark_running(job.job_id)
    await repository.ingest_message(job.job_id, message(10, "ты радуешься"))
    await repository.ingest_message(
        job.job_id,
        message(11, "вижу удачный исход и повышаю доверие", reply_to=10),
    )
    await repository.ingest_message(job.job_id, message(12, "ты злишься"))
    await repository.ingest_message(
        job.job_id,
        message(13, "напряжение есть, злость слишком человеческая", reply_to=12),
    )

    unrelated = tokenize("ты разумный")
    scaffolding_only = tokenize("хмм ну ты же")
    related = tokenize("почему ты радуешься")

    assert await repository.matching_replies(1, unrelated) == ()
    assert await repository.matching_reply_heads(1, unrelated) == ()
    assert await repository.matching_replies(1, scaffolding_only) == ()
    assert await repository.matching_reply_heads(1, scaffolding_only) == ()
    assert (await repository.matching_replies(1, related))[0] == (
        "вижу удачный исход и повышаю доверие"
    )
    assert await repository.exact_learned_reply(1, tokenize("ты радуешься")) == (
        "вижу удачный исход и повышаю доверие"
    )


async def test_exact_reply_distinguishes_address_punctuation(database: Database) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository)
    await repository.mark_running(job.job_id)
    await repository.ingest_message(job.job_id, message(20, "узнай, умник"))
    await repository.ingest_message(
        job.job_id, message(21, "это обращение", reply_to=20)
    )
    await repository.ingest_message(job.job_id, message(22, "узнай умник"))
    await repository.ingest_message(
        job.job_id, message(23, "это разговорная запись", reply_to=22)
    )

    assert await repository.exact_learned_reply(1, tokenize("узнай, умник")) == (
        "это обращение"
    )
    assert await repository.exact_learned_reply(1, tokenize("узнай умник")) == (
        "это разговорная запись"
    )


async def test_semantic_memory_retrieves_old_fact_by_object(database: Database) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository)
    await repository.mark_running(job.job_id)
    assert await repository.ingest_message(job.job_id, message(12, "Лиса положила ключ в ящик"))

    semantic = SemanticMemoryRepository(database)
    context = await semantic.context(1, tokenize("где находится ключ"))

    assert context
    assert "лиса положила ключ в ящик" in context


async def test_semantic_memory_does_not_return_unrelated_facts(database: Database) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository)
    await repository.mark_running(job.job_id)
    assert await repository.ingest_message(job.job_id, message(13, "Борис положил книгу на полку"))

    semantic = SemanticMemoryRepository(database)

    assert await semantic.context(1, tokenize("чего")) == ()
    assert await semantic.context(1, tokenize("не лезь к нему")) == ()


async def test_raw_deletion_keeps_idempotency_ledger_and_aggregates(
    database: Database,
) -> None:
    repository = ArchiveRepository(database)
    first_job = await create_job(repository, delete_raw_after=True)
    await repository.mark_running(first_job.job_id)
    original = message(20, "не удаляй статистику")
    assert await repository.ingest_message(first_job.job_id, original)
    await repository.complete_job(first_job.job_id)

    assert (await repository.language_stats(1)).raw_messages == 0
    assert await repository.token_count(1, "статистику") == 1
    assert await repository.common_bigrams(1)

    overlapping_job = await create_job(repository)
    await repository.mark_running(overlapping_job.job_id)
    assert not await repository.ingest_message(overlapping_job.job_id, original)
    assert await repository.token_count(1, "статистику") == 1


async def test_importer_batches_and_persists_progress(database: Database) -> None:
    repository = ArchiveRepository(database)
    job = await create_job(repository, max_messages=10)
    source = FakeSource([message(30, "раз"), message(31, "два"), message(32, "три")])

    await ArchiveImporter(repository, batch_size=2).run(job.job_id, source)

    finished = await repository.get_job(job.job_id)
    assert finished is not None
    assert finished.status is ImportStatus.COMPLETED
    assert finished.messages_seen == 3
    assert finished.messages_processed == 3
    assert source.cursors == [None, 31]


async def test_manager_resumes_running_job_from_saved_cursor_after_restart(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "restart.sqlite3"
    first_database = Database(db_path, MIGRATIONS)
    await first_database.initialize()
    first_repository = ArchiveRepository(first_database)
    job = await create_job(first_repository)
    await first_repository.mark_running(job.job_id)
    await first_repository.ingest_message(job.job_id, message(40, "первое"))
    await first_repository.advance_job(job.job_id, last_message_id=40, seen=1, processed=1)
    await first_database.close()

    second_database = Database(db_path, MIGRATIONS)
    await second_database.initialize()
    second_repository = ArchiveRepository(second_database)
    source = FakeSource([message(40, "первое"), message(41, "второе")])
    importer = ArchiveImporter(second_repository, batch_size=10)
    manager = ArchiveImportManager(second_repository, importer, lambda _job: source)
    await manager.start()
    try:
        await _wait_for_status(second_repository, job.job_id, ImportStatus.COMPLETED)
        finished = await second_repository.get_job(job.job_id)
        assert finished is not None
        assert finished.messages_seen == 2
        assert finished.messages_processed == 2
        assert source.cursors == [40]
    finally:
        await manager.stop()
        await second_database.close()


async def _wait_for_status(
    repository: ArchiveRepository,
    job_id: str,
    expected: ImportStatus,
) -> None:
    async def wait() -> None:
        while True:
            job = await repository.get_job(job_id)
            if job is not None and job.status is expected:
                return
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)
