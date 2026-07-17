from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "discourse.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def message(
    message_id: int,
    author_id: int,
    content: str,
    *,
    reply_to: int | None = None,
    reply_author: int | None = None,
    mentions: tuple[int, ...] = (),
    offset: int = 0,
):
    return normalize_message_created(
        message_id=message_id,
        guild_id=1,
        channel_id=10,
        author_id=author_id,
        created_at=NOW + timedelta(seconds=offset),
        content=content,
        author_is_bot=False,
        reply_to_message_id=reply_to,
        reply_to_author_id=reply_author,
        mentioned_user_ids=mentions,
    )


async def test_links_addressee_question_answer_and_reply_chain(database: Database) -> None:
    repository = DiscourseMemoryRepository(database)
    question = message(100, 1, "<@2> где лежит медный ключ?", mentions=(2,))
    answer = message(
        101,
        2,
        "медный ключ лежит в ящике",
        reply_to=100,
        reply_author=1,
        offset=5,
    )
    followup = message(
        102,
        1,
        "а кто закрыл ящик?",
        reply_to=101,
        reply_author=2,
        offset=10,
    )

    await repository.observe(question)
    await repository.observe(answer)
    context = await repository.context_for(followup, "кто закрыл медный ящик")

    assert context.addressed_to_id == 2
    assert context.reply_depth == 2
    assert context.messages[:2] == (
        "где лежит медный ключ?",
        "медный ключ лежит в ящике",
    )
    assert context.reply_heads[0] == ("медный", "ключ", "лежит", "в")
    stats = await repository.stats(1)
    assert stats.turns == 2
    assert stats.open_questions == 0
    assert stats.answered_questions == 1


async def test_keeps_unanswered_question_for_its_addressee(database: Database) -> None:
    repository = DiscourseMemoryRepository(database)
    question = message(200, 4, "<@5> почему лампа погасла?", mentions=(5,))
    current = message(201, 5, "я проверяю лампу", offset=5)
    await repository.observe(question)

    context = await repository.context_for(current, "лампа")

    assert context.open_questions == ("почему лампа погасла?",)


async def test_short_implicit_answer_recalls_addressed_question(database: Database) -> None:
    repository = DiscourseMemoryRepository(database)
    question = message(300, 4, "<@5> лампа сейчас горит?", mentions=(5,))
    answer = message(301, 5, "да", offset=5)
    await repository.observe(question)
    await repository.observe(answer)

    context = await repository.context_for(answer, "да")

    assert context.messages[0] == "лампа сейчас горит?"
    assert (await repository.stats(1)).answered_questions == 1
