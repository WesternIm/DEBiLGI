from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.memory.long_term import LongTermMemoryRepository
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


class ManualClock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "memory-lifecycle.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def message(
    message_id: int,
    at: datetime,
    content: str,
    *,
    mentioned: bool = False,
):
    return normalize_message_created(
        message_id=message_id,
        guild_id=1,
        channel_id=10,
        author_id=100,
        created_at=at,
        content=content,
        author_is_bot=False,
        bot_mentioned=mentioned,
    )


async def test_important_eviction_is_compressed_and_recalled_by_context(
    database: Database,
) -> None:
    clock = ManualClock()
    working = ShortTermMemory(
        MemoryLimits(retention=timedelta(minutes=1)),
        clock=clock,
    )
    long_term = LongTermMemoryRepository(
        database,
        working,
        max_fragment_age=timedelta(minutes=1),
    )
    important = message(
        1,
        clock.now,
        "Запомни важный вывод: красный ключ открывает лабораторию.",
        mentioned=True,
    )
    await working.observe(important)
    clock.now += timedelta(minutes=2)
    trigger = message(2, clock.now, "Теперь говорим о погоде")
    await working.observe(trigger)
    await long_term.observe(trigger)

    memories = await long_term.recall(
        1,
        "Какой ключ открывает лабораторию?",
        channel_id=10,
        user_id=100,
        now=clock.now,
    )

    assert len(memories) == 1
    assert "красный ключ" in memories[0].summary.casefold()
    assert "лабораторию" in memories[0].summary.casefold()
    assert await long_term.recall(1, "рецепт яблочного пирога", now=clock.now) == ()


async def test_unimportant_eviction_is_forgotten_instead_of_archived(
    database: Database,
) -> None:
    clock = ManualClock()
    working = ShortTermMemory(
        MemoryLimits(retention=timedelta(minutes=1)),
        clock=clock,
    )
    long_term = LongTermMemoryRepository(
        database,
        working,
        min_importance=0.50,
        max_fragment_age=timedelta(minutes=1),
    )
    await working.observe(message(1, clock.now, "ну ладно"))
    clock.now += timedelta(minutes=2)
    trigger = message(2, clock.now, "новая тема")
    await working.observe(trigger)
    await long_term.observe(trigger)

    stats = await long_term.stats(1)
    assert stats.capsules == 0
    assert stats.pending_fragments == 0
    assert stats.forgotten_fragments == 1
