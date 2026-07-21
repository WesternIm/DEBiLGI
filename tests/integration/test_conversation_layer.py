from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.normalizer import normalize_message_created
from debilgi.language.conversation import (
    ConversationRepository,
    ResponsePlanRepository,
    validate_response,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "conversation.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def message(message_id: int, content: str, *, offset: int = 0):
    return normalize_message_created(
        message_id=message_id,
        guild_id=1,
        channel_id=10,
        author_id=message_id % 2 + 1,
        created_at=NOW + timedelta(seconds=offset),
        content=content,
        author_is_bot=False,
    )


async def test_active_conversation_keeps_causal_summary_and_consolidates(
    database: Database,
) -> None:
    repository = ConversationRepository(
        database,
        idle_timeout=timedelta(minutes=30),
        max_idle_timeout=timedelta(minutes=60),
    )
    await repository.observe(message(1, "Нужен эталонный словарь для нашего сленга"))
    await repository.observe(
        message(
            2,
            "Словарь нужен потому, что собственные слова должны оставаться понятными людям",
            offset=20,
        )
    )

    state = await repository.active(1, 10)

    assert state is not None
    assert state.message_count == 2
    assert "потому" in state.current_summary
    assert "понятными людям" in state.current_summary

    assert await repository.close_stale(NOW + timedelta(minutes=46)) == 1
    assert await repository.active(1, 10) is None
    rows = await database.fetchall("SELECT * FROM language_conversation_episodes")
    assert len(rows) == 1
    assert "понятными людям" in str(rows[0]["summary"])


async def test_response_plan_requires_all_key_points(database: Database) -> None:
    plans = ResponsePlanRepository(database)
    plan = await plans.create(
        source_event_id="event:1",
        author_id=1,
        conversation_id=None,
        topic_ids=("словарь",),
        message_intent="explanation",
        response_goal="объяснить роль словаря",
        key_points=(
            "словарь является эталоном",
            "личная интерпретация хранится отдельно",
        ),
        tone="interested",
        unknowns=(),
        confidence=0.82,
        now=NOW,
    )

    incomplete = validate_response(plan, "Словарь является эталоном.")
    complete = validate_response(
        plan,
        "Словарь является эталоном, а личная интерпретация хранится отдельно.",
    )

    assert not incomplete.valid
    assert incomplete.missing_points == ("личная интерпретация хранится отдельно",)
    assert complete.valid
