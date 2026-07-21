"""Persistence and inference tests for the causal graph."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.language.causal_reasoning import (
    CausalReasoningRepository,
    analyze_causal_query,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 20, 15, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "causal.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(event_id: str, content: str) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": content, "author_is_bot": False},
        source="test",
    )


async def test_observed_cause_answers_why_in_its_own_orientation(database: Database) -> None:
    repository = CausalReasoningRepository(database)
    await repository.observe(
        event("cause:1", "Лампа перегорела, поэтому свет погас.")
    )
    query = analyze_causal_query("Почему свет погас?")

    assert query is not None
    inference = await repository.infer(1, query)

    assert inference is not None
    assert inference.direction == "cause"
    assert inference.text == "Свет погас, потому что лампа перегорела."
    assert len(inference.path) == 1


async def test_conditional_link_answers_possible_consequence(database: Database) -> None:
    repository = CausalReasoningRepository(database)
    await repository.observe(
        event("cause:2", "Если лампа перегорит, то свет погаснет.")
    )
    query = analyze_causal_query("Что будет если лампа перегорит?")

    assert query is not None
    inference = await repository.infer(1, query)

    assert inference is not None
    assert inference.text == "Если лампа перегорит, то свет погаснет."


async def test_multi_step_consequences_follow_oriented_path(database: Database) -> None:
    repository = CausalReasoningRepository(database)
    await repository.observe(event("cause:3", "Искра вызывает пожар."))
    await repository.observe(event("cause:4", "Пожар вызывает дым."))
    query = analyze_causal_query("К чему приведёт искра?")

    assert query is not None
    inference = await repository.infer(1, query)

    assert inference is not None
    assert len(inference.path) == 2
    assert "Искра вызывает пожар." in inference.text
    assert "Пожар вызывает дым." in inference.text


async def test_explicit_negation_blocks_unconfirmed_link(database: Database) -> None:
    repository = CausalReasoningRepository(database)
    await repository.observe(event("cause:5", "Дождь вызывает наводнение."))
    await repository.observe(event("cause:6", "Дождь не вызывает наводнение."))
    query = analyze_causal_query("К чему приводит дождь?")

    assert query is not None
    assert await repository.infer(1, query) is None


async def test_compositional_physical_schema_fills_missing_graph_link(
    database: Database,
) -> None:
    repository = CausalReasoningRepository(database)
    query = analyze_causal_query("Что будет, если толкнуть книгу со стола?")

    assert query is not None
    inference = await repository.infer(1, query)

    assert inference is not None
    assert inference.text == "Книга, скорее всего, упадёт."
    assert inference.path == ()
    assert inference.evidence == ("schema:force_displacement_removes_support",)


async def test_rebuild_is_idempotent_for_the_same_event(database: Database) -> None:
    repository = CausalReasoningRepository(database)
    observed = event("cause:7", "Лампа перегорела, поэтому свет погас.")
    await repository.observe(observed)
    await repository.observe(observed)

    row = await database.fetchone(
        "SELECT positive_evidence FROM language_causal_links"
    )
    evidence = await database.fetchone(
        "SELECT COUNT(*) AS count FROM language_causal_evidence"
    )
    assert row is not None and int(row["positive_evidence"]) == 1
    assert evidence is not None and int(evidence["count"]) == 1
