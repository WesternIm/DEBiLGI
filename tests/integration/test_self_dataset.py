import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.experience.repository import ExperienceRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 18, 14, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path) -> AsyncIterator[Database]:
    db = Database(tmp_path / "self-dataset.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def prompt(number: int, text: str) -> Event:
    return Event(
        event_id=f"discord:message:{number}",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW + timedelta(seconds=number),
        payload={"content": text, "author_is_bot": False, "bot_mentioned": True},
        source="test",
    )


def prepared(text: str) -> PreparedReply:
    return PreparedReply(
        text=text,
        intent_id="intent:test",
        tokens=tuple(text.casefold().rstrip(".").split()),
        genome_signature="genome:test",
        candidate_count=8,
    )


def feedback(number: int, response_message_id: int, text: str) -> Event:
    return Event(
        event_id=f"discord:message:feedback:{number}",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW + timedelta(minutes=number),
        payload={
            "content": text,
            "author_is_bot": False,
            "reply_to_message_id": response_message_id,
        },
        source="test",
    )


async def test_exchange_builds_instruction_example_and_graph(database: Database) -> None:
    dataset = SelfDatasetRepository(database)
    current = prompt(1, "<@123> Где лежит красный ключ?")
    reply = prepared("Красный ключ лежит в верхнем ящике.")

    example_id = await dataset.capture_exchange(
        current,
        response_message_id=900,
        reply=reply,
        created_at=NOW,
    )
    duplicate = await dataset.capture_exchange(
        current,
        response_message_id=900,
        reply=reply,
        created_at=NOW,
    )

    assert duplicate == example_id
    stats = await dataset.stats(1)
    assert stats.candidates == 1
    assert stats.graph_nodes >= 7
    assert stats.graph_edges >= 6
    row = await database.fetchone(
        "SELECT instruction_text, answer_text, occurrence_count FROM learning_instruction_examples"
    )
    assert row is not None
    assert str(row["instruction_text"]) == "где лежит красный ключ?"
    assert str(row["answer_text"]) == "Красный ключ лежит в верхнем ящике."
    assert int(row["occurrence_count"]) == 2


async def test_graph_lookup_keeps_protected_paronyms_in_separate_concepts(
    database: Database,
) -> None:
    dataset = SelfDatasetRepository(database)
    await dataset.link_explanation(
        guild_id=1,
        concept="варенье",
        explanation="сладкий ягодный продукт",
        source_event_id="lesson:jam",
        observed_at=NOW,
    )
    await dataset.link_explanation(
        guild_id=1,
        concept="варение",
        explanation="процесс тепловой обработки",
        source_event_id="lesson:cooking",
        observed_at=NOW,
    )

    jam_related = set(await dataset.related_concepts(1, "варенье"))
    process_related = set(await dataset.related_concepts(1, "варение"))

    assert "сладкий" in jam_related
    assert "процесс" not in jam_related
    assert "процесс" in process_related
    assert "сладкий" not in process_related


async def test_feedback_accepts_or_rejects_examples_and_controls_export(
    database: Database, tmp_path: Path
) -> None:
    dataset = SelfDatasetRepository(database)
    first = await dataset.capture_exchange(
        prompt(1, "что такое мороженое?"),
        response_message_id=900,
        reply=prepared("Мороженое — замороженный сладкий десерт."),
        created_at=NOW,
    )
    second = await dataset.capture_exchange(
        prompt(2, "ты обманщик?"),
        response_message_id=901,
        reply=prepared("Тартюф — комедийная пьеса Мольера."),
        created_at=NOW + timedelta(seconds=1),
    )
    assert first is not None and second is not None

    await dataset.observe(feedback(1, 900, "Это хорошо, правильно"))
    await dataset.observe(feedback(2, 901, "Ты не так понял, это неверно"))

    stats = await dataset.stats(1)
    assert stats.accepted == 1
    assert stats.rejected == 1
    output = tmp_path / "export"
    counts = await dataset.export(1, output)
    assert counts["instruction_qa"] == 1
    exported = [
        json.loads(line)
        for line in (output / "instruction_qa.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert exported[0]["instruction"] == "что такое мороженое?"
    assert exported[0]["output"].startswith("Мороженое")
    assert (output / "graph_nodes.jsonl").stat().st_size > 0
    assert (output / "graph_edges.jsonl").stat().st_size > 0


async def test_existing_utterance_experience_can_be_backfilled(database: Database) -> None:
    current = prompt(3, "почему небо синее?")
    await EventRepository(database).append(current)
    await ExperienceRepository(database).register(
        guild_id=1,
        channel_id=10,
        user_id=100,
        prompt_event_id=current.event_id,
        response_message_id=902,
        reply=prepared("Синий цвет сильнее рассеивается в атмосфере."),
        created_at=NOW,
    )
    await EventRepository(database).append(feedback(4, 902, "Да, это правильно"))
    dataset = SelfDatasetRepository(database)

    assert await dataset.backfill_experiences(1) == 1
    assert await dataset.backfill_experiences(1) == 0
    assert (await dataset.stats(1)).accepted == 1
