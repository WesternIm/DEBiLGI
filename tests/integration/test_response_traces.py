from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 29, 6, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "response-traces.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


async def test_response_trace_keeps_the_whole_pipeline_under_one_event(
    database: Database,
) -> None:
    event = Event(
        event_id="discord:message:trace",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "Ты герой что ли?", "author_is_bot": False},
        source="test",
    )
    await EventRepository(database).append(event)
    traces = ResponseTraceRepository(database)

    await traces.begin(event, prompt="Ты герой что ли?")
    await traces.step(
        event.event_id,
        stage="interpretation",
        component="language_understanding",
        status="completed",
        summary="self-position question recognized",
        details={
            "dialogue_act": "self_state_question",
            "api_token": "must-not-be-persisted",
        },
        created_at=NOW,
    )
    await traces.step(
        event.event_id,
        stage="selection",
        component="response_arbitrator",
        status="selected",
        summary="conversational candidate selected",
        details={"winner_text": "Нет, я не считаю себя героем."},
        created_at=NOW,
    )
    await traces.finish(
        event.event_id,
        status="sent",
        selected_route="arbitrated:semantic_thought",
        final_text="Нет, я не считаю себя героем.",
        response_message_id=999,
        arbitration_decision_id="arbitration:test",
        completed_at=NOW,
    )

    root = await database.fetchone(
        "SELECT * FROM language_response_traces WHERE source_event_id = ?",
        (event.event_id,),
    )
    steps = await traces.steps_for_event(event.event_id)

    assert root is not None
    assert root["status"] == "sent"
    assert root["selected_route"] == "arbitrated:semantic_thought"
    assert root["response_message_id"] == 999
    assert [step["stage"] for step in steps] == [
        "interpretation",
        "selection",
        "completed",
    ]
    assert steps[0]["details"]["api_token"] == "[redacted]"
    assert steps[1]["details"]["winner_text"] == "Нет, я не считаю себя героем."
