from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.continuous import (
    ContinuousCognitionConfig,
    ContinuousCognitionService,
)
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "continuous.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def _event(index: int) -> Event:
    return Event(
        event_id=f"discord:message:continuous:{index}",
        guild_id=1,
        channel_id=10,
        actor_id=20,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW + timedelta(seconds=index),
        payload={
            "content": "волк бежит за зайцем",
            "author_is_bot": False,
            "bot_mentioned": True,
        },
        source="discord",
    )


def _inquiry_details() -> dict[str, object]:
    return {
        "inquiry_id": "semantic-inquiry:test",
        "steps": [
            {
                "question_kind": "participant_action",
                "subject": "заяц",
                "predicate": None,
                "answer_role": "predicate",
                "answer": "убегать",
                "status": "hypothesis",
                "confidence": 0.64,
            },
            {
                "question_kind": "purpose",
                "subject": "волк",
                "predicate": "бежать",
                "answer_role": "goal",
                "answer": None,
                "status": "unresolved",
                "confidence": 0.0,
            },
        ],
        "conclusions": [
            {
                "epistemic_status": "inference",
                "confidence": 0.76,
                "predicate": "преследовать",
                "roles": [["subject", "волк"], ["direct_object", "заяц"]],
            },
            {
                "epistemic_status": "hypothesis",
                "confidence": 0.64,
                "predicate": "убегать",
                "roles": [["subject", "заяц"], ["location", "волк"]],
            },
        ],
    }


async def _completed_trace(
    database: Database,
    service: ContinuousCognitionService,
    index: int,
) -> Event:
    event = _event(index)
    assert await EventRepository(database).append(event)
    assert await service.observe(event)
    traces = ResponseTraceRepository(database)
    await traces.begin(event, prompt=str(event.payload["content"]))
    await traces.step(
        event.event_id,
        stage="inquiry",
        component="semantic_inquiry",
        status="concluded",
        summary="semantic inquiry completed",
        details=_inquiry_details(),
    )
    await traces.finish(
        event.event_id,
        status="sent",
        selected_route="semantic_inquiry",
        final_text="Похоже, волк преследует зайца.",
        response_message_id=1000 + index,
    )
    return event


async def test_pulse_turns_completed_trace_into_revisable_memory(
    database: Database,
) -> None:
    service = ContinuousCognitionService(database, clock=lambda: NOW)
    event = await _completed_trace(database, service, 1)

    stats = await service.pulse_once(NOW + timedelta(seconds=2))

    assert stats.work_claimed == 1
    assert stats.hypotheses_reviewed == 2
    assert stats.gaps_recorded == 1
    assert stats.signals_recorded == 1
    work = await database.fetchone(
        """SELECT status, result_summary
           FROM continuous_cognition_work_items WHERE source_event_id = ?""",
        (event.event_id,),
    )
    assert work is not None
    assert str(work["status"]) == "completed"
    assert "hypotheses=2" in str(work["result_summary"])
    facts = await service.relevant_facts(1, ("волк", "заяц"))
    assert any("преследовать" in fact and "[inference;" in fact for fact in facts)
    assert any("убегать" in fact and "[hypothesis;" in fact for fact in facts)


async def test_repeated_independent_evidence_retains_but_does_not_certify_hypothesis(
    database: Database,
) -> None:
    service = ContinuousCognitionService(database, clock=lambda: NOW)
    for index in range(1, 4):
        await _completed_trace(database, service, index)
        await service.pulse_once(NOW + timedelta(seconds=index + 1))

    rows = await database.fetchall(
        """SELECT predicate_key, epistemic_status, positive_evidence, status, confidence
           FROM continuous_cognition_hypotheses
           ORDER BY predicate_key"""
    )
    assert len(rows) == 2
    assert all(int(row["positive_evidence"]) == 3 for row in rows)
    assert all(str(row["status"]) == "retained" for row in rows)
    assert {str(row["epistemic_status"]) for row in rows} == {"inference", "hypothesis"}
    assert all(float(row["confidence"]) < 1.0 for row in rows)


async def test_runtime_is_non_blocking_and_recovers_processing_work(
    database: Database,
) -> None:
    service = ContinuousCognitionService(
        database,
        config=ContinuousCognitionConfig(
            active_interval=timedelta(milliseconds=10),
            idle_interval=timedelta(milliseconds=20),
            work_per_pulse=2,
        ),
    )
    event = _event(1)
    assert await EventRepository(database).append(event)
    assert await service.observe(event)
    await database.execute(
        """UPDATE continuous_cognition_work_items
           SET status = 'processing' WHERE source_event_id = ?""",
        (event.event_id,),
    )

    await service.start()
    assert service.running
    recovered = await database.fetchone(
        """SELECT status, last_error FROM continuous_cognition_work_items
           WHERE source_event_id = ?""",
        (event.event_id,),
    )
    assert recovered is not None
    assert str(recovered["status"]) == "queued"
    assert str(recovered["last_error"]) == "recovered_after_restart"
    await service.stop()
    assert not service.running
