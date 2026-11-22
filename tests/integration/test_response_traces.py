from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database


async def test_workflow_keeps_the_last_withheld_recovery_reason(database):
    from debilgi.discord.reply_workflow import ReplyWorkflow

    event = Event(
        "withheld",
        1,
        10,
        100,
        None,
        EventType.MESSAGE_CREATED,
        NOW,
        {"content": "привет", "bot_mentioned": True},
        "test",
    )
    await EventRepository(database).append(event)
    traces = ResponseTraceRepository(database)

    async def withhold(*args, **kwargs):
        await traces.step(
            event.event_id,
            stage="recovery",
            component="test",
            status="withheld",
            summary="no grounded surface",
            details={"reason": "exploration_did_not_fulfill_dialogue_task"},
        )
        return None

    service = SimpleNamespace(
        prepare_bound_explanation=AsyncMock(return_value=None),
        prepare_bound_short_act=AsyncMock(return_value=None),
        prepare_guaranteed_reply=withhold,
        productive_only_replies=False,
    )
    workflow = ReplyWorkflow(
        None,
        SimpleNamespace(wait_until_processed=AsyncMock(return_value=True)),
        service,
        None,
        traces,
    )
    assert await workflow.prepare(event) is None
    row = await database.fetchone("SELECT * FROM language_response_traces")
    assert row["status"] == "withheld"
    assert row["failure_stage"] == "recovery"
    assert row["failure_reason"] == "exploration_did_not_fulfill_dialogue_task"


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


@pytest.mark.parametrize(("stage", "code", "kind"), [
    ("interpretation", "parse_failed", "parse_failure"),
    ("intent", "intention_answer_not_grounded", "missing_fact"),
    ("intent", "ambiguous_intent", "ambiguity"),
    ("realization", "changed_negation", "realization_mismatch"),
    ("delivery", "receipt_not_found", "delivery_failure"),
    ("recovery", "clarification_chain_budget_exhausted", "policy"),
    ("preparation", "no_prepared_reply", "selection_failure"),
])
async def test_terminal_reasons_remain_separate(database, stage, code, kind):
    event = Event("reason", 1, 10, 100, None, EventType.MESSAGE_CREATED, NOW,
                  {"content": "вопрос"}, "test")
    await EventRepository(database).append(event)
    traces = ResponseTraceRepository(database)
    await traces.begin(event, prompt="вопрос")
    await traces.finish(event.event_id, status="withheld", failure_stage=stage, failure_reason=code)
    diagnostic = (await traces.steps_for_event(event.event_id))[-1]["details"]["diagnostic"]
    assert diagnostic["category"] == "withheld"
    assert diagnostic["content_source"] == "none"
    assert diagnostic["withhold_reason"] == {"kind": kind, "code": code, "stage": stage}
    assert [key for key, value in diagnostic["reasons"].items() if value] == [kind]
    assert diagnostic["gate"] is None and not diagnostic["receipt"]


async def test_preparation_gate_receipt_and_feedback_are_independent(database):
    from debilgi.discord.replies import _decision_trace_details
    from debilgi.language.arbitration import (
        ArbitrationProfile,
        ResponseArbitrator,
        ResponseCandidate,
    )

    event = Event("lifecycle", 1, 10, 100, None, EventType.MESSAGE_CREATED, NOW,
                  {"content": "привет"}, "test")
    await EventRepository(database).append(event)
    traces = ResponseTraceRepository(database)
    await traces.begin(event, prompt="привет")
    candidate = ResponseCandidate.create(
        event_id=event.event_id, source="learned_dialogue_structure", renderer="observed_slots",
        text="Привет.", speech_act="greeting", source_confidence=0.9,
        metadata={"corpus_source_event_ids": ("human:1",)},
    )
    decision = ResponseArbitrator(minimum_score=0).decide_exploration(
        event_id=event.event_id, prompt="привет", focus_concepts=(), candidates=(candidate,),
        recent_replies=(), profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is not None
    await traces.step(event.event_id, stage="selection", component="test", status="selected",
                      summary="selected", details=_decision_trace_details(decision))
    await traces.step(
        event.event_id, stage="route", component="test", status="selected", summary="prepared",
        details={"text": candidate.text, "response_source": f"arbitrated:{candidate.source}"},
    )
    prepared = await traces.diagnostics_for_event(event.event_id)
    assert prepared["category"] == "social"
    assert prepared["preparation"] == "prepared"
    assert prepared["content"]["plan_basis"] == "surface_only"
    assert prepared["content"]["evidence_ids"] == ["human:1"]
    assert prepared["feedback"] == [] and not prepared["receipt"]
    await traces.step(event.event_id, stage="context_gate", component="test", status="allowed",
                      summary="allowed")
    await traces.step(event.event_id, stage="delivery", component="test", status="uncertain",
                      summary="timeout", details={"reason": "receipt_not_found"})
    uncertain = await traces.diagnostics_for_event(event.event_id)
    assert uncertain["category"] == "error" and uncertain["gate"] == "allowed"
    assert uncertain["withhold_reason"]["kind"] == "delivery_failure"
    assert not uncertain["receipt"]
    await traces.step(event.event_id, stage="send", component="test", status="sent",
                      summary="receipt", details={"response_message_id": 999})
    await traces.finish(event.event_id, status="sent", final_text=candidate.text,
                        selected_route=f"arbitrated:{candidate.source}")
    await traces.step(event.event_id, stage="feedback", component="test", status="observed",
                      summary="information", details={"outcome": "question_answer",
                                                      "external_value": None})
    sent = await traces.diagnostics_for_event(event.event_id)
    assert sent["category"] == "social" and sent["receipt"]
    assert sent["plan_id"] == prepared["plan_id"]
    assert sent["feedback"][0]["external_value"] is None
