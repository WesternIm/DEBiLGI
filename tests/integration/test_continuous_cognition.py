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


async def _response_proposal(database, *, confirmation="polar"):
    import json
    from dataclasses import asdict

    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    prompt = _event(0)
    await EventRepository(database).append(prompt)
    thought = ClauseThought(
        ConceptLexeme("отдыхать", "отдыхать", "verb", (("tense", "past"),)),
        (
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme("ты", "ты", "pronoun", participant_ref="discord:user:20")
                ),
            ),
        ),
    )
    service = ContinuousCognitionService(database)
    identifier = await service.record_response_hypothesis(
        prompt,
        {
            "predicate": "отдыхать",
            "proposition": json.dumps(asdict(thought)),
            "confirmation": confirmation,
        },
        0.95,
        700,
    )
    return service, identifier


@pytest.mark.parametrize("confirmation", ["polar", "none"])
async def test_response_weight_and_bot_repetitions_are_not_facts(database, confirmation):
    from dataclasses import replace

    from debilgi.language.proposition_memory import PropositionMemory

    service, identifier = await _response_proposal(database, confirmation=confirmation)
    for i in range(1, 5):
        repeated = replace(
            _event(i),
            payload={
                "content": "да",
                "author_is_bot": True,
                "reply_to_message_id": 700,
            },
        )
        await EventRepository(database).append(repeated)
        assert await service.observe_response_hypotheses(repeated) == 0
    assert not await PropositionMemory(database).review_response_hypothesis(
        identifier, _event(5), promote=True
    )
    row = await database.fetchone(
        "SELECT * FROM continuous_cognition_hypotheses WHERE hypothesis_id=?", (identifier,)
    )
    assert row["epistemic_status"] == "hypothesis" and row["status"] == "working"
    assert row["positive_evidence"] == row["negative_evidence"] == 0
    assert not await database.fetchone("SELECT 1 FROM meaning_claims")


@pytest.mark.parametrize(
    "text,confirmation,promoted",
    [
        ("да", "polar", True),
        ("да", "none", False),
        ("не, ты чё", "polar", False),
    ],
)
async def test_only_explicit_polar_confirmation_promotes_response_proposal(
    database,
    text,
    confirmation,
    promoted,
):
    from dataclasses import replace

    service, identifier = await _response_proposal(database, confirmation=confirmation)
    feedback = replace(
        _event(1),
        payload={
            "content": text,
            "author_is_bot": False,
            "reply_to_message_id": 700,
        },
    )
    await EventRepository(database).append(feedback)
    await service.observe_response_hypotheses(feedback)
    await service.observe_response_hypotheses(feedback)  # event identity is idempotent
    claims = await database.fetchall("SELECT * FROM meaning_claims WHERE resolved=1")
    assert bool(claims) == promoted
    if promoted:
        assert claims[0]["predicate"] == "отдыхать"
        sources = await database.fetchall("SELECT source_event_id FROM meaning_evidence")
        assert [r[0] for r in sources] == [feedback.event_id]
        again = replace(
            _event(3),
            payload={
                "content": "да",
                "author_is_bot": False,
                "reply_to_message_id": 700,
            },
        )
        await EventRepository(database).append(again)
        await service.observe_response_hypotheses(again)
        assert (await database.fetchone("SELECT COUNT(*) FROM meaning_claims"))[0] == 1
        rejection = replace(
            _event(4),
            payload={
                "content": "не, ты чё",
                "author_is_bot": False,
                "reply_to_message_id": 700,
            },
        )
        await EventRepository(database).append(rejection)
        await service.observe_response_hypotheses(rejection)
        assert not await database.fetchone("SELECT 1 FROM meaning_claims WHERE resolved=1")
    state = await database.fetchone(
        "SELECT positive_evidence,negative_evidence FROM continuous_cognition_hypotheses "
        "WHERE hypothesis_id=?",
        (identifier,),
    )
    assert max(state) <= 1


@pytest.mark.parametrize(
    "actors,matching_time,expected",
    [
        ((20, 20, 20), True, False),
        ((20, 21, 20), True, True),
        ((20, 21, 20), False, False),
    ],
)
async def test_independent_confirmation_requires_people_events_and_same_time(
    database,
    actors,
    matching_time,
    expected,
):
    import json
    from dataclasses import replace

    from debilgi.language.proposition_memory import PropositionMemory

    _, identifier = await _response_proposal(database)
    row = await database.fetchone(
        "SELECT roles_json FROM continuous_cognition_hypotheses WHERE hypothesis_id=?",
        (identifier,),
    )
    thought = json.loads(json.loads(row[0])["descriptor"]["proposition"])
    if not matching_time:
        thought["semantic_time"]["tense"] = "future"
    for i, actor in enumerate(actors, 1):
        observation = replace(_event(i), actor_id=actor)
        await EventRepository(database).append(observation)
        # Canonical observations stand for different human reports about this
        # resolved participant; evidence must be checked against their claims.
        await database.execute(
            "INSERT INTO meaning_claims "
            "(claim_id,source_event_id,guild_id,channel_id,speaker_id,subject_ref,predicate,polarity,"
            "modality_json,source_kind,source_time,valid_from,valid_to,"
            "confidence,resolved,parser_version,thought_json,correction_target,is_correction) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"observation:{i}",
                observation.event_id,
                1,
                10,
                actor,
                "discord:user:20",
                "отдыхать",
                1,
                "[]",
                "other_report",
                observation.utc_timestamp.isoformat(),
                None,
                None,
                0.9,
                1,
                "meaning-ir/1",
                json.dumps(thought),
                None,
                0,
            ),
        )
        await database.execute(
            "INSERT INTO continuous_cognition_evidence VALUES (?,?,?,?,?,?,?)",
            (
                f"proof:{i}",
                identifier,
                observation.event_id,
                "supports",
                1.0,
                json.dumps({"canonical_claim": f"observation:{i}"}),
                observation.utc_timestamp.isoformat(),
            ),
        )
        promoted = await PropositionMemory(database).review_response_hypothesis(
            identifier, observation, promote=True
        )
        assert promoted == (expected and i == 3)
    proof = await database.fetchall(
        "SELECT * FROM meaning_evidence WHERE source_key=?", ("confirmed_hypothesis:" + identifier,)
    )
    assert len(proof) == (3 if expected else 0)


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


async def test_one_author_cannot_retain_by_repeating_conclusions(
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
    assert all(int(row["positive_evidence"]) == 1 for row in rows)
    assert all(str(row["status"]) == "working" for row in rows)
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
