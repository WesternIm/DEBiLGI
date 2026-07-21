from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.position_reasoning import (
    PositionDecision,
    PositionReasoningService,
    PositionRepository,
)
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "positions.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def _decision() -> PositionDecision:
    return PositionDecision(
        position_key="position:test",
        target="бобун",
        quality="лох",
        stance="support",
        score=0.82,
        confidence=0.84,
        evidence=("grounded_fact",),
        act_kind="opinion_question",
        response_obligation="express_position",
        move="state_position",
        expression_mode="direct",
    )


async def test_expressed_position_is_recalled_without_storing_its_sentence(
    database: Database,
) -> None:
    repository = PositionRepository(database)
    decision = _decision()
    await repository.record_expressed(
        decision,
        guild_id=1,
        source_event_id="discord:prompt:1",
        candidate_id="candidate:position:1",
        response_message_id=900,
        observed_at=NOW,
    )

    recalled = await repository.recall(1, decision.position_key)
    row = await database.fetchone(
        "SELECT * FROM language_expressed_positions WHERE response_message_id = 900"
    )

    assert recalled is not None
    assert recalled.stance == "support"
    assert recalled.evidence == ("grounded_fact",)
    assert row is not None
    assert "text" not in row


async def test_feedback_adjusts_confidence_and_expression_utility_without_flipping_truth(
    database: Database,
) -> None:
    repository = PositionRepository(database)
    decision = _decision()
    await repository.record_expressed(
        decision,
        guild_id=1,
        source_event_id="discord:prompt:1",
        candidate_id="candidate:position:1",
        response_message_id=900,
        observed_at=NOW,
    )
    feedback = Event(
        event_id="discord:feedback:1",
        guild_id=1,
        channel_id=10,
        actor_id=20,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW,
        payload={
            "content": "Нет, это неправильно",
            "reply_to_message_id": 900,
            "author_is_bot": False,
        },
        source="discord",
    )

    await repository.observe_feedback(feedback)

    recalled = await repository.recall(1, decision.position_key)
    preference = await database.fetchone(
        """SELECT utility, negative_feedback
           FROM language_dialogue_intent_preferences WHERE guild_id = 1"""
    )
    assert recalled is not None
    assert recalled.stance == "support"
    assert recalled.score > 0.0
    assert recalled.confidence < decision.confidence
    assert preference is not None
    assert float(preference["utility"]) < 0.0
    assert int(preference["negative_feedback"]) == 1
    assert (
        await repository.preferred_expression_mode(
            1,
            act_kind=decision.act_kind,
            response_obligation=decision.response_obligation,
            move=decision.move,
        )
        == "qualified"
    )


async def test_reasoner_uses_a_durable_position_as_one_piece_of_evidence(
    database: Database,
) -> None:
    repository = PositionRepository(database)
    frame = DialogueActFrame(
        "opinion_question",
        "express_position",
        None,
        None,
        ("бобун", "лох"),
        0.0,
        "neutral",
        True,
        0.9,
        (),
    )
    decision = await PositionReasoningService().resolve(
        1,
        frame,
        facts=("Бобун — лох.",),
    )
    assert decision is not None
    await repository.record_expressed(
        decision,
        guild_id=1,
        source_event_id="discord:prompt:1",
        candidate_id=None,
        response_message_id=900,
        observed_at=NOW,
    )
    resolved = await PositionReasoningService(repository).resolve(1, frame)

    assert resolved is not None
    assert resolved.stance == "support"
    assert "stored_position" in resolved.evidence
