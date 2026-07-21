from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.arbitration import (
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
)
from debilgi.language.arbitration_repository import ArbitrationRepository
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 7, 19, 10, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    db = Database(tmp_path / "arbitration.sqlite3", MIGRATIONS)
    await db.initialize()
    try:
        yield db
    finally:
        await db.close()


def event(
    event_id: str,
    *,
    event_type: EventType = EventType.MESSAGE_CREATED,
    payload: dict[str, object] | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=event_type,
        timestamp=NOW,
        payload=payload or {"content": "А футбол любишь?"},
        source="test",
    )


async def test_decision_persists_every_candidate_and_explicit_feedback(
    database: Database,
) -> None:
    prompt = event("discord:message:prompt")
    events = EventRepository(database)
    await events.append(prompt)
    candidates = (
        ResponseCandidate.create(
            event_id=prompt.event_id,
            source="dialogue_rule",
            renderer="preference",
            text="Да, футбол мне интересен. А тебе?",
            speech_act="question",
            source_confidence=0.85,
        ),
        ResponseCandidate.create(
            event_id=prompt.event_id,
            source="graph",
            renderer="graph",
            text="Футбол — командная игра с мячом.",
            speech_act="question",
            source_confidence=0.7,
        ),
    )
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id=prompt.event_id,
        prompt="А футбол любишь?",
        focus_concepts=("футбол",),
        candidates=candidates,
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    repository = ArbitrationRepository(database)

    await repository.save_decision(
        prompt,
        decision,
        mode="shadow",
        legacy_text="По «футбол» не хватает самого действия или вопроса.",
        legacy_source="graph",
    )
    await repository.mark_sent(decision.decision_id, response_message_id=555, sent_at=NOW)

    rows = await database.fetchall(
        "SELECT * FROM language_response_candidates WHERE decision_id = ?",
        (decision.decision_id,),
    )
    stored = await database.fetchone(
        "SELECT * FROM language_arbitration_decisions WHERE decision_id = ?",
        (decision.decision_id,),
    )
    assert stored is not None
    assert stored["mode"] == "shadow"
    assert stored["legacy_source"] == "graph"
    assert stored["legacy_text_hash"] is not None
    assert stored["shadow_disagreed"] == 1
    assert stored["response_message_id"] == 555
    assert len(rows) == 2
    assert sum(int(row["selected"]) for row in rows) == 1

    feedback = event(
        "discord:message:feedback",
        event_type=EventType.MESSAGE_REPLIED,
        payload={"content": "нет", "reply_to_message_id": 555},
    )
    await events.append(feedback)
    await repository.observe_feedback(feedback)

    feedback_row = await database.fetchone(
        "SELECT * FROM language_arbitration_feedback WHERE feedback_event_id = ?",
        (feedback.event_id,),
    )
    assert feedback_row is not None
    assert feedback_row["reward"] < 0
    assert feedback_row["applied_at"] is None


async def test_enabled_learning_updates_a_versioned_profile_after_bound_feedback(
    database: Database,
) -> None:
    prompt = event("discord:message:learning-prompt")
    events = EventRepository(database)
    await events.append(prompt)
    candidates = (
        ResponseCandidate.create(
            event_id=prompt.event_id,
            source="dialogue_rule",
            renderer="relevant",
            text="Футбол мне интересен. А тебе?",
            speech_act="question",
            source_confidence=0.9,
        ),
        ResponseCandidate.create(
            event_id=prompt.event_id,
            source="graph",
            renderer="alternative",
            text="Телесериал вышел в 1987 году.",
            speech_act="question",
            source_confidence=0.6,
        ),
    )
    decision = ResponseArbitrator(minimum_score=0.0).decide(
        event_id=prompt.event_id,
        prompt="А футбол любишь?",
        focus_concepts=("футбол",),
        candidates=candidates,
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    repository = ArbitrationRepository(
        database,
        learning_enabled=True,
        minimum_feedback_observations=1,
    )
    await repository.save_decision(prompt, decision, mode="active")
    await repository.mark_sent(decision.decision_id, response_message_id=777, sent_at=NOW)
    feedback = event(
        "discord:message:learning-feedback",
        event_type=EventType.MESSAGE_REPLIED,
        payload={"content": "да", "reply_to_message_id": 777},
    )
    await events.append(feedback)

    await repository.observe_feedback(feedback)

    profile = await repository.profile(1)
    feedback_row = await database.fetchone(
        "SELECT * FROM language_arbitration_feedback WHERE feedback_event_id = ?",
        (feedback.event_id,),
    )
    assert profile.version == 2
    assert profile.observations == 1
    assert feedback_row is not None
    assert feedback_row["new_version"] == 2
    assert feedback_row["applied_at"] is not None
    learned_weights = await repository.structural_weights(1)
    stored_features = await database.fetchall(
        "SELECT * FROM language_candidate_structural_features WHERE guild_id = 1"
    )
    assert learned_weights
    assert len(stored_features) == 2
    assert any(key.startswith("cross:") for key in learned_weights)


async def test_feedback_reinforces_selected_abstract_construction(
    database: Database,
) -> None:
    construction_id = "construction:test-feedback"
    await database.execute(
        """INSERT INTO language_productive_constructions (
               guild_id, construction_id, clause_kind, predicate_class,
               roles_json, required_roles_json, optional_roles_json,
               negation_mode, confidence, source, created_at, updated_at
           ) VALUES (1, ?, 'statement', 'lexical', ?, ?, ?, 'forbidden',
                     0.8, 'local_observation', ?, ?)""",
        (
            construction_id,
            '["subject","predicate","direct_object"]',
            '["subject","predicate"]',
            '["direct_object"]',
            NOW.isoformat(),
            NOW.isoformat(),
        ),
    )
    prompt = event("discord:message:construction-prompt")
    events = EventRepository(database)
    await events.append(prompt)
    candidate = ResponseCandidate.create(
        event_id=prompt.event_id,
        source="semantic_reasoning",
        renderer="productive_dependencies:local_observation",
        text="Да, я понял идею.",
        speech_act="statement",
        source_confidence=0.9,
        metadata={"construction_id": construction_id},
    )
    decision = ResponseArbitrator(minimum_score=-1.0).decide(
        event_id=prompt.event_id,
        prompt="Ты понял идею?",
        focus_concepts=("идея",),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    repository = ArbitrationRepository(database)
    await repository.save_decision(prompt, decision, mode="active")
    await repository.mark_sent(
        decision.decision_id,
        response_message_id=991,
        sent_at=NOW,
    )
    feedback = event(
        "discord:message:construction-feedback",
        event_type=EventType.MESSAGE_REPLIED,
        payload={"content": "да", "reply_to_message_id": 991},
    )
    await events.append(feedback)

    await repository.observe_feedback(feedback)

    row = await database.fetchone(
        """SELECT utility, positive_feedback, negative_feedback
           FROM language_productive_constructions
           WHERE guild_id = 1 AND construction_id = ?""",
        (construction_id,),
    )
    assert row is not None
    assert float(row["utility"]) > 0.0
    assert int(row["positive_feedback"]) == 1
    assert int(row["negative_feedback"]) == 0
