import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.diagnostics.quality_loop import LearningQualityLoop
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.lexicon import LexiconService
from debilgi.language.processing_outcomes import ProcessingOutcomeRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.language.semantic_reply_plan import ProcessingOutcome, ProcessingOutcomeType
from debilgi.storage.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = PROJECT_ROOT / "migrations"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
async def database(tmp_path: Path):
    database = Database(tmp_path / "learning-quality.sqlite3", MIGRATIONS)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


async def test_negative_feedback_routes_unknown_terms_into_semantic_study(
    database: Database,
) -> None:
    await database.execute(
        """INSERT INTO language_utterance_experiences (
               experience_id, guild_id, channel_id, user_id, prompt_event_id,
               intent_id, response_message_id, response_tokens_json,
               genome_signature, candidate_count, status, outcome_type, reward,
               created_at, resolved_at, expected_reward
           ) VALUES (?, 1, 10, 100, ?, 'intent:test', 999, '["ответ"]',
                     'genome:test', 1, 'resolved', 'explicit_rejection', -1.0,
                     ?, ?, 0.2)""",
        ("experience:test", "prompt:test", NOW.isoformat(), NOW.isoformat()),
    )
    await database.execute(
        """INSERT INTO language_utterance_understandings (
               event_id, guild_id, channel_id, content_terms_json, known_terms_json,
               unknown_terms_json, relations_json, context_links_json,
               comprehension_score, sufficient, observed_at
           ) VALUES ('prompt:test', 1, 10, '["трошка"]', '[]', '["трошка"]',
                     '[]', '[]', 0.1, 0, ?)""",
        (NOW.isoformat(),),
    )
    feedback = Event(
        event_id="feedback:test",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW,
        payload={
            "content": "ты не понял",
            "reply_to_message_id": 999,
            "author_is_bot": False,
        },
        source="test",
    )

    quality_loop = LearningQualityLoop(database, LexiconService(database))
    await quality_loop.observe_feedback(feedback)

    quality_case = await database.fetchone(
        "SELECT * FROM learning_quality_cases WHERE source_id = 'feedback:test'"
    )
    assert quality_case is not None
    assert quality_case["category"] == "text_understanding"
    assert quality_case["status"] == "monitoring"
    assert json.loads(str(quality_case["details_json"]))["unknown_terms"] == ["трошка"]
    study = await database.fetchone(
        """SELECT * FROM language_semantic_study_queue
           WHERE guild_id = 1 AND channel_id = 10 AND concept_key = 'трошка'"""
    )
    assert study is not None
    assert study["reason"] == "negative_feedback_understanding"
    actions = await database.fetchall(
        """SELECT action_type, status FROM learning_quality_actions
           WHERE case_id = ? ORDER BY action_type""",
        (quality_case["case_id"],),
    )
    assert {(row["action_type"], row["status"]) for row in actions} == {
        ("semantic_study", "queued"),
        ("transition_penalty", "applied"),
    }

    await database.execute(
        """INSERT INTO language_utterance_experiences (
               experience_id, guild_id, channel_id, user_id, prompt_event_id,
               intent_id, response_message_id, response_tokens_json,
               genome_signature, candidate_count, status, outcome_type, reward,
               created_at, resolved_at, expected_reward
           ) VALUES ('experience:repair', 1, 10, 100, 'prompt:repair',
                     'intent:repair', 1000, '["ответ"]', 'genome:test', 1,
                     'resolved', 'explicit_confirmation', 1.0, ?, ?, 0.4)""",
        (NOW.isoformat(), NOW.isoformat()),
    )
    await database.execute(
        """INSERT INTO language_utterance_understandings (
               event_id, guild_id, channel_id, content_terms_json, known_terms_json,
               unknown_terms_json, relations_json, context_links_json,
               comprehension_score, sufficient, observed_at
           ) VALUES ('prompt:repair', 1, 10, '["трошка"]', '["трошка"]', '[]',
                     '[]', '[]', 0.9, 1, ?)""",
        (NOW.isoformat(),),
    )
    repaired = Event(
        event_id="feedback:repair",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW,
        payload={
            "content": "да, правильно",
            "reply_to_message_id": 1000,
            "author_is_bot": False,
        },
        source="test",
    )

    await quality_loop.observe_feedback(repaired)

    repaired_case = await database.fetchone(
        "SELECT status, resolved_at FROM learning_quality_cases WHERE case_id = ?",
        (quality_case["case_id"],),
    )
    assert repaired_case is not None
    assert repaired_case["status"] == "resolved"
    assert repaired_case["resolved_at"] == NOW.isoformat()
    assert await database.fetchone(
        """SELECT 1 FROM learning_quality_actions
           WHERE case_id = ? AND action_type = 'positive_repair_verification'""",
        (quality_case["case_id"],),
    )


async def test_processing_failure_is_journaled_through_repository_observer(
    database: Database,
) -> None:
    event = Event(
        event_id="prompt:pipeline-failure",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "объясни штуковину", "author_is_bot": False},
        source="test",
    )
    await EventRepository(database).append(event)
    await database.execute(
        """INSERT INTO language_utterance_understandings (
               event_id, guild_id, channel_id, content_terms_json, known_terms_json,
               unknown_terms_json, relations_json, context_links_json,
               comprehension_score, sufficient, observed_at
           ) VALUES (?, 1, 10, '["объяснить","штуковина"]', '["объяснить"]',
                     '["штуковина"]', '[]', '[]', 0.3, 0, ?)""",
        (event.event_id, NOW.isoformat()),
    )
    quality_loop = LearningQualityLoop(database, LexiconService(database))
    outcomes = ProcessingOutcomeRepository(database)
    outcomes.set_quality_observer(quality_loop.observe_processing_outcome)

    await outcomes.save(
        event,
        ProcessingOutcome(
            outcome_type=ProcessingOutcomeType.FAILED_TO_FORM_PLAN,
            intention_id="intent:failed",
            reply_plan_id=None,
            generated_text=None,
            failure_stage="planning",
            failure_reason="missing semantic relation",
            was_intentional=False,
        ),
        response_message_id=None,
        created_at=NOW,
    )

    quality_case = await database.fetchone(
        """SELECT * FROM learning_quality_cases
           WHERE prompt_event_id = ? AND source_kind = 'pipeline'""",
        (event.event_id,),
    )
    assert quality_case is not None
    assert quality_case["category"] == "text_understanding"
    assert quality_case["status"] == "monitoring"
    assert await database.fetchone(
        """SELECT 1 FROM language_semantic_study_queue
           WHERE concept_key = 'штуковина' AND reason = 'pipeline_understanding_failure'"""
    )


async def test_terminal_trace_withholding_is_journaled_by_quality_loop(
    database: Database,
) -> None:
    event = Event(
        event_id="prompt:productive-withholding",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": "Зачем?", "author_is_bot": False},
        source="test",
    )
    await EventRepository(database).append(event)
    quality_loop = LearningQualityLoop(database, LexiconService(database))
    traces = ResponseTraceRepository(database)
    traces.set_quality_observer(quality_loop.observe_response_trace)

    await traces.begin(event, prompt="Зачем?")
    await traces.finish(
        event.event_id,
        status="withheld",
        failure_stage="preparation",
        failure_reason="productive_only_no_surface",
        completed_at=NOW,
    )

    quality_case = await database.fetchone(
        """SELECT * FROM learning_quality_cases
           WHERE prompt_event_id = ? AND source_kind = 'pipeline'""",
        (event.event_id,),
    )
    assert quality_case is not None
    assert quality_case["category"] == "pipeline_failure"
    assert quality_case["reason"] == "productive_only_no_surface"
    assert quality_case["status"] == "queued"
    assert await database.fetchone(
        """SELECT 1 FROM learning_quality_actions
           WHERE case_id = ? AND action_type = 'implementation_review'
             AND status = 'queued'""",
        (quality_case["case_id"],),
    )
