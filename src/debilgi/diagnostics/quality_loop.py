"""Route observed answer failures into existing learning mechanisms."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from hashlib import sha256
from typing import Any

from debilgi.events.models import Event
from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
from debilgi.language.lexicon import LexiconService
from debilgi.language.response_trace import ResponseTraceTerminal
from debilgi.language.semantic_reply_plan import ProcessingOutcome, ProcessingOutcomeType
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.learning_quality")


class LearningQualityLoop:
    """Create durable failure cases and apply safe, already-supported learning actions."""

    def __init__(self, database: Database, lexicon: LexiconService) -> None:
        self._database = database
        self._lexicon = lexicon

    async def observe_feedback(self, event: Event) -> None:
        try:
            await self._observe_feedback(event)
        except Exception:
            logger.exception("quality_feedback_loop_failed event=%s", event.event_id)

    async def _observe_feedback(self, event: Event) -> None:
        if event.payload.get("author_is_bot") is True:
            return
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        if feedback.reward == 0.0:
            return
        experience = await self._experience_for_feedback(event, feedback.target_message_id)
        if experience is None:
            return
        context = await self._failure_context(str(experience["prompt_event_id"]))
        unknown_terms = _json_tuple(context["unknown_terms_json"] if context else None)
        content_terms = _json_tuple(context["content_terms_json"] if context else None)
        fingerprint_terms = content_terms or unknown_terms or (feedback.kind,)
        if feedback.reward > 0.10:
            await self._resolve_matching_cases(
                event.guild_id,
                (
                    _fingerprint("text_understanding", fingerprint_terms),
                    _fingerprint("response_quality", fingerprint_terms),
                    _fingerprint("pipeline_failure", fingerprint_terms),
                ),
                source_event_id=event.event_id,
                resolved_at=event.utc_timestamp,
            )
            return
        if feedback.reward >= -0.10:
            return
        insufficient = context is not None and not bool(context["sufficient"])
        category = (
            "text_understanding" if unknown_terms or insufficient else "response_quality"
        )
        fingerprint = _fingerprint(category, fingerprint_terms)
        case_id = f"quality-feedback:{event.event_id}"
        details = {
            "feedback_kind": feedback.kind,
            "reward": feedback.reward,
            "attribution_confidence": feedback.attribution_confidence,
            "comprehension_score": (
                float(context["comprehension_score"]) if context is not None else None
            ),
            "understanding_sufficient": (
                bool(context["sufficient"]) if context is not None else None
            ),
            "unknown_terms": unknown_terms,
            "content_terms": content_terms,
        }
        inserted = await self._insert_case(
            case_id=case_id,
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            category=category,
            source_kind="feedback",
            source_id=event.event_id,
            prompt_event_id=str(experience["prompt_event_id"]),
            response_message_id=int(experience["response_message_id"]),
            fingerprint=fingerprint,
            reason=feedback.kind,
            severity=min(1.0, abs(feedback.reward)),
            details=details,
            created_at=event.utc_timestamp,
        )
        if not inserted:
            return

        applied = 0
        applied += await self._record_existing_learning(case_id, experience, event.utc_timestamp)
        if unknown_terms:
            queued = await self._lexicon.queue_semantic_study(
                event,
                unknown_terms,
                reason="negative_feedback_understanding",
                priority=min(1.0, 0.75 + abs(feedback.reward) * 0.20),
            )
            for term in unknown_terms:
                await self._record_action(
                    case_id,
                    "semantic_study",
                    term,
                    "queued" if queued else "failed",
                    {"term": term, "queued": queued > 0},
                    event.utc_timestamp,
                )
            applied += int(queued > 0)
        await self._set_status(
            case_id,
            "monitoring" if applied else "queued",
            event.utc_timestamp,
        )

    async def observe_processing_outcome(
        self,
        event: Event,
        outcome: ProcessingOutcome,
        created_at: datetime,
    ) -> None:
        if outcome.outcome_type in {
            ProcessingOutcomeType.REPLIED,
            ProcessingOutcomeType.INTENTIONAL_SILENCE,
        }:
            return
        context = await self._failure_context(event.event_id)
        unknown_terms = _json_tuple(context["unknown_terms_json"] if context else None)
        content_terms = _json_tuple(context["content_terms_json"] if context else None)
        category = "text_understanding" if unknown_terms else "pipeline_failure"
        reason = outcome.failure_stage or outcome.outcome_type.value
        case_id = f"quality-pipeline:{event.event_id}:{outcome.outcome_type.value}"
        inserted = await self._insert_case(
            case_id=case_id,
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            category=category,
            source_kind="pipeline",
            source_id=f"{event.event_id}:{outcome.outcome_type.value}",
            prompt_event_id=event.event_id,
            response_message_id=None,
            fingerprint=_fingerprint(category, content_terms or unknown_terms or (reason,)),
            reason=reason,
            severity=_pipeline_severity(outcome.outcome_type),
            details={
                "outcome_type": outcome.outcome_type.value,
                "failure_stage": outcome.failure_stage,
                "failure_reason": outcome.failure_reason,
                "unknown_terms": unknown_terms,
                "content_terms": content_terms,
            },
            created_at=created_at,
        )
        if not inserted:
            return
        if unknown_terms:
            queued = await self._lexicon.queue_semantic_study(
                event,
                unknown_terms,
                reason="pipeline_understanding_failure",
                priority=0.95,
            )
            for term in unknown_terms:
                await self._record_action(
                    case_id,
                    "semantic_study",
                    term,
                    "queued" if queued else "failed",
                    {"term": term, "queued": queued > 0},
                    created_at,
                )
            await self._set_status(
                case_id,
                "monitoring" if queued else "queued",
                created_at,
            )
            return
        await self._record_action(
            case_id,
            "implementation_review",
            reason,
            "queued",
            {"failure_reason": outcome.failure_reason},
            created_at,
        )

    async def observe_response_trace(self, terminal: ResponseTraceTerminal) -> None:
        """Journal terminal reply failures missed by semantic outcome storage."""

        if terminal.status in {"sent", "no_reply"}:
            return
        if terminal.status == "withheld" and terminal.failure_stage == "context_gate":
            return
        trace = await self._database.fetchone(
            """SELECT guild_id, channel_id, prompt
               FROM language_response_traces WHERE source_event_id = ?""",
            (terminal.event_id,),
        )
        if trace is None:
            return
        existing = await self._database.fetchone(
            """SELECT 1 FROM learning_quality_cases
               WHERE source_kind = 'pipeline' AND prompt_event_id = ? LIMIT 1""",
            (terminal.event_id,),
        )
        if existing is not None:
            return
        context = await self._failure_context(terminal.event_id)
        unknown_terms = _json_tuple(context["unknown_terms_json"] if context else None)
        content_terms = _json_tuple(context["content_terms_json"] if context else None)
        category = "text_understanding" if unknown_terms else "pipeline_failure"
        reason = terminal.failure_reason or terminal.failure_stage or terminal.status
        case_id = f"quality-trace:{terminal.event_id}:{terminal.status}"
        inserted = await self._insert_case(
            case_id=case_id,
            guild_id=int(trace["guild_id"]),
            channel_id=(int(trace["channel_id"]) if trace["channel_id"] is not None else None),
            category=category,
            source_kind="pipeline",
            source_id=f"trace:{terminal.event_id}:{terminal.status}",
            prompt_event_id=terminal.event_id,
            response_message_id=None,
            fingerprint=_fingerprint(category, content_terms or unknown_terms or (reason,)),
            reason=reason,
            severity=0.88 if terminal.status == "failed" else 0.72,
            details={
                "trace_status": terminal.status,
                "selected_route": terminal.selected_route,
                "failure_stage": terminal.failure_stage,
                "failure_reason": terminal.failure_reason,
                "prompt": str(trace["prompt"]),
                "unknown_terms": unknown_terms,
                "content_terms": content_terms,
            },
            created_at=terminal.completed_at,
        )
        if not inserted:
            return
        await self._record_action(
            case_id,
            "implementation_review",
            reason,
            "queued",
            {
                "trace_status": terminal.status,
                "selected_route": terminal.selected_route,
            },
            terminal.completed_at,
        )

    async def _experience_for_feedback(
        self, event: Event, response_message_id: int | None
    ) -> Any | None:
        if response_message_id is not None:
            row = await self._database.fetchone(
                """SELECT * FROM language_utterance_experiences
                   WHERE guild_id = ? AND response_message_id = ?""",
                (event.guild_id, response_message_id),
            )
            if row is not None:
                return row
        if event.channel_id is None or event.actor_id is None:
            return None
        return await self._database.fetchone(
            """SELECT * FROM language_utterance_experiences
               WHERE guild_id = ? AND channel_id = ? AND user_id = ?
                 AND resolved_at = ? AND reward < -0.10
               ORDER BY created_at DESC LIMIT 1""",
            (
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.utc_timestamp.isoformat(),
            ),
        )

    async def _failure_context(self, prompt_event_id: str) -> Any | None:
        return await self._database.fetchone(
            """SELECT comprehension_score, sufficient, content_terms_json,
                      unknown_terms_json
               FROM language_utterance_understandings WHERE event_id = ?""",
            (prompt_event_id,),
        )

    async def _resolve_matching_cases(
        self,
        guild_id: int,
        fingerprints: tuple[str, ...],
        *,
        source_event_id: str,
        resolved_at: datetime,
    ) -> int:
        placeholders = ",".join("?" for _ in fingerprints)
        row = await self._database.fetchone(
            f"""SELECT case_id FROM learning_quality_cases
                WHERE guild_id = ? AND fingerprint IN ({placeholders})
                  AND status IN ('queued', 'monitoring')
                ORDER BY severity DESC, created_at DESC LIMIT 1""",
            (guild_id, *fingerprints),
        )
        if row is None:
            return 0
        case_id = str(row["case_id"])
        await self._database.execute(
            """UPDATE learning_quality_cases
               SET status = 'resolved', updated_at = ?, resolved_at = ?
               WHERE case_id = ? AND status IN ('queued', 'monitoring')""",
            (resolved_at.isoformat(), resolved_at.isoformat(), case_id),
        )
        await self._record_action(
            case_id,
            "positive_repair_verification",
            source_event_id,
            "applied",
            {"feedback_event_id": source_event_id},
            resolved_at,
        )
        return 1

    async def _record_existing_learning(
        self, case_id: str, experience: Any, observed_at: datetime
    ) -> int:
        applied = 0
        experience_id = str(experience["experience_id"])
        reflection = await self._database.fetchone(
            "SELECT 1 FROM cognition_reflections WHERE experience_id = ?",
            (experience_id,),
        )
        if reflection is not None:
            await self._record_action(
                case_id,
                "strategy_reflection",
                experience_id,
                "applied",
                {},
                observed_at,
            )
            applied += 1
        source = await self._database.fetchone(
            """SELECT example.status, example.example_id
               FROM learning_example_sources AS source
               JOIN learning_instruction_examples AS example
                 ON example.example_id = source.example_id
               WHERE source.guild_id = ? AND source.response_message_id = ?""",
            (int(experience["guild_id"]), int(experience["response_message_id"])),
        )
        if source is not None and str(source["status"]) == "rejected":
            await self._record_action(
                case_id,
                "bad_example_quarantined",
                str(source["example_id"]),
                "applied",
                {},
                observed_at,
            )
            applied += 1
        if float(experience["reward"] or 0.0) < 0.0:
            await self._record_action(
                case_id,
                "transition_penalty",
                experience_id,
                "applied",
                {"reward": float(experience["reward"])},
                observed_at,
            )
            applied += 1
        return applied

    async def _insert_case(
        self,
        *,
        case_id: str,
        guild_id: int,
        channel_id: int | None,
        category: str,
        source_kind: str,
        source_id: str,
        prompt_event_id: str | None,
        response_message_id: int | None,
        fingerprint: str,
        reason: str,
        severity: float,
        details: dict[str, object],
        created_at: datetime,
    ) -> bool:
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO learning_quality_cases (
                   case_id, guild_id, channel_id, category, source_kind, source_id,
                   prompt_event_id, response_message_id, fingerprint, reason, severity,
                   details_json, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                guild_id,
                channel_id,
                category,
                source_kind,
                source_id,
                prompt_event_id,
                response_message_id,
                fingerprint,
                reason[:120],
                max(0.0, min(1.0, severity)),
                _json(details),
                created_at.isoformat(),
                created_at.isoformat(),
            ),
        )
        if cursor.rowcount == 1:
            logger.info(
                "quality_case=%s category=%s source=%s severity=%.3f",
                case_id,
                category,
                source_kind,
                severity,
            )
        return cursor.rowcount == 1

    async def _record_action(
        self,
        case_id: str,
        action_type: str,
        target_key: str,
        status: str,
        details: dict[str, object],
        created_at: datetime,
    ) -> None:
        await self._database.execute(
            """INSERT OR IGNORE INTO learning_quality_actions (
                   case_id, action_type, target_key, status, details_json, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (
                case_id,
                action_type,
                target_key[:500],
                status,
                _json(details),
                created_at.isoformat(),
            ),
        )

    async def _set_status(self, case_id: str, status: str, updated_at: datetime) -> None:
        await self._database.execute(
            """UPDATE learning_quality_cases SET status = ?, updated_at = ?
               WHERE case_id = ?""",
            (status, updated_at.isoformat(), case_id),
        )


def _json_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(str(item) for item in decoded if isinstance(item, str))


def _fingerprint(category: str, parts: tuple[str, ...]) -> str:
    normalized = "|".join((category, *(part.casefold() for part in sorted(parts))))
    return sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _pipeline_severity(outcome_type: ProcessingOutcomeType) -> float:
    return {
        ProcessingOutcomeType.INTERNAL_EXCEPTION: 1.0,
        ProcessingOutcomeType.FAILED_TO_FORM_PLAN: 0.90,
        ProcessingOutcomeType.FAILED_TO_REALIZE_LANGUAGE: 0.85,
        ProcessingOutcomeType.VALIDATION_REJECTED: 0.75,
        ProcessingOutcomeType.FAILED_TO_FORM_INTENTION: 0.70,
    }.get(outcome_type, 0.70)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
