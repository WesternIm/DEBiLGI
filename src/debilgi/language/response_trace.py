"""Durable, event-scoped observability for the complete reply pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from debilgi.events.models import Event
from debilgi.storage.database import Database

_logger = logging.getLogger("debilgi.response_trace")
_SENSITIVE_KEY_PARTS = ("authorization", "cookie", "password", "secret", "token")
_MAX_TEXT = 4_000
_MAX_ITEMS = 100
_MAX_DEPTH = 7


def content_diagnostic(
    *, source: str, renderer: str, speech_act: str, metadata: Mapping[str, Any],
    candidate_id: str, text: str = "",
) -> dict[str, Any]:
    """Describe existing content, never create a plan or affect admission/scoring.

    A surface-only ID explicitly records the absence of a prior semantic plan.
    It must not be mistaken for a validated meaning or a new memory claim.
    """
    origin = metadata.get("meaning_origin") or metadata.get("semantic_origin") or source
    plan_key = metadata.get("intention_plan_key") or metadata.get("message_plan_key")
    if source == "interlocutor_clarification":
        content_source = "clarification_recovery"
    elif source == "learned_distribution":
        content_source = "corpus_generation"
    elif source == "learned_dialogue_structure":
        content_source = "social_form"
    elif source == "discord_identity":
        content_source = "runtime_identity"
    elif source == "affective_stance":
        content_source = "affective_stance"
    elif plan_key:
        content_source = "semantic_plan"
    elif source in {"proposition_memory", "knowledge_retrieval", "retrieval"}:
        content_source = "proposition_memory"
    elif source == "dialogue_task":
        content_source = "current_turn_request"
    else:
        content_source = "other"
    act = metadata.get("learned_response_act") or speech_act
    if source == "interlocutor_clarification" or metadata.get("genuine_clarification"):
        category = "clarification"
    elif (metadata.get("communicative_status") == "express_limit"
          or metadata.get("derivation") == "query_unobserved_relation"):
        category = "epistemic"
    elif (metadata.get("reaction_meaning") or metadata.get("observed_form", {}).get("reactive_act")
          or act in {"greeting", "farewell", "thanks", "gratitude", "contact"}
          or (source == "learned_dialogue_structure" and act == "fragment"
              and set(text.casefold().rstrip(".!?").split()) & {"спасибо", "благодарю"})):
        category = "social"
    else:
        category = "informative"
    identity = str(plan_key or candidate_id)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:24]
    premises = metadata.get("premises") or ()
    return {
        "content_source": content_source, "origin": origin, "producer": source,
        "renderer": renderer, "speech_act": act, "category": category,
        "plan_id": f"plan:{digest}" if plan_key else f"surface-only:{digest}",
        "meaning_id": f"meaning:{digest}" if plan_key else f"observed-content:{digest}",
        "plan_basis": "semantic_plan" if plan_key else "surface_only",
        "plan_key": plan_key, "intention_id": metadata.get("meaning_intention_id"),
        "operation": metadata.get("derivation") or metadata.get("semantic_operation"),
        "exchange_pattern_id": metadata.get("exchange_pattern_id"),
        "exchange_clause_kind": metadata.get("exchange_clause_kind"),
        "reaction_meaning": metadata.get("reaction_meaning"),
        "event_valence": metadata.get("event_valence"),
        "condition_report": metadata.get("condition_report"),
        "part_search": metadata.get("part_search"),
        "reverse_check": metadata.get("reverse_check"),
        "response_mixture": metadata.get("response_mixture"),
        "known_slots": metadata.get("known_slots"),
        "hypothesis": metadata.get("hypothesis"),
        "hypothetical_assertions": metadata.get("hypothetical_assertions"),
        "personal_choice": metadata.get("choice_context", {}).get("choice"),
        "surface_choice": metadata.get("surface_choice"),
        "scope_candidates": metadata.get("choice_context", {}).get("scope_candidates", ()),
        "fact_previews": metadata.get("choice_context", {}).get("fact_previews", ()),
        "coalition": metadata.get("coalition"),
        "borrowed_support": metadata.get("borrowed_support"),
        "reconsideration": {k: metadata.get("choice_context", {}).get(k)
                            for k in ("failures", "questions", "reactivation")},
        "reaction_rule": next(({k: p.get(k) for k in (
            "rule_id", "rule_weight", "activation", "input_features", "target_features",
            "activation_source_ids", "evidence_ids",
        )} for p in metadata.get("reaction_parts", ()) if p["component"] == "core"), None),
        "reaction_parts": tuple({"part_id": p["pattern_id"], "component": p["component"],
                                 "meaning": p["meaning"], "evidence_ids": p["evidence_ids"]}
                                for p in metadata.get("reaction_parts", ())),
        "evidence_kinds": tuple(dict.fromkeys(getattr(p, "kind", "unknown") for p in premises)),
        "premise_references": tuple({
            "evidence_id": getattr(p, "evidence_id", None),
            "kind": getattr(p, "kind", "unknown"), "reference": getattr(p, "text", None),
        } for p in premises),
        "evidence_ids": tuple(dict.fromkeys((
            *metadata.get("reasoning_evidence", ()),
            *metadata.get("corpus_source_event_ids", ()),
        ))),
    }


def reason_kind(code: str, stage: str = "") -> str:
    """Diagnostic labels only; preserve the original code and its stage as evidence."""
    if stage in {"send", "delivery"}:
        return "delivery_failure"
    if any(part in code for part in (
        "parse_failure", "parse_failed", "unparseable", "unsupported_query_construction",
    )):
        return "parse_failure"
    if any(part in code for part in (
        "ungrounded", "not_grounded", "missing_fact", "no_evidence", "query_has_no_supported_fact",
    )):
        return "missing_fact"
    if any(part in code for part in ("ambiguous", "ambiguity", "intent_uncertain")):
        return "ambiguity"
    if any(part in code for part in (
        "surface", "mismatch", "changed_", "missing_subject", "missing_predicate",
        "missing_object", "agreement", "case_government", "unrealizable_selected_plan",
    )):
        return "realization_mismatch"
    if ("timeout" in code or "exception" in code
            or code in {"query_not_executed", "query_execution_failed"}):
        return "upstream_failure"
    if any(part in code for part in ("disabled", "budget", "automatic_definition")):
        return "policy"
    return "selection_failure"


@dataclass(frozen=True, slots=True)
class ResponseTraceTerminal:
    """The terminal facts needed by diagnostics without coupling to Discord."""

    event_id: str
    status: str
    selected_route: str | None
    failure_stage: str | None
    failure_reason: str | None
    completed_at: datetime


class ResponseTraceRepository:
    """Record what the reply system observed, produced, rejected and sent.

    Trace persistence is deliberately best-effort. Observability must never
    replace an otherwise valid reply with a terminal failure.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._completion_notifier: Callable[[], None] | None = None
        self._quality_observer: (
            Callable[[ResponseTraceTerminal], Awaitable[None]] | None
        ) = None

    def set_completion_notifier(self, notifier: Callable[[], None] | None) -> None:
        self._completion_notifier = notifier

    def set_quality_observer(
        self,
        observer: Callable[[ResponseTraceTerminal], Awaitable[None]] | None,
    ) -> None:
        self._quality_observer = observer

    async def begin(self, event: Event, *, prompt: str) -> str:
        trace_id = _trace_id(event.event_id)
        payload = {
            "event_id": event.event_id,
            "trace_id": trace_id,
            "stage": "received",
            "component": "discord_event",
            "status": "started",
            "summary": "reply trace started",
            "details": {
                "guild_id": event.guild_id,
                "channel_id": event.channel_id,
                "user_id": event.actor_id,
                "prompt": prompt,
                "event_type": event.event_type.value,
            },
        }
        try:
            cursor = await self._database.execute(
                """INSERT OR IGNORE INTO language_response_traces (
                       trace_id, source_event_id, guild_id, channel_id, user_id,
                       prompt, status, started_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'started', ?)""",
                (
                    trace_id,
                    event.event_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    prompt,
                    event.utc_timestamp.isoformat(),
                ),
            )
            if cursor.rowcount == 1:
                _write_log(payload)
        except Exception:
            _write_log(payload)
            _logger.exception(
                "trace_persistence_failed event=%s stage=received",
                event.event_id,
            )
        return trace_id

    async def step(
        self,
        event_id: str,
        *,
        stage: str,
        component: str,
        status: str,
        summary: str,
        details: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        safe_details = _safe_value(details or {})
        payload = {
            "event_id": event_id,
            "trace_id": _trace_id(event_id),
            "stage": stage,
            "component": component,
            "status": status,
            "summary": summary,
            "details": safe_details,
        }
        _write_log(payload)
        try:
            async with self._database.transaction() as connection:
                row = await (
                    await connection.execute(
                        "SELECT trace_id FROM language_response_traces WHERE source_event_id = ?",
                        (event_id,),
                    )
                ).fetchone()
                if row is None:
                    return
                trace_id = str(row["trace_id"])
                sequence_row = await (
                    await connection.execute(
                        """SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
                           FROM language_response_trace_steps
                           WHERE trace_id = ?""",
                        (trace_id,),
                    )
                ).fetchone()
                sequence = int(sequence_row["next_sequence"])
                await connection.execute(
                    """INSERT INTO language_response_trace_steps (
                           trace_id, sequence, stage, component, status,
                           summary, details_json, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trace_id,
                        sequence,
                        stage,
                        component,
                        status,
                        summary,
                        json.dumps(
                            safe_details,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        (created_at or datetime.now(UTC)).isoformat(),
                    ),
                )
        except Exception:
            _logger.exception(
                "trace_persistence_failed event=%s stage=%s",
                event_id,
                stage,
            )

    async def finish(
        self,
        event_id: str,
        *,
        status: str,
        selected_route: str | None = None,
        final_text: str | None = None,
        response_message_id: int | None = None,
        arbitration_decision_id: str | None = None,
        failure_stage: str | None = None,
        failure_reason: str | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        timestamp = completed_at or datetime.now(UTC)
        diagnostic = None
        try:
            diagnostic = await self.diagnostics_for_event(
                event_id, status=status, selected_route=selected_route, final_text=final_text,
                failure_stage=failure_stage, failure_reason=failure_reason,
            )
        except Exception:
            _logger.exception("trace_diagnostic_failed event=%s", event_id)
        await self.step(
            event_id,
            stage="completed",
            component="response_pipeline",
            status=status,
            summary=(
                "reply sent"
                if status == "sent"
                else "reply pipeline completed without a sent response"
            ),
            details={
                "selected_route": selected_route,
                "final_text": final_text,
                "response_message_id": response_message_id,
                "arbitration_decision_id": arbitration_decision_id,
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "diagnostic": diagnostic,
            },
            created_at=timestamp,
        )
        try:
            await self._database.execute(
                """UPDATE language_response_traces
                   SET status = ?, selected_route = ?, final_text = ?,
                       response_message_id = ?, arbitration_decision_id = ?,
                       failure_stage = ?, failure_reason = ?, completed_at = ?
                   WHERE source_event_id = ?""",
                (
                    status,
                    selected_route,
                    final_text,
                    response_message_id,
                    arbitration_decision_id,
                    failure_stage,
                    failure_reason,
                    timestamp.isoformat(),
                    event_id,
                ),
            )
        except Exception:
            _logger.exception(
                "trace_persistence_failed event=%s stage=completed",
                event_id,
            )
        finally:
            if self._completion_notifier is not None:
                self._completion_notifier()
            if self._quality_observer is not None:
                terminal = ResponseTraceTerminal(
                    event_id=event_id,
                    status=status,
                    selected_route=selected_route,
                    failure_stage=failure_stage,
                    failure_reason=failure_reason,
                    completed_at=timestamp,
                )
                try:
                    await self._quality_observer(terminal)
                except Exception:
                    _logger.exception(
                        "trace_quality_observer_failed event=%s status=%s",
                        event_id,
                        status,
                    )

    async def steps_for_event(self, event_id: str) -> tuple[dict[str, Any], ...]:
        rows = await self._database.fetchall(
            """SELECT step.sequence, step.stage, step.component, step.status,
                      step.summary, step.details_json, step.created_at
               FROM language_response_trace_steps AS step
               JOIN language_response_traces AS trace
                 ON trace.trace_id = step.trace_id
               WHERE trace.source_event_id = ?
               ORDER BY step.sequence""",
            (event_id,),
        )
        return tuple(
            {
                "sequence": int(row["sequence"]),
                "stage": str(row["stage"]),
                "component": str(row["component"]),
                "status": str(row["status"]),
                "summary": str(row["summary"]),
                "details": json.loads(str(row["details_json"])),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        )

    async def diagnostics_for_event(
        self, event_id: str, *, status: str | None = None, selected_route: str | None = None,
        final_text: str | None = None, failure_stage: str | None = None,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        """Read-only projection over existing traces, not a second reply policy.

        Categories describe acts, not usefulness. `ambiguity` may be merely the
        legacy recovery label; its code remains visible, not certified as a parse.
        Feedback is an observed consequence, separately from preparation/delivery.
        """
        steps = await self.steps_for_event(event_id)
        row = await self._database.fetchone(
            "SELECT * FROM language_response_traces WHERE source_event_id = ?", (event_id,),
        )
        root = dict(row) if row else {}
        route = next((s["details"] for s in reversed(steps) if s["stage"] == "route"), {})
        status = status or root.get("status") or "started"
        selected_route = (
            selected_route or root.get("selected_route") or route.get("response_source")
        )
        final_text = final_text or root.get("final_text") or route.get("text")
        failure_stage = failure_stage or root.get("failure_stage")
        failure_reason = failure_reason or root.get("failure_reason")
        delivery = next((s for s in reversed(steps) if s["stage"] == "delivery"), None)
        if delivery and delivery["status"] in {"failed", "uncertain"} and status != "sent":
            status, failure_stage = delivery["status"], "delivery"
            failure_reason = delivery["details"].get("reason")
        selections = [item for item in (
            s["details"] if s["stage"] == "selection" else s["details"].get("selection", {})
            for s in steps
            if s["stage"] == "selection" or "selection" in s["details"]
        ) if isinstance(item, dict)]
        selected = next((
            item for item in reversed(selections)
            if item.get("winner_text") == final_text and item.get("diagnostic")
            and (not selected_route or selected_route.removeprefix("arbitrated:")
                 == item.get("winner_source"))
        ), {})
        content = selected.get("diagnostic")
        if content is None and final_text:
            content = content_diagnostic(
                source=(selected_route or "other").removeprefix("arbitrated:"),
                renderer="route_without_candidate", speech_act="unknown", metadata={},
                candidate_id=f"{event_id}:{selected_route}",
            )
        category = (
            "error" if status in {"failed", "uncertain", "error"} else
            "withheld" if status in {"withheld", "no_reply"} else
            content["category"] if content else "withheld"
        )
        reasons: dict[str, list[dict[str, Any]]] = {
            key: [] for key in (
                "parse_failure", "missing_fact", "ambiguity", "realization_mismatch",
                "delivery_failure", "policy", "upstream_failure", "selection_failure",
            )
        }

        def add_reason(code, stage):
            if not isinstance(code, str) or not code:
                return
            bucket = reasons[reason_kind(code, stage)]
            item = {"code": code, "stage": stage}
            if item not in bucket:
                bucket.append(item)

        for step in steps:
            if step["component"] == "response_mode":
                decision = step["details"].get("decision", {})
                if decision.get("mode") in {"silence_or_unknown", "clarification"}:
                    add_reason(decision.get("reason"), "response_mode")
            if step["status"] in {"withheld", "failed", "clarify", "uncertain"}:
                add_reason(step["details"].get("reason"), step["stage"])
            parsed = step["details"].get("input_parse")
            if parsed and not parsed["has_tokens"]:
                add_reason("parse_failure:no_tokens", "input_parse")
            if parsed and parsed["ambiguities"]:
                add_reason("ambiguity:parser_alternatives", "input_parse")
        for selection in selections:
            for score in selection.get("scores", ()):
                for code in score.get("rejection_reasons", ()):
                    add_reason(code, "candidate_rejection")
        if category in {"withheld", "error"}:
            failure_reason = failure_reason or "no_admissible_candidate"
            add_reason(failure_reason, failure_stage or "preparation")
        gate = next((s["status"] for s in reversed(steps) if s["stage"] == "context_gate"), None)
        return {
            "category": category, "content": content,
            "plan_id": content["plan_id"] if content else f"no-content:{event_id}",
            "meaning_id": content["meaning_id"] if content else f"meaning-turn:{event_id}",
            "content_source": content["content_source"] if content else "none",
            "decision_id": selected.get("decision_id"),
            "withhold_reason": (
                {"kind": reason_kind(failure_reason, failure_stage or "preparation"),
                 "code": failure_reason, "stage": failure_stage or "preparation"}
                if category in {"withheld", "error"} else None
            ),
            "reasons": reasons,
            "selection_path": [
                {"stage": s["stage"], "component": s["component"], "status": s["status"],
                 "selection_reason": s["details"].get("selection_reason")}
                for s in steps if s["stage"] != "completed"
            ],
            "preparation": "prepared" if route else "no_surface",
            "gate": gate, "receipt": any(s["stage"] == "send" and s["status"] == "sent"
                                         for s in steps),
            "feedback": [s["details"] for s in steps if s["stage"] == "feedback"],
        }

    async def trace_for_guild(
        self,
        guild_id: int,
        *,
        event_id: str | None = None,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]] | None:
        """Return one trace visible to a guild administrator.

        The guild predicate is intentionally part of both lookup paths so an
        event id copied from another server cannot expose its trace.
        """

        if event_id is None:
            row = await self._database.fetchone(
                """SELECT *
                   FROM language_response_traces
                   WHERE guild_id = ?
                   ORDER BY started_at DESC
                   LIMIT 1""",
                (guild_id,),
            )
        else:
            row = await self._database.fetchone(
                """SELECT *
                   FROM language_response_traces
                   WHERE guild_id = ? AND source_event_id = ?
                   LIMIT 1""",
                (guild_id, event_id),
            )
        if row is None:
            return None
        root = dict(row)
        return root, await self.steps_for_event(str(row["source_event_id"]))


def _trace_id(event_id: str) -> str:
    return f"response-trace:{event_id}"


def _write_log(payload: Mapping[str, Any]) -> None:
    _logger.info(
        "%s",
        json.dumps(
            _safe_value(payload),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )


def _safe_value(value: Any, *, depth: int = 0, key: str = "") -> Any:
    if any(part in key.casefold() for part in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if depth >= _MAX_DEPTH:
        return "[max-depth]"
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, Enum):
        return _safe_value(value.value, depth=depth + 1, key=key)
    if is_dataclass(value) and not isinstance(value, type):
        return _safe_value(asdict(value), depth=depth + 1, key=key)
    if isinstance(value, str):
        return value if len(value) <= _MAX_TEXT else f"{value[:_MAX_TEXT]}…"
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_value(
                item_value,
                depth=depth + 1,
                key=str(item_key),
            )
            for item_key, item_value in list(value.items())[:_MAX_ITEMS]
        }
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [
            _safe_value(item, depth=depth + 1, key=key)
            for item in list(value)[:_MAX_ITEMS]
        ]
    return _safe_value(str(value), depth=depth + 1, key=key)
