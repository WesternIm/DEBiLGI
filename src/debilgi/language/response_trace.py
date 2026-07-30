"""Durable, event-scoped observability for the complete reply pipeline."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
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


class ResponseTraceRepository:
    """Record what the reply system observed, produced, rejected and sent.

    Trace persistence is deliberately best-effort. Observability must never
    replace an otherwise valid reply with a terminal failure.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._completion_notifier: Callable[[], None] | None = None

    def set_completion_notifier(self, notifier: Callable[[], None] | None) -> None:
        self._completion_notifier = notifier

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
