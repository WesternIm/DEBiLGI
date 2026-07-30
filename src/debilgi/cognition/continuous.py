"""Persistent local cognition that keeps working between Discord events."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from debilgi.events.models import Event, EventType
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.continuous_cognition")

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


@dataclass(frozen=True, slots=True)
class ContinuousCognitionConfig:
    active_interval: timedelta = timedelta(seconds=2)
    idle_interval: timedelta = timedelta(seconds=15)
    work_per_pulse: int = 8
    trace_retry_limit: int = 20

    def __post_init__(self) -> None:
        if self.active_interval <= timedelta(0) or self.idle_interval <= timedelta(0):
            raise ValueError("continuous cognition intervals must be positive")
        if self.work_per_pulse <= 0 or self.trace_retry_limit <= 0:
            raise ValueError("continuous cognition limits must be positive")


@dataclass(frozen=True, slots=True)
class CognitivePulseStats:
    pulse_id: str
    work_claimed: int = 0
    hypotheses_reviewed: int = 0
    gaps_recorded: int = 0
    signals_recorded: int = 0


class ContinuousCognitionService:
    """Keep a bounded cognitive actor alive for the lifetime of the bot.

    The actor never authors or sends Discord messages. It turns completed
    semantic inquiry traces into revisable hypotheses, preserves unresolved
    questions as durable gaps and exposes relevant retained knowledge to later
    reply planning.
    """

    def __init__(
        self,
        database: Database,
        *,
        enabled: bool = True,
        config: ContinuousCognitionConfig | None = None,
        clock: Any | None = None,
    ) -> None:
        self._database = database
        self.enabled = enabled
        self.config = config or ContinuousCognitionConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._task: asyncio.Task[None] | None = None
        self._pulse_lock = asyncio.Lock()
        self._wake = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if not self.enabled or self.running:
            return
        now = self._now()
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'queued', next_attempt_at = ?, updated_at = ?,
                   last_error = 'recovered_after_restart'
               WHERE status = 'processing'""",
            (now.isoformat(), now.isoformat()),
        )
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
               SET status = 'running', started_at = ?, last_heartbeat_at = ?,
                   stopped_at = NULL, last_error = NULL
               WHERE runtime_id = 1""",
            (now.isoformat(), now.isoformat()),
        )
        self._task = asyncio.create_task(
            self._run_loop(),
            name="debilgi-continuous-cognition",
        )
        logger.info(
            "runtime=started active_interval=%.3f idle_interval=%.3f work_per_pulse=%d",
            self.config.active_interval.total_seconds(),
            self.config.idle_interval.total_seconds(),
            self.config.work_per_pulse,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        now = self._now()
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
               SET status = 'stopped', stopped_at = ?, last_heartbeat_at = ?
               WHERE runtime_id = 1""",
            (now.isoformat(), now.isoformat()),
        )
        logger.info("runtime=stopped")

    async def observe(self, event: Event) -> bool:
        """Queue human language for continued processing after the live turn."""

        if (
            not self.enabled
            or event.event_type not in _MESSAGE_TYPES
            or event.payload.get("author_is_bot") is True
        ):
            return False
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return False
        now = self._now()
        addressed = event.payload.get("bot_mentioned") is True
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_work_items (
                   work_id, source_event_id, guild_id, channel_id, priority,
                   next_attempt_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"continuous-work:{event.event_id}",
                event.event_id,
                event.guild_id,
                event.channel_id,
                0.90 if addressed else 0.55,
                now.isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        self._wake.set()
        return cursor.rowcount == 1

    def wake(self) -> None:
        """Notify the actor that a response trace or another evidence source changed."""

        self._wake.set()

    async def pulse_once(self, now: datetime | None = None) -> CognitivePulseStats:
        async with self._pulse_lock:
            started = self._aware(now or self._now())
            pulse_id = f"continuous-pulse:{uuid4()}"
            await self._database.execute(
                """INSERT INTO continuous_cognition_pulses (
                       pulse_id, started_at, status
                   ) VALUES (?, ?, 'running')""",
                (pulse_id, started.isoformat()),
            )
            claimed = hypotheses = gaps = signals = 0
            try:
                rows = await self._database.fetchall(
                    """SELECT work_id, source_event_id, guild_id, attempts
                       FROM continuous_cognition_work_items
                       WHERE status = 'queued' AND next_attempt_at <= ?
                       ORDER BY priority DESC, created_at
                       LIMIT ?""",
                    (started.isoformat(), self.config.work_per_pulse),
                )
                for row in rows:
                    if not await self._claim(str(row["work_id"]), started):
                        continue
                    claimed += 1
                    try:
                        result = await self._process_work_item(row, started)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        await self._fail_work(row, started, exc)
                        logger.exception(
                            "work=%s event=%s status=failed",
                            row["work_id"],
                            row["source_event_id"],
                        )
                        continue
                    hypotheses += result[0]
                    gaps += result[1]
                    signals += result[2]
                stats = CognitivePulseStats(
                    pulse_id,
                    claimed,
                    hypotheses,
                    gaps,
                    signals,
                )
                await self._finish_pulse(stats, started, status="completed")
                if claimed or hypotheses or gaps or signals:
                    logger.info(
                        "pulse=%s status=completed work=%d hypotheses=%d gaps=%d signals=%d",
                        pulse_id,
                        claimed,
                        hypotheses,
                        gaps,
                        signals,
                    )
                return stats
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._database.execute(
                    """UPDATE continuous_cognition_pulses
                       SET finished_at = ?, status = 'failed', error_type = ?,
                           error_message = ? WHERE pulse_id = ?""",
                    (
                        self._now().isoformat(),
                        type(exc).__name__,
                        str(exc)[:1000],
                        pulse_id,
                    ),
                )
                await self._database.execute(
                    """UPDATE continuous_cognition_runtime
                       SET status = 'degraded', last_heartbeat_at = ?,
                           total_pulses = total_pulses + 1, last_error = ?
                       WHERE runtime_id = 1""",
                    (
                        self._now().isoformat(),
                        f"{type(exc).__name__}: {exc}"[:1000],
                    ),
                )
                logger.exception("pulse=%s status=failed", pulse_id)
                raise

    async def relevant_facts(
        self,
        guild_id: int,
        terms: Iterable[str],
        *,
        limit: int = 8,
    ) -> tuple[str, ...]:
        """Recall working conclusions without promoting hypotheses to facts."""

        normalized = {
            _normalize_term(term)
            for term in terms
            if isinstance(term, str) and _normalize_term(term)
        }
        if not normalized:
            return ()
        rows = await self._database.fetchall(
            """SELECT subject_key, predicate_key, object_key, epistemic_status,
                      confidence, status
               FROM continuous_cognition_hypotheses
               WHERE guild_id = ? AND status != 'rejected'
               ORDER BY
                   CASE status WHEN 'retained' THEN 0 ELSE 1 END,
                   confidence DESC, last_reviewed_at DESC
               LIMIT 100""",
            (guild_id,),
        )
        ranked: list[tuple[int, float, str]] = []
        for row in rows:
            values = tuple(
                _normalize_term(str(row[key]))
                for key in ("subject_key", "predicate_key", "object_key")
                if row[key] is not None
            )
            overlap = len(normalized & set(values))
            if overlap == 0:
                continue
            subject = str(row["subject_key"] or "").strip()
            predicate = str(row["predicate_key"]).strip()
            object_key = str(row["object_key"] or "").strip()
            statement = " ".join(part for part in (subject, predicate, object_key) if part)
            marker = str(row["epistemic_status"])
            ranked.append(
                (
                    overlap,
                    float(row["confidence"]),
                    f"{statement} [{marker}; continuous_cognition]",
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple(item[2] for item in ranked[: max(1, min(limit, 24))])

    async def _run_loop(self) -> None:
        while True:
            self._wake.clear()
            try:
                stats = await self.pulse_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                stats = None
            if stats is not None and stats.work_claimed >= self.config.work_per_pulse:
                continue
            delay = (
                self.config.active_interval
                if stats is not None and stats.work_claimed
                else self.config.idle_interval
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=delay.total_seconds(),
                )

    async def _claim(self, work_id: str, now: datetime) -> bool:
        cursor = await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'processing', attempts = attempts + 1, updated_at = ?
               WHERE work_id = ? AND status = 'queued'""",
            (now.isoformat(), work_id),
        )
        return cursor.rowcount == 1

    async def _process_work_item(
        self,
        row: Mapping[str, Any],
        now: datetime,
    ) -> tuple[int, int, int]:
        work_id = str(row["work_id"])
        event_id = str(row["source_event_id"])
        guild_id = int(row["guild_id"])
        attempts = int(row["attempts"]) + 1
        trace = await self._database.fetchone(
            """SELECT status, selected_route, failure_stage, failure_reason
               FROM language_response_traces
               WHERE source_event_id = ?""",
            (event_id,),
        )
        if trace is None or str(trace["status"]) == "started":
            if attempts < self.config.trace_retry_limit:
                await self._defer(work_id, now, attempts, "response_trace_not_complete")
                return 0, 0, 0
            await self._complete(
                work_id,
                now,
                "trace_unavailable_after_retry_budget",
            )
            return 0, 0, 0

        step = await self._database.fetchone(
            """SELECT details_json
               FROM language_response_trace_steps AS step
               JOIN language_response_traces AS trace
                 ON trace.trace_id = step.trace_id
               WHERE trace.source_event_id = ? AND step.stage = 'inquiry'
               ORDER BY step.sequence DESC LIMIT 1""",
            (event_id,),
        )
        hypotheses = gaps = 0
        if step is not None:
            details = _json_mapping(step["details_json"])
            hypotheses = await self._review_conclusions(
                guild_id,
                event_id,
                details.get("conclusions"),
                now,
            )
            gaps = await self._record_gaps(
                guild_id,
                event_id,
                details.get("steps"),
                now,
            )
        signals = await self._record_signal(guild_id, event_id, trace, now)
        summary = (
            f"trace={trace['status']}; hypotheses={hypotheses}; "
            f"gaps={gaps}; signals={signals}"
        )
        await self._complete(work_id, now, summary)
        return hypotheses, gaps, signals

    async def _review_conclusions(
        self,
        guild_id: int,
        event_id: str,
        raw_conclusions: object,
        now: datetime,
    ) -> int:
        if not isinstance(raw_conclusions, list):
            return 0
        reviewed = 0
        for raw in raw_conclusions[:12]:
            if not isinstance(raw, dict):
                continue
            predicate = _optional_text(raw.get("predicate"))
            epistemic_status = _optional_text(raw.get("epistemic_status"))
            roles = _roles(raw.get("roles"))
            if predicate is None or epistemic_status not in {"inference", "hypothesis"}:
                continue
            subject = next((value for role, value in roles if role == "subject"), None)
            object_key = next(
                (value for role, value in roles if role != "subject"),
                None,
            )
            confidence = _unit_float(raw.get("confidence"))
            signature = _signature(predicate, roles)
            hypothesis_id = f"continuous-hypothesis:{guild_id}:{signature}"
            evidence_id = f"continuous-evidence:{event_id}:{signature}:supports"
            async with self._database.transaction() as connection:
                existing = await (
                    await connection.execute(
                        """SELECT confidence, positive_evidence, negative_evidence
                           FROM continuous_cognition_hypotheses
                           WHERE guild_id = ? AND signature = ?""",
                        (guild_id, signature),
                    )
                ).fetchone()
                if existing is None:
                    await connection.execute(
                        """INSERT INTO continuous_cognition_hypotheses (
                               hypothesis_id, guild_id, signature, subject_key,
                               predicate_key, object_key, roles_json,
                               epistemic_status, confidence, positive_evidence,
                               status, first_seen_at, last_reviewed_at
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'working', ?, ?)""",
                        (
                            hypothesis_id,
                            guild_id,
                            signature,
                            subject,
                            predicate,
                            object_key,
                            json.dumps(roles, ensure_ascii=False, separators=(",", ":")),
                            epistemic_status,
                            confidence,
                            now.isoformat(),
                            now.isoformat(),
                        ),
                    )
                inserted = await connection.execute(
                    """INSERT OR IGNORE INTO continuous_cognition_evidence (
                           evidence_id, hypothesis_id, source_event_id, relation,
                           confidence, details_json, observed_at
                       ) VALUES (?, ?, ?, 'supports', ?, ?, ?)""",
                    (
                        evidence_id,
                        hypothesis_id,
                        event_id,
                        confidence,
                        json.dumps(raw, ensure_ascii=False, separators=(",", ":")),
                        now.isoformat(),
                    ),
                )
                if inserted.rowcount == 0:
                    continue
                if existing is None:
                    positive = 1
                    combined_confidence = confidence
                    status = _hypothesis_status(
                        epistemic_status,
                        combined_confidence,
                        positive,
                        0,
                    )
                    await connection.execute(
                        """UPDATE continuous_cognition_hypotheses
                           SET confidence = ?, positive_evidence = ?, status = ?,
                               last_reviewed_at = ?
                           WHERE hypothesis_id = ?""",
                        (
                            combined_confidence,
                            positive,
                            status,
                            now.isoformat(),
                            hypothesis_id,
                        ),
                    )
                else:
                    positive = int(existing["positive_evidence"]) + 1
                    negative = int(existing["negative_evidence"])
                    old_confidence = float(existing["confidence"])
                    combined_confidence = min(
                        0.96,
                        old_confidence + (confidence - old_confidence) / positive + 0.015,
                    )
                    status = _hypothesis_status(
                        epistemic_status,
                        combined_confidence,
                        positive,
                        negative,
                    )
                    await connection.execute(
                        """UPDATE continuous_cognition_hypotheses
                           SET confidence = ?, positive_evidence = ?, status = ?,
                               last_reviewed_at = ?
                           WHERE hypothesis_id = ?""",
                        (
                            combined_confidence,
                            positive,
                            status,
                            now.isoformat(),
                            hypothesis_id,
                        ),
                    )
            reviewed += 1
        return reviewed

    async def _record_gaps(
        self,
        guild_id: int,
        event_id: str,
        raw_steps: object,
        now: datetime,
    ) -> int:
        if not isinstance(raw_steps, list):
            return 0
        recorded = 0
        for raw in raw_steps[:24]:
            if not isinstance(raw, dict) or raw.get("status") != "unresolved":
                continue
            question_kind = _optional_text(raw.get("question_kind"))
            answer_role = _optional_text(raw.get("answer_role"))
            if question_kind is None or answer_role is None:
                continue
            subject = _optional_text(raw.get("subject"))
            predicate = _optional_text(raw.get("predicate"))
            signature = _signature(
                question_kind,
                (
                    ("subject", subject or ""),
                    ("predicate", predicate or ""),
                    ("answer_role", answer_role),
                ),
            )
            cursor = await self._database.execute(
                """INSERT INTO continuous_cognition_gaps (
                       gap_id, guild_id, signature, question_kind, subject_key,
                       predicate_key, answer_role, source_event_id,
                       first_seen_at, last_seen_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id, signature) DO UPDATE SET
                       occurrence_count = occurrence_count + 1,
                       source_event_id = excluded.source_event_id,
                       last_seen_at = excluded.last_seen_at,
                       status = CASE
                           WHEN status = 'abandoned' THEN 'open' ELSE status END""",
                (
                    f"continuous-gap:{guild_id}:{signature}",
                    guild_id,
                    signature,
                    question_kind,
                    subject,
                    predicate,
                    answer_role,
                    event_id,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            recorded += int(cursor.rowcount > 0)
        return recorded

    async def _record_signal(
        self,
        guild_id: int,
        event_id: str,
        trace: Mapping[str, Any],
        now: datetime,
    ) -> int:
        status = str(trace["status"])
        kind = {
            "sent": "reply_sent",
            "failed": "reply_failed",
            "withheld": "reply_withheld",
            "no_reply": "initiative_not_selected",
        }.get(status, f"reply_{status}")
        details = {
            "status": status,
            "failure_stage": trace["failure_stage"],
            "failure_reason": trace["failure_reason"],
        }
        cursor = await self._database.execute(
            """INSERT OR IGNORE INTO continuous_cognition_signals (
                   signal_id, source_event_id, guild_id, signal_kind, route,
                   details_json, observed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                f"continuous-signal:{event_id}",
                event_id,
                guild_id,
                kind,
                trace["selected_route"],
                json.dumps(details, ensure_ascii=False, separators=(",", ":")),
                now.isoformat(),
            ),
        )
        return int(cursor.rowcount == 1)

    async def _defer(
        self,
        work_id: str,
        now: datetime,
        attempts: int,
        reason: str,
    ) -> None:
        retry_at = now + self.config.active_interval
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'queued', next_attempt_at = ?, updated_at = ?,
                   result_summary = ?
               WHERE work_id = ?""",
            (retry_at.isoformat(), now.isoformat(), reason, work_id),
        )
    async def _complete(self, work_id: str, now: datetime, summary: str) -> None:
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = 'completed', result_summary = ?, updated_at = ?,
                   completed_at = ?, last_error = NULL
               WHERE work_id = ?""",
            (summary, now.isoformat(), now.isoformat(), work_id),
        )

    async def _fail_work(
        self,
        row: Mapping[str, Any],
        now: datetime,
        exc: Exception,
    ) -> None:
        attempts = int(row["attempts"]) + 1
        retry = attempts < self.config.trace_retry_limit
        await self._database.execute(
            """UPDATE continuous_cognition_work_items
               SET status = ?, next_attempt_at = ?, updated_at = ?, last_error = ?
               WHERE work_id = ?""",
            (
                "queued" if retry else "failed",
                (now + self.config.active_interval).isoformat(),
                now.isoformat(),
                f"{type(exc).__name__}: {exc}"[:1000],
                str(row["work_id"]),
            ),
        )

    async def _finish_pulse(
        self,
        stats: CognitivePulseStats,
        started: datetime,
        *,
        status: str,
    ) -> None:
        await self._database.execute(
            """UPDATE continuous_cognition_pulses
               SET finished_at = ?, status = ?, work_claimed = ?,
                   hypotheses_reviewed = ?, gaps_recorded = ?,
                   signals_recorded = ?
               WHERE pulse_id = ?""",
            (
                max(started, self._now()).isoformat(),
                status,
                stats.work_claimed,
                stats.hypotheses_reviewed,
                stats.gaps_recorded,
                stats.signals_recorded,
                stats.pulse_id,
            ),
        )
        await self._database.execute(
            """UPDATE continuous_cognition_runtime
                   SET status = 'running', last_heartbeat_at = ?,
                   total_pulses = total_pulses + 1,
                   total_work_processed = total_work_processed + ?,
                   last_error = NULL
               WHERE runtime_id = 1""",
            (
                max(started, self._now()).isoformat(),
                stats.work_claimed,
            ),
        )
        if stats.work_claimed == 0:
            await self._database.execute(
                "DELETE FROM continuous_cognition_pulses WHERE pulse_id = ?",
                (stats.pulse_id,),
            )

    def _now(self) -> datetime:
        return self._aware(self._clock())

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("continuous cognition clock must be timezone-aware")
        return value.astimezone(UTC)


def _roles(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        (str(pair[0]), str(pair[1]))
        for pair in value
        if isinstance(pair, list)
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
    )


def _signature(predicate: str, roles: Iterable[tuple[str, str]]) -> str:
    body = json.dumps(
        (predicate, tuple(roles)),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:24]


def _hypothesis_status(
    epistemic_status: str,
    confidence: float,
    positive: int,
    negative: int,
) -> str:
    if negative >= max(2, positive):
        return "rejected"
    required = 2 if epistemic_status == "inference" else 3
    return "retained" if positive >= required and confidence >= 0.66 else "working"


def _json_mapping(value: object) -> Mapping[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _normalize_term(value: str) -> str:
    return value.casefold().replace("ё", "е").strip(".,!?;:—–-()[]{}«»… ")


def _unit_float(value: object) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(1.0, float(value)))
    return 0.0
