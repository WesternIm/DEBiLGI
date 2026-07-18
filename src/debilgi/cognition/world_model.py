"""Persistent expectations and learned categorical pattern weights."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from debilgi.cognition.difference import categorical_difference
from debilgi.cognition.expectations import (
    OUTCOMES,
    Expectation,
    ExpectationStatus,
    OutcomeType,
    PatternEvidence,
    Prediction,
    Situation,
)
from debilgi.events.models import Event
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class WorldModelStats:
    pending_expectations: int
    resolved_expectations: int
    learned_patterns: int
    average_difference: float | None


class WorldModelRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def evidence(self, guild_id: int, limit: int = 5000) -> list[PatternEvidence]:
        rows = await self._database.fetchall(
            """
            SELECT situation_json, outcome_type, observations
            FROM cognition_patterns
            WHERE guild_id = ? AND observations > 0
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (guild_id, max(1, min(limit, 20_000))),
        )
        return [
            PatternEvidence(
                situation=Situation.from_json(str(row["situation_json"])),
                outcome=OutcomeType(str(row["outcome_type"])),
                observations=int(row["observations"]),
            )
            for row in rows
        ]

    async def create_expectation(
        self,
        event: Event,
        situation: Situation,
        prediction: Prediction,
        horizon: timedelta,
    ) -> Expectation:
        if event.channel_id is None:
            raise ValueError("expectations require a channel")
        created_at = event.utc_timestamp
        expectation_id = f"expectation:{uuid4()}"
        await self._database.execute(
            """
            INSERT OR IGNORE INTO cognition_expectations (
                expectation_id, guild_id, channel_id, actor_id, trigger_event_id,
                trigger_message_id, situation_json, distribution_json,
                predicted_outcome, predicted_probability, evidence_support,
                created_at, expires_at, status, source, observation_window_seconds,
                allowed_outcomes_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                expectation_id,
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.event_id,
                self._payload_int(event, "message_id"),
                situation.to_json(),
                self._distribution_json(prediction.distribution),
                prediction.outcome.value,
                prediction.probability,
                prediction.evidence_support,
                created_at.isoformat(),
                (created_at + horizon).isoformat(),
                "message_observation",
                horizon.total_seconds(),
                json.dumps([outcome.value for outcome in OUTCOMES]),
            ),
        )
        expectation = await self.by_trigger_event(event.event_id)
        if expectation is None:
            raise RuntimeError("failed to create or restore expectation")
        return expectation

    async def by_trigger_event(self, event_id: str) -> Expectation | None:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_expectations WHERE trigger_event_id = ?", (event_id,)
        )
        return self._from_row(row) if row else None

    async def pending_for_message(
        self, guild_id: int, channel_id: int, message_id: int, before: datetime
    ) -> Expectation | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_expectations
            WHERE guild_id = ? AND channel_id = ? AND trigger_message_id = ?
              AND status = 'pending' AND created_at < ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (guild_id, channel_id, message_id, before.astimezone(UTC).isoformat()),
        )
        return self._from_row(row) if row else None

    async def latest_pending_in_channel(
        self, guild_id: int, channel_id: int, before: datetime
    ) -> Expectation | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_expectations
            WHERE guild_id = ? AND channel_id = ? AND status = 'pending'
              AND created_at < ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (guild_id, channel_id, before.astimezone(UTC).isoformat()),
        )
        return self._from_row(row) if row else None

    async def pending_in_channel(
        self, guild_id: int, channel_id: int, before: datetime, limit: int = 100
    ) -> list[Expectation]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_expectations
            WHERE guild_id = ? AND channel_id = ? AND status = 'pending'
              AND created_at < ? AND expires_at >= ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (
                guild_id,
                channel_id,
                before.astimezone(UTC).isoformat(),
                before.astimezone(UTC).isoformat(),
                max(1, min(limit, 1000)),
            ),
        )
        return [self._from_row(row) for row in rows]

    async def overdue(self, now: datetime, limit: int = 1000) -> list[Expectation]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_expectations
            WHERE status = 'pending' AND expires_at <= ?
            ORDER BY expires_at ASC
            LIMIT ?
            """,
            (now.astimezone(UTC).isoformat(), max(1, min(limit, 10_000))),
        )
        return [self._from_row(row) for row in rows]

    async def resolve(
        self,
        expectation_id: str,
        actual: OutcomeType,
        *,
        resolved_at: datetime,
        resolved_by_event_id: str | None,
    ) -> Expectation | None:
        resolved_at = resolved_at.astimezone(UTC)
        async with self._database.transaction() as connection:
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM cognition_expectations
                    WHERE expectation_id = ? AND status = 'pending'
                    """,
                    (expectation_id,),
                )
            ).fetchone()
            if row is None:
                return None

            distribution = self._distribution_from_json(str(row["distribution_json"]))
            difference = categorical_difference(distribution, actual)
            situation_json = str(row["situation_json"])
            guild_id = int(row["guild_id"])
            await connection.execute(
                """
                UPDATE cognition_expectations
                SET status = 'resolved', actual_outcome = ?, difference = ?,
                    resolved_by_event_id = ?, resolved_at = ?
                WHERE expectation_id = ? AND status = 'pending'
                """,
                (
                    actual.value,
                    difference,
                    resolved_by_event_id,
                    resolved_at.isoformat(),
                    expectation_id,
                ),
            )
            await connection.execute(
                """
                INSERT INTO cognition_patterns (
                    guild_id, situation_json, outcome_type, observations, weight
                ) VALUES (?, ?, ?, 1, 0.2)
                ON CONFLICT (guild_id, situation_json, outcome_type)
                DO UPDATE SET observations = observations + 1,
                              updated_at = CURRENT_TIMESTAMP
                """,
                (guild_id, situation_json, actual.value),
            )
            pattern_rows = await (
                await connection.execute(
                    """
                    SELECT outcome_type, observations
                    FROM cognition_patterns
                    WHERE guild_id = ? AND situation_json = ?
                    """,
                    (guild_id, situation_json),
                )
            ).fetchall()
            total_observations = sum(int(pattern[1]) for pattern in pattern_rows)
            denominator = total_observations + len(OUTCOMES)
            await connection.executemany(
                """
                UPDATE cognition_patterns
                SET weight = ?, updated_at = CURRENT_TIMESTAMP
                WHERE guild_id = ? AND situation_json = ? AND outcome_type = ?
                """,
                (
                    (
                        (int(pattern[1]) + 1) / denominator,
                        guild_id,
                        situation_json,
                        str(pattern[0]),
                    )
                    for pattern in pattern_rows
                ),
            )

        resolved = await self.get_expectation(expectation_id)
        return resolved

    async def get_expectation(self, expectation_id: str) -> Expectation | None:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_expectations WHERE expectation_id = ?",
            (expectation_id,),
        )
        return self._from_row(row) if row else None

    async def observation_seen(self, event_id: str) -> bool:
        row = await self._database.fetchone(
            "SELECT 1 FROM cognition_observed_events WHERE event_id = ?", (event_id,)
        )
        return row is not None

    async def mark_observed(self, event_id: str) -> None:
        await self._database.execute(
            "INSERT OR IGNORE INTO cognition_observed_events (event_id) VALUES (?)",
            (event_id,),
        )

    async def stats(self, guild_id: int) -> WorldModelStats:
        row = await self._database.fetchone(
            """
            SELECT
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved,
                AVG(CASE WHEN status = 'resolved' THEN difference END) AS average_d
            FROM cognition_expectations
            WHERE guild_id = ?
            """,
            (guild_id,),
        )
        pattern_row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM cognition_patterns WHERE guild_id = ?",
            (guild_id,),
        )
        return WorldModelStats(
            pending_expectations=int(row["pending"] or 0) if row else 0,
            resolved_expectations=int(row["resolved"] or 0) if row else 0,
            learned_patterns=int(pattern_row["count"] or 0) if pattern_row else 0,
            average_difference=(
                float(row["average_d"])
                if row is not None and row["average_d"] is not None
                else None
            ),
        )

    async def recent_expectations(self, guild_id: int, limit: int = 10) -> list[Expectation]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_expectations
            WHERE guild_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (guild_id, max(1, min(limit, 20))),
        )
        return [self._from_row(row) for row in rows]

    async def resolved_by_event(self, event_id: str) -> list[Expectation]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_expectations
            WHERE resolved_by_event_id = ? AND status = 'resolved'
            ORDER BY created_at ASC
            """,
            (event_id,),
        )
        return [self._from_row(row) for row in rows]

    async def unvectorized_resolved(self, limit: int = 1000) -> list[Expectation]:
        rows = await self._database.fetchall(
            """
            SELECT ce.*
            FROM cognition_expectations AS ce
            LEFT JOIN cognition_difference_vectors AS dv
              ON dv.expectation_id = ce.expectation_id
            WHERE ce.status = 'resolved' AND dv.expectation_id IS NULL
            ORDER BY ce.resolved_at ASC
            LIMIT ?
            """,
            (max(1, min(limit, 10_000)),),
        )
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _distribution_json(distribution: dict[OutcomeType, float]) -> str:
        return json.dumps(
            {outcome.value: probability for outcome, probability in distribution.items()},
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _distribution_from_json(value: str) -> dict[OutcomeType, float]:
        data = json.loads(value)
        return {OutcomeType(key): float(probability) for key, probability in data.items()}

    @classmethod
    def _from_row(cls, row: Any) -> Expectation:
        distribution = cls._distribution_from_json(str(row["distribution_json"]))
        predicted_outcome = OutcomeType(str(row["predicted_outcome"]))
        return Expectation(
            expectation_id=str(row["expectation_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            actor_id=int(row["actor_id"]) if row["actor_id"] is not None else None,
            trigger_event_id=str(row["trigger_event_id"]),
            trigger_message_id=(
                int(row["trigger_message_id"]) if row["trigger_message_id"] is not None else None
            ),
            situation=Situation.from_json(str(row["situation_json"])),
            prediction=Prediction(
                outcome=predicted_outcome,
                probability=float(row["predicted_probability"]),
                distribution=distribution,
                evidence_support=float(row["evidence_support"]),
            ),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            expires_at=datetime.fromisoformat(str(row["expires_at"])),
            status=ExpectationStatus(str(row["status"])),
            actual_outcome=(
                OutcomeType(str(row["actual_outcome"]))
                if row["actual_outcome"] is not None
                else None
            ),
            difference=float(row["difference"]) if row["difference"] is not None else None,
            resolved_by_event_id=(
                str(row["resolved_by_event_id"])
                if row["resolved_by_event_id"] is not None
                else None
            ),
            resolved_at=(
                datetime.fromisoformat(str(row["resolved_at"]))
                if row["resolved_at"] is not None
                else None
            ),
            source=str(row["source"]),
            observation_window_seconds=float(row["observation_window_seconds"]),
            allowed_outcomes=tuple(
                OutcomeType(value) for value in json.loads(str(row["allowed_outcomes_json"]))
            ),
        )

    @staticmethod
    def _payload_int(event: Event, key: str) -> int | None:
        value = event.payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
