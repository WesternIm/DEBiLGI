"""Persistence and slow aggregation of private reflections."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from debilgi.reflection.models import (
    Reflection,
    ReflectionGuidance,
    ReflectionStats,
    StrategyBelief,
)
from debilgi.storage.database import Database


class ReflectionRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def save(self, reflection: Reflection) -> bool:
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO cognition_reflections (
                    reflection_id, experience_id, guild_id, user_id, outcome_type,
                    expected_reward, actual_reward, prediction_error, surprise,
                    lesson_kind, strategies_json, evidence_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, json(?), json(?), ?)
                """,
                (
                    reflection.reflection_id,
                    reflection.experience_id,
                    reflection.guild_id,
                    reflection.user_id,
                    reflection.outcome_type,
                    reflection.expected_reward,
                    reflection.actual_reward,
                    reflection.prediction_error,
                    reflection.surprise,
                    reflection.lesson_kind,
                    _json(reflection.strategies),
                    _json(reflection.evidence),
                    reflection.created_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return False
            for strategy in reflection.strategies:
                await connection.execute(
                    """
                    INSERT INTO cognition_strategy_beliefs (
                        guild_id, strategy, observations, mean_reward,
                        mean_surprise, confidence, updated_at
                    ) VALUES (?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT (guild_id, strategy) DO UPDATE SET
                        mean_reward = (
                            cognition_strategy_beliefs.mean_reward
                                * cognition_strategy_beliefs.observations
                            + excluded.mean_reward
                        ) / (cognition_strategy_beliefs.observations + 1),
                        mean_surprise = (
                            cognition_strategy_beliefs.mean_surprise
                                * cognition_strategy_beliefs.observations
                            + excluded.mean_surprise
                        ) / (cognition_strategy_beliefs.observations + 1),
                        observations = cognition_strategy_beliefs.observations + 1,
                        confidence = CAST(cognition_strategy_beliefs.observations + 1 AS REAL)
                            / (cognition_strategy_beliefs.observations + 7),
                        updated_at = excluded.updated_at
                    """,
                    (
                        reflection.guild_id,
                        strategy,
                        reflection.actual_reward,
                        reflection.surprise,
                        1.0 / 7.0,
                        reflection.created_at.isoformat(),
                    ),
                )
        return True

    async def predict(self, guild_id: int, strategies: tuple[str, ...]) -> float:
        if not strategies:
            return 0.0
        placeholders = ",".join("?" for _ in strategies)
        rows = await self._database.fetchall(
            f"""
            SELECT mean_reward, confidence FROM cognition_strategy_beliefs
            WHERE guild_id = ? AND strategy IN ({placeholders})
            """,  # noqa: S608 -- placeholders are generated, not user supplied
            (guild_id, *strategies),
        )
        weighted = sum(float(row["mean_reward"]) * float(row["confidence"]) for row in rows)
        confidence = sum(float(row["confidence"]) for row in rows)
        return _clamp(weighted / confidence, -1.0, 1.0) if confidence else 0.0

    async def reflection_for_experience(self, experience_id: str) -> Reflection | None:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_reflections WHERE experience_id = ?", (experience_id,)
        )
        return self._reflection(row) if row else None

    async def beliefs(self, guild_id: int, limit: int = 20) -> tuple[StrategyBelief, ...]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_strategy_beliefs
            WHERE guild_id = ?
            ORDER BY confidence DESC, ABS(mean_reward) DESC, observations DESC
            LIMIT ?
            """,
            (guild_id, max(1, min(limit, 100))),
        )
        return tuple(self._belief(row) for row in rows)

    async def guidance(self, guild_id: int) -> ReflectionGuidance:
        beliefs = await self.beliefs(guild_id, 100)
        if not beliefs:
            return ReflectionGuidance()
        by_name = {belief.strategy: belief for belief in beliefs}
        length_options = tuple(
            belief
            for name in ("length:concise", "length:balanced", "length:extended")
            if (belief := by_name.get(name)) is not None
        )
        strongest_length = max(
            length_options,
            key=lambda item: (item.mean_reward * item.confidence, item.observations),
            default=None,
        )
        scales = {
            "length:concise": 0.78,
            "length:balanced": 1.0,
            "length:extended": 1.18,
        }
        total_confidence = sum(item.confidence for item in beliefs)
        confidence = total_confidence / len(beliefs)
        surprise = sum(item.mean_surprise * item.confidence for item in beliefs)
        surprise /= max(0.01, total_confidence)
        negative = sum(
            max(0.0, -item.mean_reward) * item.confidence for item in beliefs
        ) / max(0.01, total_confidence)
        strongest = max(
            beliefs,
            key=lambda item: (abs(item.mean_reward) * item.confidence, item.observations),
        )
        return ReflectionGuidance(
            length_scale=scales.get(strongest_length.strategy, 1.0) if strongest_length else 1.0,
            exploration=_clamp(surprise / 1.25, 0.0, 1.0),
            caution=_clamp(negative, 0.0, 1.0),
            confidence=_clamp(confidence, 0.0, 1.0),
            strongest_strategy=strongest.strategy,
        )

    async def stats(self, guild_id: int) -> ReflectionStats:
        row = await self._database.fetchone(
            """
            SELECT COUNT(*) AS reflections,
                   SUM(lesson_kind = 'revise_strategy') AS revisions,
                   AVG(surprise) AS average_surprise
            FROM cognition_reflections WHERE guild_id = ?
            """,
            (guild_id,),
        )
        belief_row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM cognition_strategy_beliefs WHERE guild_id = ?",
            (guild_id,),
        )
        return ReflectionStats(
            reflections=int(row["reflections"] or 0) if row else 0,
            revisions=int(row["revisions"] or 0) if row else 0,
            average_surprise=float(row["average_surprise"] or 0.0) if row else 0.0,
            strategy_beliefs=int(belief_row["count"] or 0) if belief_row else 0,
        )

    @staticmethod
    def _reflection(row: Any) -> Reflection:
        return Reflection(
            reflection_id=str(row["reflection_id"]),
            experience_id=str(row["experience_id"]),
            guild_id=int(row["guild_id"]),
            user_id=int(row["user_id"]) if row["user_id"] is not None else None,
            outcome_type=str(row["outcome_type"]),
            expected_reward=float(row["expected_reward"]),
            actual_reward=float(row["actual_reward"]),
            prediction_error=float(row["prediction_error"]),
            surprise=float(row["surprise"]),
            lesson_kind=str(row["lesson_kind"]),
            strategies=tuple(json.loads(str(row["strategies_json"]))),
            evidence=tuple(json.loads(str(row["evidence_json"]))),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _belief(row: Any) -> StrategyBelief:
        return StrategyBelief(
            guild_id=int(row["guild_id"]),
            strategy=str(row["strategy"]),
            observations=int(row["observations"]),
            mean_reward=float(row["mean_reward"]),
            mean_surprise=float(row["mean_surprise"]),
            confidence=float(row["confidence"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
