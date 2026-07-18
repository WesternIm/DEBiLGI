"""SQLite persistence and reinforcement for utterance consequences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from debilgi.experience.models import PreparedReply, SelfModel, UtteranceExperience
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class ExperienceStats:
    pending: int
    resolved: int
    expired: int
    average_reward: float | None
    learned_transitions: int


class ExperienceRepository:
    def __init__(self, database: Database, *, learning_rate: float = 0.35) -> None:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1")
        self._database = database
        self._learning_rate = learning_rate

    async def register(
        self,
        *,
        guild_id: int,
        channel_id: int,
        user_id: int | None,
        prompt_event_id: str,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
        expected_reward: float = 0.0,
    ) -> UtteranceExperience:
        experience_id = f"experience:{uuid4()}"
        expected_reward = max(-1.0, min(1.0, expected_reward))
        await self._database.execute(
            """
            INSERT OR IGNORE INTO language_utterance_experiences (
                experience_id, guild_id, channel_id, user_id, prompt_event_id,
                intent_id, response_message_id, response_tokens_json,
                genome_signature, candidate_count, created_at, initiative, goal_id,
                expected_reward
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experience_id,
                guild_id,
                channel_id,
                user_id,
                prompt_event_id,
                reply.intent_id,
                response_message_id,
                _json(reply.tokens),
                reply.genome_signature,
                reply.candidate_count,
                created_at.isoformat(),
                int(reply.initiative),
                reply.goal_id,
                expected_reward,
            ),
        )
        row = await self._database.fetchone(
            "SELECT * FROM language_utterance_experiences WHERE prompt_event_id = ?",
            (prompt_event_id,),
        )
        if row is None:
            raise RuntimeError("failed to register utterance experience")
        return self._from_row(row)

    async def pending_for_response_message(
        self, guild_id: int, response_message_id: int
    ) -> UtteranceExperience | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM language_utterance_experiences
            WHERE guild_id = ? AND response_message_id = ? AND status = 'pending'
            """,
            (guild_id, response_message_id),
        )
        return self._from_row(row) if row else None

    async def latest_pending_for_user(
        self, guild_id: int, channel_id: int, user_id: int
    ) -> UtteranceExperience | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM language_utterance_experiences
            WHERE guild_id = ? AND channel_id = ? AND user_id = ? AND status = 'pending'
            ORDER BY created_at DESC LIMIT 1
            """,
            (guild_id, channel_id, user_id),
        )
        return self._from_row(row) if row else None

    async def pending_before(self, cutoff: datetime, limit: int = 100) -> list[UtteranceExperience]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM language_utterance_experiences
            WHERE status = 'pending' AND created_at <= ?
            ORDER BY created_at ASC LIMIT ?
            """,
            (cutoff.isoformat(), max(1, min(limit, 1000))),
        )
        return [self._from_row(row) for row in rows]

    async def resolve(
        self,
        experience: UtteranceExperience,
        *,
        outcome_type: str,
        reward: float,
        resolved_at: datetime,
        expired: bool = False,
    ) -> bool:
        reward = max(-1.0, min(1.0, reward))
        status = "expired" if expired else "resolved"
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                UPDATE language_utterance_experiences
                SET status = ?, outcome_type = ?, reward = ?, resolved_at = ?
                WHERE experience_id = ? AND status = 'pending'
                """,
                (
                    status,
                    outcome_type,
                    reward,
                    resolved_at.isoformat(),
                    experience.experience_id,
                ),
            )
            if cursor.rowcount != 1:
                return False
            adjustment = reward * self._learning_rate
            positive = int(reward > 0.0)
            negative = int(reward < 0.0)
            for context, next_token in self._transitions(experience.response_tokens):
                await connection.execute(
                    """
                    INSERT INTO language_experience_transitions (
                        guild_id, context_json, next_token, weight, uses,
                        positive_outcomes, negative_outcomes, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT (guild_id, context_json, next_token) DO UPDATE SET
                        weight = MAX(-4.0, MIN(4.0,
                            language_experience_transitions.weight + excluded.weight)),
                        uses = language_experience_transitions.uses + 1,
                        positive_outcomes = language_experience_transitions.positive_outcomes
                            + excluded.positive_outcomes,
                        negative_outcomes = language_experience_transitions.negative_outcomes
                            + excluded.negative_outcomes,
                        updated_at = excluded.updated_at
                    """,
                    (
                        experience.guild_id,
                        _json(context),
                        next_token,
                        adjustment,
                        positive,
                        negative,
                        resolved_at.isoformat(),
                    ),
                )
            await connection.execute(
                """
                INSERT INTO cognition_self_models (
                    guild_id, utterances_observed, positive_outcomes, negative_outcomes,
                    average_reward, initiative_attempts, initiative_successes,
                    efficacy, updated_at
                ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (guild_id) DO UPDATE SET
                    average_reward = (
                        cognition_self_models.average_reward
                            * cognition_self_models.utterances_observed
                        + excluded.average_reward
                    ) / (cognition_self_models.utterances_observed + 1),
                    utterances_observed = cognition_self_models.utterances_observed + 1,
                    positive_outcomes = cognition_self_models.positive_outcomes
                        + excluded.positive_outcomes,
                    negative_outcomes = cognition_self_models.negative_outcomes
                        + excluded.negative_outcomes,
                    initiative_attempts = cognition_self_models.initiative_attempts
                        + excluded.initiative_attempts,
                    initiative_successes = cognition_self_models.initiative_successes
                        + excluded.initiative_successes,
                    efficacy = MAX(0.0, MIN(1.0,
                        cognition_self_models.efficacy * 0.85 + excluded.efficacy * 0.15)),
                    updated_at = excluded.updated_at
                """,
                (
                    experience.guild_id,
                    positive,
                    negative,
                    reward,
                    int(experience.initiative),
                    int(experience.initiative and reward > 0.0),
                    (reward + 1.0) / 2.0,
                    resolved_at.isoformat(),
                ),
            )
        return True

    async def self_model(self, guild_id: int) -> SelfModel:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_self_models WHERE guild_id = ?", (guild_id,)
        )
        if row is None:
            return SelfModel(guild_id, 0, 0, 0, 0.0, 0, 0, 0.5, None)
        return SelfModel(
            guild_id=guild_id,
            utterances_observed=int(row["utterances_observed"]),
            positive_outcomes=int(row["positive_outcomes"]),
            negative_outcomes=int(row["negative_outcomes"]),
            average_reward=float(row["average_reward"]),
            initiative_attempts=int(row["initiative_attempts"]),
            initiative_successes=int(row["initiative_successes"]),
            efficacy=float(row["efficacy"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    async def transition_adjustments(
        self, guild_id: int, limit: int = 1000
    ) -> tuple[tuple[tuple[str, ...], float], ...]:
        rows = await self._database.fetchall(
            """
            SELECT context_json, next_token, weight
            FROM language_experience_transitions
            WHERE guild_id = ? AND weight != 0.0
            ORDER BY ABS(weight) DESC, updated_at DESC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 5000))),
        )
        return tuple(
            (
                (*tuple(json.loads(str(row["context_json"]))), str(row["next_token"])),
                float(row["weight"]),
            )
            for row in rows
        )

    async def stats(self, guild_id: int) -> ExperienceStats:
        row = await self._database.fetchone(
            """
            SELECT
                SUM(status = 'pending') AS pending,
                SUM(status = 'resolved') AS resolved,
                SUM(status = 'expired') AS expired,
                AVG(reward) AS average_reward
            FROM language_utterance_experiences WHERE guild_id = ?
            """,
            (guild_id,),
        )
        transition_row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM language_experience_transitions WHERE guild_id = ?",
            (guild_id,),
        )
        return ExperienceStats(
            pending=int(row["pending"] or 0) if row else 0,
            resolved=int(row["resolved"] or 0) if row else 0,
            expired=int(row["expired"] or 0) if row else 0,
            average_reward=(
                float(row["average_reward"])
                if row is not None and row["average_reward"] is not None
                else None
            ),
            learned_transitions=int(transition_row["count"] or 0) if transition_row else 0,
        )

    async def unreflected(self, limit: int = 500) -> tuple[UtteranceExperience, ...]:
        rows = await self._database.fetchall(
            """
            SELECT experience.*
            FROM language_utterance_experiences AS experience
            LEFT JOIN cognition_reflections AS reflection
              ON reflection.experience_id = experience.experience_id
            WHERE experience.reward IS NOT NULL
              AND experience.resolved_at IS NOT NULL
              AND reflection.experience_id IS NULL
            ORDER BY experience.resolved_at ASC LIMIT ?
            """,
            (max(1, min(limit, 5000)),),
        )
        return tuple(self._from_row(row) for row in rows)

    @staticmethod
    def _transitions(tokens: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], str], ...]:
        result: list[tuple[tuple[str, ...], str]] = []
        for position in range(1, len(tokens)):
            for size in range(1, min(3, position) + 1):
                result.append((tokens[position - size : position], tokens[position]))
        return tuple(result)

    @staticmethod
    def _from_row(row: Any) -> UtteranceExperience:
        return UtteranceExperience(
            experience_id=str(row["experience_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            user_id=int(row["user_id"]) if row["user_id"] is not None else None,
            prompt_event_id=str(row["prompt_event_id"]),
            intent_id=str(row["intent_id"]),
            response_message_id=int(row["response_message_id"]),
            response_tokens=tuple(json.loads(str(row["response_tokens_json"]))),
            genome_signature=str(row["genome_signature"]),
            candidate_count=int(row["candidate_count"]),
            status=str(row["status"]),
            outcome_type=str(row["outcome_type"]) if row["outcome_type"] else None,
            reward=float(row["reward"]) if row["reward"] is not None else None,
            created_at=datetime.fromisoformat(str(row["created_at"])),
            resolved_at=(
                datetime.fromisoformat(str(row["resolved_at"])) if row["resolved_at"] else None
            ),
            initiative=bool(row["initiative"]),
            goal_id=str(row["goal_id"]) if row["goal_id"] else None,
            expected_reward=float(row["expected_reward"]),
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
