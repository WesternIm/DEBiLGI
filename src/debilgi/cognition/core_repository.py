"""SQLite persistence for episodes, semantic memory, models, state, and shadow intents."""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from debilgi.cognition.difference import DifferenceVector
from debilgi.cognition.expectations import Expectation, OutcomeType
from debilgi.cognition.models import (
    ActionIntent,
    ActionType,
    Assessment,
    EntityModel,
    EntityType,
    Episode,
    EpisodeStatus,
    InternalState,
    SemanticMemory,
)
from debilgi.events.models import Event, EventType
from debilgi.storage.database import Database


class CognitiveRepository:
    def __init__(self, database: Database, *, hot_model_cache_size: int = 256) -> None:
        if hot_model_cache_size <= 0:
            raise ValueError("hot_model_cache_size must be positive")
        self._database = database
        self._hot_model_cache_size = hot_model_cache_size
        self._model_cache: OrderedDict[tuple[int, EntityType, int], EntityModel] = OrderedDict()

    async def claim_observation(self, event_id: str) -> bool:
        cursor = await self._database.execute(
            """
            INSERT OR IGNORE INTO cognition_core_observations (event_id, status)
            VALUES (?, 'processing')
            """,
            (event_id,),
        )
        if cursor.rowcount == 1:
            return True
        retry = await self._database.execute(
            """
            UPDATE cognition_core_observations
            SET status = 'processing', last_error = NULL
            WHERE event_id = ? AND status = 'failed'
            """,
            (event_id,),
        )
        return retry.rowcount == 1

    async def finish_observation(self, event_id: str, processing_ms: float) -> None:
        await self._database.execute(
            """
            UPDATE cognition_core_observations
            SET status = 'processed', processing_ms = ?, last_error = NULL
            WHERE event_id = ?
            """,
            (max(0.0, processing_ms), event_id),
        )

    async def fail_observation(self, event_id: str, error: str) -> None:
        await self._database.execute(
            """
            UPDATE cognition_core_observations
            SET status = 'failed', last_error = ?
            WHERE event_id = ?
            """,
            (error[:1000], event_id),
        )

    async def record_difference(self, expectation: Expectation, vector: DifferenceVector) -> None:
        await self._database.execute(
            """
            INSERT OR IGNORE INTO cognition_difference_vectors (
                expectation_id, guild_id, scalar_d, outcome_mismatch,
                participant_mismatch, timing_mismatch, intensity_mismatch,
                duration_mismatch, unexpectedness, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                expectation.expectation_id,
                expectation.guild_id,
                vector.scalar,
                vector.outcome_mismatch,
                vector.participant_mismatch,
                vector.timing_mismatch,
                vector.intensity_mismatch,
                vector.duration_mismatch,
                vector.unexpectedness,
                expectation.resolved_at.isoformat() if expectation.resolved_at else "",
            ),
        )

    async def recent_differences(self, guild_id: int, limit: int = 10) -> list[dict[str, Any]]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_difference_vectors
            WHERE guild_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 100))),
        )
        return [dict(row) for row in rows]

    async def attach_event(
        self,
        event: Event,
        *,
        unexpectedness: float,
        episode_gap: timedelta,
        observed_outcome: str | None,
    ) -> Episode:
        existing = await self._database.fetchone(
            """
            SELECT episode_id FROM cognition_episode_events WHERE event_id = ?
            """,
            (event.event_id,),
        )
        if existing:
            episode = await self.get_episode(str(existing["episode_id"]))
            if episode is None:
                raise RuntimeError("episode event points to a missing episode")
            return episode

        channel_id = event.channel_id or 0
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_episodes
            WHERE guild_id = ? AND channel_id = ? AND status = 'open'
            ORDER BY ended_at DESC LIMIT 1
            """,
            (event.guild_id, channel_id),
        )
        if row is not None:
            ended_at = datetime.fromisoformat(str(row["ended_at"]))
            if event.utc_timestamp - ended_at > episode_gap:
                await self._database.execute(
                    "UPDATE cognition_episodes SET status = 'closed' WHERE episode_id = ?",
                    (str(row["episode_id"]),),
                )
                row = None

        if row is None:
            episode_id = f"episode:{uuid4()}"
            participants = self._event_participants(event)
            event_types = [event.event_type.value]
            signature = self._episode_signature(event_types)
            importance = _clamp(0.10 + unexpectedness * 0.60)
            await self._database.execute(
                """
                INSERT INTO cognition_episodes (
                    episode_id, guild_id, channel_id, started_at, ended_at, status,
                    participants_json, key_event_types_json, observed_outcome,
                    unexpectedness, importance, confidence, signature
                ) VALUES (?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, 0.5, ?)
                """,
                (
                    episode_id,
                    event.guild_id,
                    channel_id,
                    event.utc_timestamp.isoformat(),
                    event.utc_timestamp.isoformat(),
                    _json(participants),
                    _json(event_types),
                    observed_outcome,
                    unexpectedness,
                    importance,
                    signature,
                ),
            )
            position = 0
        else:
            episode_id = str(row["episode_id"])
            participants = sorted(
                set(json.loads(str(row["participants_json"])))
                | set(self._event_participants(event))
            )
            event_types = list(json.loads(str(row["key_event_types_json"])))
            event_types.append(event.event_type.value)
            event_types = event_types[-20:]
            signature = self._episode_signature(event_types)
            importance = _clamp(
                max(float(row["importance"]), 0.10 + unexpectedness * 0.60)
                + min(0.20, len(event_types) * 0.01)
            )
            unexpectedness = max(float(row["unexpectedness"]), unexpectedness)
            count_row = await self._database.fetchone(
                "SELECT COUNT(*) AS count FROM cognition_episode_events WHERE episode_id = ?",
                (episode_id,),
            )
            position = int(count_row["count"]) if count_row else 0
            await self._database.execute(
                """
                UPDATE cognition_episodes
                SET ended_at = ?, participants_json = ?, key_event_types_json = ?,
                    observed_outcome = COALESCE(?, observed_outcome),
                    unexpectedness = ?, importance = ?, signature = ?
                WHERE episode_id = ?
                """,
                (
                    event.utc_timestamp.isoformat(),
                    _json(participants),
                    _json(event_types),
                    observed_outcome,
                    unexpectedness,
                    importance,
                    signature,
                    episode_id,
                ),
            )

        await self._database.execute(
            """
            INSERT OR IGNORE INTO cognition_episode_events (episode_id, event_id, position)
            VALUES (?, ?, ?)
            """,
            (episode_id, event.event_id, position),
        )
        episode = await self.get_episode(episode_id)
        if episode is None:
            raise RuntimeError("failed to create episode")
        return episode

    async def get_episode(self, episode_id: str) -> Episode | None:
        row = await self._database.fetchone(
            "SELECT * FROM cognition_episodes WHERE episode_id = ?", (episode_id,)
        )
        return await self._episode_from_row(row) if row else None

    async def recent_episodes(self, guild_id: int, limit: int = 10) -> list[Episode]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_episodes
            WHERE guild_id = ? ORDER BY ended_at DESC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 100))),
        )
        return [await self._episode_from_row(row) for row in rows]

    async def close_stale_episodes(self, now: datetime, episode_gap: timedelta) -> int:
        cursor = await self._database.execute(
            """
            UPDATE cognition_episodes SET status = 'closed'
            WHERE status = 'open' AND ended_at <= ?
            """,
            ((now.astimezone(UTC) - episode_gap).isoformat(),),
        )
        return cursor.rowcount

    async def consolidate(self, guild_id: int | None = None, limit: int = 100) -> int:
        where = "status = 'closed' AND consolidated_at IS NULL"
        parameters: list[Any] = []
        if guild_id is not None:
            where += " AND guild_id = ?"
            parameters.append(guild_id)
        parameters.append(max(1, min(limit, 1000)))
        rows = await self._database.fetchall(
            f"""
            SELECT * FROM cognition_episodes WHERE {where}
            ORDER BY ended_at ASC LIMIT ?
            """,
            parameters,
        )
        for row in rows:
            episode = await self._episode_from_row(row)
            observed = 1.0 if len(episode.event_ids) > 1 else 0.0
            first_type = episode.key_event_types[0] if episode.key_event_types else "unknown"
            feature = f"after:{first_type}:continuation_probability"
            scopes = [
                ("server", episode.guild_id),
                ("channel", episode.channel_id),
                *(("user", participant) for participant in episode.participants),
            ]
            for scope_type, scope_id in scopes:
                await self._update_semantic(
                    episode,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    feature=feature,
                    observed=observed,
                )
                await self._update_subconscious(
                    episode,
                    scope_type=scope_type,
                    scope_id=scope_id,
                    observed=observed,
                )
            await self._database.execute(
                """
                UPDATE cognition_episodes SET consolidated_at = ? WHERE episode_id = ?
                """,
                (datetime.now(UTC).isoformat(), episode.episode_id),
            )
        return len(rows)

    async def semantic_memories(
        self,
        guild_id: int,
        *,
        scope_type: str | None = None,
        scope_id: int | None = None,
        limit: int = 20,
    ) -> list[SemanticMemory]:
        clauses = ["guild_id = ?"]
        parameters: list[Any] = [guild_id]
        if scope_type is not None:
            clauses.append("scope_type = ?")
            parameters.append(scope_type)
        if scope_id is not None:
            clauses.append("scope_id = ?")
            parameters.append(scope_id)
        parameters.append(max(1, min(limit, 200)))
        rows = await self._database.fetchall(
            f"""
            SELECT * FROM cognition_semantic_memories
            WHERE {" AND ".join(clauses)}
            ORDER BY confidence DESC, updated_at DESC LIMIT ?
            """,
            parameters,
        )
        return [self._semantic_from_row(row) for row in rows]

    async def decay_semantic(self, now: datetime) -> int:
        rows = await self._database.fetchall(
            """
            SELECT semantic_id, confidence, decay_rate, updated_at
            FROM cognition_semantic_memories
            """
        )
        changed = 0
        for row in rows:
            updated_at = datetime.fromisoformat(str(row["updated_at"]))
            days = max(0.0, (now.astimezone(UTC) - updated_at).total_seconds() / 86400.0)
            if days <= 0:
                continue
            confidence = float(row["confidence"]) * math.exp(-float(row["decay_rate"]) * days)
            await self._database.execute(
                """
                UPDATE cognition_semantic_memories
                SET confidence = ?, updated_at = ? WHERE semantic_id = ?
                """,
                (confidence, now.astimezone(UTC).isoformat(), str(row["semantic_id"])),
            )
            changed += 1
        return changed

    async def update_entity_model(
        self,
        event: Event,
        *,
        unexpectedness: float,
        active_expectations: int,
        actual_outcomes: tuple[OutcomeType, ...],
        reply_delays: tuple[float, ...] = (),
    ) -> EntityModel:
        entities = [
            (EntityType.SERVER, event.guild_id),
            (EntityType.SELF, 0),
        ]
        if event.channel_id is not None:
            entities.append((EntityType.CHANNEL, event.channel_id))
        if event.actor_id is not None:
            entities.append((EntityType.USER, event.actor_id))
        result: EntityModel | None = None
        for entity_type, entity_id in entities:
            result_item = await self._upsert_entity(
                event,
                entity_type,
                entity_id,
                unexpectedness=unexpectedness,
                active_expectations=active_expectations,
                actual_outcomes=actual_outcomes,
                reply_delays=reply_delays,
            )
            if entity_type is EntityType.USER or result is None:
                result = result_item
        assert result is not None
        return result

    async def record_bot_action_response(
        self, guild_id: int, user_id: int, sample: float, rate: float
    ) -> EntityModel | None:
        """How this person answers the agent's actions (4.42, flag only): the moving average of
        the sign of their reactions to the agent's moves, 0 cold, 0.5 neutral, 1 warm. The model
        of the person is the only place it lives (the cache is kept in step)."""
        old = await self.entity_model(guild_id, EntityType.USER, user_id)
        if old is None:
            return None
        value = _ema(old.bot_action_response, sample, rate)
        await self._database.execute(
            """UPDATE cognition_entity_models SET bot_action_response = ?
               WHERE guild_id = ? AND entity_type = ? AND entity_id = ?""",
            (value, guild_id, EntityType.USER.value, user_id),
        )
        model = replace(old, bot_action_response=value)
        self._cache_model(model)
        return model

    async def entity_model(
        self, guild_id: int, entity_type: EntityType, entity_id: int
    ) -> EntityModel | None:
        key = (guild_id, entity_type, entity_id)
        cached = self._model_cache.get(key)
        if cached is not None:
            self._model_cache.move_to_end(key)
            return cached
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_entity_models
            WHERE guild_id = ? AND entity_type = ? AND entity_id = ?
            """,
            (guild_id, entity_type.value, entity_id),
        )
        if row is None:
            return None
        model = self._entity_from_row(row)
        self._cache_model(model)
        return model

    async def update_internal_state(
        self,
        event: Event,
        model: EntityModel,
        *,
        unexpectedness: float,
        active_expectations: int,
        importance: float,
    ) -> InternalState:
        subject_id = event.actor_id or 0
        previous = await self.internal_state(event.guild_id, subject_id)
        if previous is None:
            decay = 0.0
            old_tension = 0.0
            old_interest = 0.0
        else:
            elapsed = max(0.0, (event.utc_timestamp - previous.updated_at).total_seconds())
            decay = 0.5 ** (elapsed / 1800.0)
            old_tension = previous.tension
            old_interest = previous.interest
        activation = self._event_intensity(event)
        uncertainty = _clamp(1.0 - model.predictability)
        background_bias = await self._subconscious_bias(
            event.guild_id, subject_id, event.event_type.value
        )
        tension = _clamp(
            old_tension * decay * 0.75
            + unexpectedness * 0.35
            + min(0.2, active_expectations * 0.04)
            + max(0.0, -background_bias) * 0.10
        )
        interest = _clamp(
            old_interest * decay * 0.60
            + importance * 0.25
            + activation * 0.15
            + abs(background_bias) * 0.15
        )
        state = InternalState(
            guild_id=event.guild_id,
            subject_id=subject_id,
            unexpectedness=_clamp(unexpectedness),
            uncertainty=uncertainty,
            activation=activation,
            tension=tension,
            interest=interest,
            familiarity=model.familiarity,
            expected_control=model.expectation_reliability,
            goal_alignment=_clamp(1.0 - active_expectations / 10.0),
            updated_at=event.utc_timestamp,
        )
        await self._database.execute(
            """
            INSERT INTO cognition_internal_states (
                guild_id, subject_id, unexpectedness, uncertainty, activation,
                tension, interest, familiarity, expected_control, goal_alignment,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (guild_id, subject_id) DO UPDATE SET
                unexpectedness = excluded.unexpectedness,
                uncertainty = excluded.uncertainty,
                activation = excluded.activation,
                tension = excluded.tension,
                interest = excluded.interest,
                familiarity = excluded.familiarity,
                expected_control = excluded.expected_control,
                goal_alignment = excluded.goal_alignment,
                updated_at = excluded.updated_at
            """,
            (
                state.guild_id,
                state.subject_id,
                state.unexpectedness,
                state.uncertainty,
                state.activation,
                state.tension,
                state.interest,
                state.familiarity,
                state.expected_control,
                state.goal_alignment,
                state.updated_at.isoformat(),
            ),
        )
        return state

    async def internal_state(self, guild_id: int, subject_id: int) -> InternalState | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_internal_states WHERE guild_id = ? AND subject_id = ?
            """,
            (guild_id, subject_id),
        )
        return self._state_from_row(row) if row else None

    async def create_assessment(
        self, event: Event, model: EntityModel, episode: Episode
    ) -> Assessment:
        confidence = _clamp(model.observations / (model.observations + 5.0))
        contradiction_rows = await self._database.fetchall(
            """
            SELECT provenance_json FROM cognition_semantic_memories
            WHERE guild_id = ? AND scope_type = ? AND scope_id = ?
              AND contradictions > 0
            ORDER BY updated_at DESC LIMIT 5
            """,
            (event.guild_id, model.entity_type.value, model.entity_id),
        )
        contradicting = tuple(
            str(episode_id)
            for row in contradiction_rows
            for episode_id in json.loads(str(row["provenance_json"]))[-2:]
        )
        assessment = Assessment(
            assessment_id=f"assessment:{uuid4()}",
            guild_id=event.guild_id,
            object_type=model.entity_type.value,
            object_id=model.entity_id,
            dimensions={
                "familiarity": model.familiarity,
                "predictability": model.predictability,
                "reply_probability": model.reply_probability,
                "pattern_stability": model.pattern_stability,
            },
            confidence=confidence,
            supporting_episode_ids=(episode.episode_id,),
            contradicting_episode_ids=contradicting,
            alternative_explanations=(
                "channel-level activity may explain the observation",
                "the current sample may be too small",
            ),
            data_gap=1.0 - confidence,
            next_review_at=event.utc_timestamp + timedelta(days=1),
            created_at=event.utc_timestamp,
        )
        await self._database.execute(
            """
            INSERT INTO cognition_assessments (
                assessment_id, guild_id, object_type, object_id, dimensions_json,
                confidence, supporting_episode_ids_json,
                contradicting_episode_ids_json, alternative_explanations_json,
                data_gap, next_review_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment.assessment_id,
                assessment.guild_id,
                assessment.object_type,
                assessment.object_id,
                _json(assessment.dimensions),
                assessment.confidence,
                _json(assessment.supporting_episode_ids),
                _json(assessment.contradicting_episode_ids),
                _json(assessment.alternative_explanations),
                assessment.data_gap,
                assessment.next_review_at.isoformat(),
                assessment.created_at.isoformat(),
            ),
        )
        return assessment

    async def save_intent(self, intent: ActionIntent) -> None:
        await self._database.execute(
            """
            INSERT INTO cognition_action_intents (
                intent_id, guild_id, action_type, target_id, channel_id, urgency,
                confidence, expected_effect_json, reasoning_trace_json, evidence_json,
                decision_context_json, created_at, expires_at, shadow_only, executed_at,
                actual_outcome, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, ?, ?)
            """,
            (
                intent.intent_id,
                intent.guild_id,
                intent.action_type.value,
                intent.target_id,
                intent.channel_id,
                intent.urgency,
                intent.confidence,
                _json(intent.expected_effect),
                _json(intent.reasoning_trace),
                _json(intent.evidence),
                _json(intent.decision_context),
                intent.created_at.isoformat(),
                intent.expires_at.isoformat(),
                intent.actual_outcome,
                intent.evaluated_at.isoformat() if intent.evaluated_at else None,
            ),
        )

    async def evaluate_prior_intents(self, event: Event, limit: int = 20) -> int:
        # These proposals are strictly shadow-only (see ActionIntent). A next
        # channel event is not evidence that an unexecuted action had an effect.
        # Actual speech actions learn in MeaningIntentRepository after sending.
        return 0

    async def recent_intents(self, guild_id: int, limit: int = 10) -> list[ActionIntent]:
        rows = await self._database.fetchall(
            """
            SELECT * FROM cognition_action_intents
            WHERE guild_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (guild_id, max(1, min(limit, 100))),
        )
        return [self._intent_from_row(row) for row in rows]

    async def intent_for_event(self, event_id: str) -> ActionIntent | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_action_intents
            WHERE json_extract(evidence_json, '$[0]') = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (event_id,),
        )
        return self._intent_from_row(row) if row else None

    async def mark_raw_retention(self, event: Event, retention: timedelta) -> None:
        content = event.payload.get("content")
        if not isinstance(content, str):
            return
        await self._database.execute(
            """
            UPDATE events SET raw_expires_at = COALESCE(raw_expires_at, ?)
            WHERE event_id = ?
            """,
            ((event.utc_timestamp + retention).isoformat(), event.event_id),
        )

    async def backfill_raw_retention(self, retention: timedelta, limit: int = 1000) -> int:
        rows = await self._database.fetchall(
            """
            SELECT event_id, occurred_at FROM events
            WHERE raw_expires_at IS NULL AND json_type(payload_json, '$.content') IS NOT NULL
            ORDER BY occurred_at ASC LIMIT ?
            """,
            (max(1, min(limit, 10_000)),),
        )
        for row in rows:
            occurred_at = datetime.fromisoformat(str(row["occurred_at"]))
            await self._database.execute(
                "UPDATE events SET raw_expires_at = ? WHERE event_id = ?",
                ((occurred_at + retention).isoformat(), str(row["event_id"])),
            )
        return len(rows)

    async def redact_expired_raw_text(self, now: datetime, limit: int = 1000) -> int:
        rows = await self._database.fetchall(
            """
            SELECT event_id FROM events
            WHERE raw_expires_at IS NOT NULL AND raw_expires_at <= ?
              AND raw_redacted_at IS NULL
            ORDER BY raw_expires_at ASC LIMIT ?
            """,
            (now.astimezone(UTC).isoformat(), max(1, min(limit, 10_000))),
        )
        for row in rows:
            await self._database.execute(
                """
                UPDATE events
                SET payload_json = json_remove(payload_json, '$.content'), raw_redacted_at = ?
                WHERE event_id = ? AND raw_redacted_at IS NULL
                """,
                (now.astimezone(UTC).isoformat(), str(row["event_id"])),
            )
        return len(rows)

    async def average_processing_ms(self, guild_id: int) -> float | None:
        row = await self._database.fetchone(
            """
            SELECT AVG(co.processing_ms) AS average_ms
            FROM cognition_core_observations AS co
            JOIN events AS e ON e.event_id = co.event_id
            WHERE e.guild_id = ? AND co.status = 'processed'
            """,
            (guild_id,),
        )
        return float(row["average_ms"]) if row and row["average_ms"] is not None else None

    async def _update_semantic(
        self,
        episode: Episode,
        *,
        scope_type: str,
        scope_id: int,
        feature: str,
        observed: float,
    ) -> None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_semantic_memories
            WHERE guild_id = ? AND scope_type = ? AND scope_id = ? AND feature = ?
            """,
            (episode.guild_id, scope_type, scope_id, feature),
        )
        now = episode.ended_at.isoformat()
        if row is None:
            await self._database.execute(
                """
                INSERT INTO cognition_semantic_memories (
                    semantic_id, guild_id, scope_type, scope_id, feature, estimate,
                    confidence, confirmations, contradictions, updated_at,
                    decay_rate, provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, 0.01, ?)
                """,
                (
                    f"semantic:{uuid4()}",
                    episode.guild_id,
                    scope_type,
                    scope_id,
                    feature,
                    observed,
                    1.0 / 3.0,
                    now,
                    _json([episode.episode_id]),
                ),
            )
            return
        old_estimate = float(row["estimate"])
        confirmations = int(row["confirmations"])
        contradictions = int(row["contradictions"])
        aligned = (old_estimate >= 0.5) == (observed >= 0.5)
        confirmations += int(aligned)
        contradictions += int(not aligned)
        old_count = max(1, confirmations + contradictions - 1)
        estimate = (old_estimate * old_count + observed) / (old_count + 1)
        total = confirmations + contradictions
        consistency = 1.0 - contradictions / max(1, total)
        confidence = _clamp(total / (total + 2.0) * consistency)
        provenance = list(json.loads(str(row["provenance_json"])))
        provenance.append(episode.episode_id)
        await self._database.execute(
            """
            UPDATE cognition_semantic_memories
            SET estimate = ?, confidence = ?, confirmations = ?, contradictions = ?,
                updated_at = ?, provenance_json = ?
            WHERE semantic_id = ?
            """,
            (
                estimate,
                confidence,
                confirmations,
                contradictions,
                now,
                _json(provenance[-20:]),
                str(row["semantic_id"]),
            ),
        )

    async def _update_subconscious(
        self,
        episode: Episode,
        *,
        scope_type: str,
        scope_id: int,
        observed: float,
    ) -> None:
        key = f"episode:{episode.signature}:continuation"
        signed = observed * 2.0 - 1.0
        await self._database.execute(
            """
            INSERT INTO cognition_subconscious_weights (
                guild_id, scope_type, scope_id, association_key, value,
                confidence, updated_at, provenance_json
            ) VALUES (?, ?, ?, ?, ?, 0.2, ?, ?)
            ON CONFLICT (guild_id, scope_type, scope_id, association_key) DO UPDATE SET
                value = cognition_subconscious_weights.value * 0.8 + excluded.value * 0.2,
                confidence = MIN(1.0, cognition_subconscious_weights.confidence + 0.1),
                updated_at = excluded.updated_at,
                provenance_json = excluded.provenance_json
            """,
            (
                episode.guild_id,
                scope_type,
                scope_id,
                key,
                signed,
                episode.ended_at.isoformat(),
                _json([episode.episode_id]),
            ),
        )

    async def _upsert_entity(
        self,
        event: Event,
        entity_type: EntityType,
        entity_id: int,
        *,
        unexpectedness: float,
        active_expectations: int,
        actual_outcomes: tuple[OutcomeType, ...],
        reply_delays: tuple[float, ...],
    ) -> EntityModel:
        old = await self.entity_model(event.guild_id, entity_type, entity_id)
        observations = (old.observations if old else 0) + 1
        familiarity = 1.0 - math.exp(-observations / 10.0)
        intensity_input = self._event_intensity(event)
        intensity = _ema(old.interaction_intensity if old else 0.0, intensity_input, 0.20)
        predictability = _ema(old.predictability if old else 0.5, 1.0 - unexpectedness, 0.15)
        stability = _ema(old.pattern_stability if old else 0.5, 1.0 - unexpectedness, 0.12)
        reliability = _ema(
            old.expectation_reliability if old else 0.5,
            1.0 - unexpectedness,
            0.15,
        )
        reply_sample = float(any(outcome is OutcomeType.REPLY for outcome in actual_outcomes))
        reply_probability = (
            _ema(old.reply_probability if old else 0.5, reply_sample, 0.15)
            if actual_outcomes
            else (old.reply_probability if old else 0.5)
        )
        average_reply_delay = old.average_reply_delay if old else None
        if reply_delays:
            sample_delay = sum(reply_delays) / len(reply_delays)
            average_reply_delay = (
                sample_delay
                if average_reply_delay is None
                else average_reply_delay * 0.8 + sample_delay * 0.2
            )
        model = EntityModel(
            guild_id=event.guild_id,
            entity_type=entity_type,
            entity_id=entity_id,
            familiarity=familiarity,
            interaction_intensity=intensity,
            predictability=predictability,
            reply_probability=reply_probability,
            average_reply_delay=average_reply_delay,
            pattern_stability=stability,
            expectation_reliability=reliability,
            unfinished_interactions=active_expectations,
            bot_action_response=old.bot_action_response if old else 0.5,
            observations=observations,
            updated_at=event.utc_timestamp,
        )
        await self._database.execute(
            """
            INSERT INTO cognition_entity_models (
                guild_id, entity_type, entity_id, familiarity, interaction_intensity,
                predictability, reply_probability, average_reply_delay, pattern_stability,
                expectation_reliability, unfinished_interactions, bot_action_response,
                observations, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (guild_id, entity_type, entity_id) DO UPDATE SET
                familiarity = excluded.familiarity,
                interaction_intensity = excluded.interaction_intensity,
                predictability = excluded.predictability,
                reply_probability = excluded.reply_probability,
                average_reply_delay = excluded.average_reply_delay,
                pattern_stability = excluded.pattern_stability,
                expectation_reliability = excluded.expectation_reliability,
                unfinished_interactions = excluded.unfinished_interactions,
                bot_action_response = excluded.bot_action_response,
                observations = excluded.observations,
                updated_at = excluded.updated_at
            """,
            (
                model.guild_id,
                model.entity_type.value,
                model.entity_id,
                model.familiarity,
                model.interaction_intensity,
                model.predictability,
                model.reply_probability,
                model.average_reply_delay,
                model.pattern_stability,
                model.expectation_reliability,
                model.unfinished_interactions,
                model.bot_action_response,
                model.observations,
                model.updated_at.isoformat(),
            ),
        )
        self._cache_model(model)
        return model

    async def _subconscious_bias(self, guild_id: int, subject_id: int, event_type: str) -> float:
        row = await self._database.fetchone(
            """
            SELECT value, confidence FROM cognition_subconscious_weights
            WHERE guild_id = ? AND scope_type = 'user' AND scope_id = ?
              AND association_key LIKE ?
            ORDER BY confidence DESC, updated_at DESC LIMIT 1
            """,
            (guild_id, subject_id, f"episode:{event_type}%"),
        )
        return float(row["value"]) * float(row["confidence"]) if row else 0.0

    def _cache_model(self, model: EntityModel) -> None:
        key = (model.guild_id, model.entity_type, model.entity_id)
        self._model_cache[key] = model
        self._model_cache.move_to_end(key)
        while len(self._model_cache) > self._hot_model_cache_size:
            self._model_cache.popitem(last=False)

    async def _episode_from_row(self, row: Any) -> Episode:
        event_rows = await self._database.fetchall(
            """
            SELECT event_id FROM cognition_episode_events
            WHERE episode_id = ? ORDER BY position ASC
            """,
            (str(row["episode_id"]),),
        )
        return Episode(
            episode_id=str(row["episode_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            ended_at=datetime.fromisoformat(str(row["ended_at"])),
            status=EpisodeStatus(str(row["status"])),
            participants=tuple(int(value) for value in json.loads(str(row["participants_json"]))),
            key_event_types=tuple(
                str(value) for value in json.loads(str(row["key_event_types_json"]))
            ),
            observed_outcome=str(row["observed_outcome"]) if row["observed_outcome"] else None,
            unexpectedness=float(row["unexpectedness"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            signature=str(row["signature"]),
            event_ids=tuple(str(item["event_id"]) for item in event_rows),
            consolidated_at=(
                datetime.fromisoformat(str(row["consolidated_at"]))
                if row["consolidated_at"]
                else None
            ),
        )

    @staticmethod
    def _semantic_from_row(row: Any) -> SemanticMemory:
        return SemanticMemory(
            semantic_id=str(row["semantic_id"]),
            guild_id=int(row["guild_id"]),
            scope_type=str(row["scope_type"]),
            scope_id=int(row["scope_id"]),
            feature=str(row["feature"]),
            estimate=float(row["estimate"]),
            confidence=float(row["confidence"]),
            confirmations=int(row["confirmations"]),
            contradictions=int(row["contradictions"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            decay_rate=float(row["decay_rate"]),
            provenance=tuple(json.loads(str(row["provenance_json"]))),
        )

    @staticmethod
    def _entity_from_row(row: Any) -> EntityModel:
        return EntityModel(
            guild_id=int(row["guild_id"]),
            entity_type=EntityType(str(row["entity_type"])),
            entity_id=int(row["entity_id"]),
            familiarity=float(row["familiarity"]),
            interaction_intensity=float(row["interaction_intensity"]),
            predictability=float(row["predictability"]),
            reply_probability=float(row["reply_probability"]),
            average_reply_delay=float(row["average_reply_delay"])
            if row["average_reply_delay"] is not None
            else None,
            pattern_stability=float(row["pattern_stability"]),
            expectation_reliability=float(row["expectation_reliability"]),
            unfinished_interactions=int(row["unfinished_interactions"]),
            bot_action_response=float(row["bot_action_response"]),
            observations=int(row["observations"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _state_from_row(row: Any) -> InternalState:
        return InternalState(
            guild_id=int(row["guild_id"]),
            subject_id=int(row["subject_id"]),
            unexpectedness=float(row["unexpectedness"]),
            uncertainty=float(row["uncertainty"]),
            activation=float(row["activation"]),
            tension=float(row["tension"]),
            interest=float(row["interest"]),
            familiarity=float(row["familiarity"]),
            expected_control=float(row["expected_control"]),
            goal_alignment=float(row["goal_alignment"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    @staticmethod
    def _intent_from_row(row: Any) -> ActionIntent:
        return ActionIntent(
            intent_id=str(row["intent_id"]),
            guild_id=int(row["guild_id"]),
            action_type=ActionType(str(row["action_type"])),
            target_id=int(row["target_id"]) if row["target_id"] is not None else None,
            channel_id=int(row["channel_id"]) if row["channel_id"] is not None else None,
            urgency=float(row["urgency"]),
            confidence=float(row["confidence"]),
            expected_effect=json.loads(str(row["expected_effect_json"])),
            reasoning_trace=tuple(json.loads(str(row["reasoning_trace_json"]))),
            evidence=tuple(json.loads(str(row["evidence_json"]))),
            decision_context=json.loads(str(row["decision_context_json"])),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            expires_at=datetime.fromisoformat(str(row["expires_at"])),
            shadow_only=bool(row["shadow_only"]),
            executed_at=None,
            actual_outcome=str(row["actual_outcome"]) if row["actual_outcome"] else None,
            evaluated_at=datetime.fromisoformat(str(row["evaluated_at"]))
            if row["evaluated_at"]
            else None,
        )

    @staticmethod
    def _event_participants(event: Event) -> list[int]:
        return sorted({value for value in (event.actor_id, event.target_id) if value is not None})

    @staticmethod
    def _episode_signature(event_types: list[str]) -> str:
        return ">".join(event_types[:4])

    @staticmethod
    def _event_intensity(event: Event) -> float:
        weights = {
            EventType.REACTION_ADDED: 0.35,
            EventType.REACTION_REMOVED: 0.25,
            EventType.MESSAGE_CREATED: 0.60,
            EventType.MESSAGE_REPLIED: 0.80,
            EventType.BOT_REPLIED: 0.85,
            EventType.COMMAND_USED: 0.70,
        }
        return weights.get(event.event_type, 0.45)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ema(old: float, sample: float, rate: float) -> float:
    return _clamp(old * (1.0 - rate) + sample * rate)
