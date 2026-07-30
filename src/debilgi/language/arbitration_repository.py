"""SQLite persistence and bounded feedback updates for response arbitration."""

from __future__ import annotations

import json
from datetime import datetime
from types import MappingProxyType

from debilgi.events.models import Event
from debilgi.experience.feedback import DEFAULT_FEEDBACK_ATTRIBUTOR
from debilgi.language.arbitration import (
    DEFAULT_SOURCE_BIASES,
    DEFAULT_WEIGHTS,
    ArbitrationDecision,
    ArbitrationProfile,
)
from debilgi.language.structural_learning import structural_response_features
from debilgi.storage.database import Database

_PENALTY_FEATURES = {
    "prompt_echo_penalty",
    "reference_copy_penalty",
    "retrieval_dump_penalty",
    "repetition_penalty",
    "broken_surface_penalty",
}


class ArbitrationRepository:
    def __init__(
        self,
        database: Database,
        *,
        learning_enabled: bool = False,
        learning_rate: float = 0.03,
        minimum_feedback_observations: int = 20,
        maximum_weight_delta: float = 0.05,
    ) -> None:
        self._database = database
        self.learning_enabled = learning_enabled
        self.learning_rate = max(0.001, min(0.20, learning_rate))
        self.minimum_feedback_observations = max(1, minimum_feedback_observations)
        self.maximum_weight_delta = max(0.001, min(0.25, maximum_weight_delta))

    async def profile(self, guild_id: int) -> ArbitrationProfile:
        row = await self._database.fetchone(
            """SELECT version.*
               FROM language_arbitration_profile_pointers AS pointer
               JOIN language_arbitration_profile_versions AS version
                 ON version.guild_id = pointer.guild_id
                AND version.stage = pointer.stage
                AND version.version = pointer.active_version
               WHERE pointer.guild_id = ? AND pointer.stage = 'surface'""",
            (guild_id,),
        )
        if row is None:
            row = await self._database.fetchone(
                """SELECT * FROM language_arbitration_weight_profiles
                   WHERE guild_id = ? AND stage = 'surface'""",
                (guild_id,),
            )
        if row is None:
            return ArbitrationProfile.baseline(guild_id)
        return ArbitrationProfile(
            profile_id=str(row["profile_id"]),
            guild_id=guild_id,
            weights=MappingProxyType(
                {key: float(value) for key, value in json.loads(str(row["weights_json"])).items()}
            ),
            source_biases=MappingProxyType(
                {
                    key: float(value)
                    for key, value in json.loads(str(row["source_biases_json"])).items()
                }
            ),
            observations=int(row["observations"]),
            confidence=float(row["confidence"]),
            version=int(row["version"]),
        )

    async def activate_profile_version(self, guild_id: int, version: int) -> bool:
        """Atomically roll surface arbitration back to an immutable snapshot."""

        row = await self._database.fetchone(
            """SELECT 1 FROM language_arbitration_profile_versions
               WHERE guild_id = ? AND stage = 'surface' AND version = ?""",
            (guild_id, version),
        )
        if row is None:
            return False
        await self._database.execute(
            """INSERT INTO language_arbitration_profile_pointers (
                   guild_id, stage, active_version, updated_at
               ) VALUES (?, 'surface', ?, CURRENT_TIMESTAMP)
               ON CONFLICT (guild_id, stage) DO UPDATE SET
                   active_version = excluded.active_version,
                   updated_at = excluded.updated_at""",
            (guild_id, version),
        )
        return True

    async def structural_weights(self, guild_id: int) -> dict[str, float]:
        """Load sparse learned form weights for the current community."""

        rows = await self._database.fetchall(
            """SELECT feature_key, weight
               FROM language_structural_feature_weights
               WHERE guild_id = ? AND observation_count > 0""",
            (guild_id,),
        )
        return {str(row["feature_key"]): float(row["weight"]) for row in rows}

    async def save_decision(
        self,
        event: Event,
        decision: ArbitrationDecision,
        *,
        mode: str,
        prompt: str | None = None,
        legacy_text: str | None = None,
        legacy_source: str | None = None,
    ) -> None:
        now = event.utc_timestamp.isoformat()
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR REPLACE INTO language_arbitration_decisions (
                       decision_id, event_id, guild_id, channel_id, user_id, mode,
                       winner_candidate_id, minimum_score, weight_profile_id, weight_version,
                       candidate_count, eligible_count, used_fallback, selection_reason,
                       latency_ms, legacy_text_hash, legacy_source, shadow_disagreed,
                       created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.decision_id,
                    event.event_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    mode,
                    decision.winner.candidate_id if decision.winner else None,
                    decision.minimum_score,
                    decision.profile_id,
                    decision.weight_version,
                    len(decision.scores),
                    sum(item.eligible for item in decision.scores),
                    int(decision.used_fallback),
                    decision.selection_reason,
                    decision.latency_ms,
                    _digest(legacy_text) if legacy_text is not None else None,
                    legacy_source,
                    int(
                        legacy_text is not None
                        and decision.winner is not None
                        and " ".join(legacy_text.casefold().split())
                        != " ".join(decision.winner.text.casefold().split())
                    ),
                    now,
                ),
            )
            await connection.execute(
                "DELETE FROM language_response_candidates WHERE decision_id = ?",
                (decision.decision_id,),
            )
            await connection.execute(
                "DELETE FROM language_content_proposals WHERE decision_id = ?",
                (decision.decision_id,),
            )
            for scored in decision.content_scores:
                proposal = scored.proposal
                await connection.execute(
                    """INSERT INTO language_content_proposals (
                           proposal_id, decision_id, source, speech_act, response_goal,
                           plan_json, source_confidence, applicability, content_score,
                           eligible, rejection_reasons_json, selected_for_surface,
                           created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        proposal.proposal_id,
                        decision.decision_id,
                        proposal.source,
                        proposal.plan.speech_act,
                        proposal.plan.response_goal or proposal.plan.response_act,
                        _json(
                            {
                                "target_id": proposal.plan.target_id,
                                "topic_ids": proposal.plan.topic_ids,
                                "key_points": proposal.plan.key_points,
                                "required_facts": tuple(
                                    fact.as_dict()
                                    for fact in proposal.plan.required_facts
                                ),
                                "optional_facts": tuple(
                                    fact.as_dict()
                                    for fact in proposal.plan.optional_facts
                                ),
                                "unknowns": proposal.plan.unknowns,
                                "tone": proposal.plan.tone,
                                "certainty": proposal.plan.confidence,
                                "follow_up": proposal.plan.follow_up,
                                "components": dict(scored.components),
                            }
                        ),
                        proposal.source_confidence,
                        proposal.applicability,
                        scored.total,
                        int(scored.eligible),
                        _json(scored.rejection_reasons),
                        int(
                            proposal.proposal_id
                            in decision.selected_proposal_ids
                        ),
                        now,
                    ),
                )
            for order, scored in enumerate(decision.scores):
                candidate = scored.candidate
                await connection.execute(
                    """INSERT INTO language_response_candidates (
                           candidate_id, decision_id, source, renderer, speech_act, text,
                           text_hash, required_facts_json, features_json,
                           weighted_components_json, source_confidence, total_score,
                           eligible, rejection_reasons_json, selected, generation_order,
                           created_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        candidate.candidate_id,
                        decision.decision_id,
                        candidate.source,
                        candidate.renderer,
                        candidate.speech_act,
                        candidate.text[:4000],
                        _digest(candidate.text),
                        _json(tuple(fact.as_dict() for fact in candidate.required_facts)),
                        _json(scored.features.as_dict()),
                        _json(dict(scored.weighted_components)),
                        candidate.source_confidence,
                        scored.total,
                        int(scored.eligible),
                        _json(scored.rejection_reasons),
                        int(
                            decision.winner is not None
                            and candidate.candidate_id == decision.winner.candidate_id
                        ),
                        order,
                        now,
                    ),
                )
                await connection.execute(
                    """INSERT OR REPLACE INTO language_surface_proposals (
                           candidate_id, decision_id, proposal_id
                       ) VALUES (?, ?, ?)""",
                    (
                        candidate.candidate_id,
                        decision.decision_id,
                        candidate.proposal.proposal_id,
                    ),
                )
                construction_id = candidate.metadata.get("construction_id")
                if isinstance(construction_id, str) and construction_id:
                    await connection.execute(
                        """INSERT OR REPLACE INTO language_candidate_constructions (
                               candidate_id, guild_id, construction_id
                           ) VALUES (?, ?, ?)""",
                        (candidate.candidate_id, event.guild_id, construction_id),
                    )
                structural_prompt = prompt or event.payload.get("content")
                if isinstance(structural_prompt, str):
                    structural_features = structural_response_features(
                        prompt=structural_prompt,
                        candidate_text=candidate.text,
                        source=candidate.source,
                        speech_act=candidate.speech_act,
                        metadata=candidate.metadata,
                    )
                    await connection.execute(
                        """INSERT OR REPLACE INTO language_candidate_structural_features (
                               candidate_id, guild_id, features_json, created_at
                           ) VALUES (?, ?, ?, ?)""",
                        (
                            candidate.candidate_id,
                            event.guild_id,
                            _json(structural_features),
                            now,
                        ),
                    )

    async def mark_sent(
        self, decision_id: str, *, response_message_id: int, sent_at: datetime
    ) -> None:
        await self._database.execute(
            """UPDATE language_arbitration_decisions
               SET response_message_id = ?, sent_at = ? WHERE decision_id = ?""",
            (response_message_id, sent_at.isoformat(), decision_id),
        )

    async def observe_feedback(self, event: Event) -> None:
        feedback = DEFAULT_FEEDBACK_ATTRIBUTOR.attribute(event)
        target = feedback.target_message_id
        if target is None:
            return
        row = await self._database.fetchone(
            """SELECT * FROM language_arbitration_decisions
               WHERE guild_id = ? AND response_message_id = ?""",
            (event.guild_id, target),
        )
        if row is None:
            return
        if not feedback.is_learnable():
            return
        reward = feedback.reward
        kind = feedback.kind
        attribution = feedback.attribution_confidence
        decision_id = str(row["decision_id"])
        old_version = int(row["weight_version"])
        inserted = await self._database.execute(
            """INSERT OR IGNORE INTO language_arbitration_feedback (
                   feedback_event_id, decision_id, feedback_kind, reward,
                   attribution_confidence, old_version, observed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                decision_id,
                kind,
                reward,
                attribution,
                old_version,
                event.utc_timestamp.isoformat(),
            ),
        )
        if inserted.rowcount != 1:
            return
        await self._apply_construction_feedback(
            guild_id=event.guild_id,
            decision_id=decision_id,
            reward=reward,
            attribution=attribution,
            observed_at=event.utc_timestamp,
        )
        if not self.learning_enabled:
            return
        await self._apply_structural_feedback(
            guild_id=event.guild_id,
            decision_id=decision_id,
            reward=reward,
            attribution=attribution,
            observed_at=event.utc_timestamp,
        )
        await self._apply_feedback(
            guild_id=event.guild_id,
            decision_id=decision_id,
            feedback_event_id=event.event_id,
            reward=reward,
            attribution=attribution,
            observed_at=event.utc_timestamp,
        )

    async def _apply_structural_feedback(
        self,
        *,
        guild_id: int,
        decision_id: str,
        reward: float,
        attribution: float,
        observed_at: datetime,
    ) -> None:
        """Pairwise online update over abstract prompt/response structures."""

        rows = await self._database.fetchall(
            """SELECT candidate.selected, candidate.total_score, learned.features_json
               FROM language_response_candidates AS candidate
               JOIN language_candidate_structural_features AS learned
                 ON learned.candidate_id = candidate.candidate_id
               WHERE candidate.decision_id = ? AND candidate.eligible = 1
               ORDER BY candidate.total_score DESC""",
            (decision_id,),
        )
        winner = next((row for row in rows if bool(row["selected"])), None)
        alternatives = tuple(row for row in rows if not bool(row["selected"]))[:4]
        if winner is None:
            return
        winner_features = {
            key: float(value)
            for key, value in json.loads(str(winner["features_json"])).items()
            if value is not None
        }
        alternative_features = [
            {
                key: float(value)
                for key, value in json.loads(str(row["features_json"])).items()
                if value is not None
            }
            for row in alternatives
        ]
        keys = set(winner_features)
        for item in alternative_features:
            keys.update(item)
        positive = int(reward > 0.0)
        negative = int(reward < 0.0)
        now = observed_at.isoformat()
        for key in keys:
            winner_value = winner_features.get(key, 0.0)
            alternative_value = (
                sum(item.get(key, 0.0) for item in alternative_features)
                / len(alternative_features)
                if alternative_features
                else 0.0
            )
            # With no alternative, negative feedback can still teach the model
            # which structural traits to avoid. Positive singleton feedback is
            # also useful, but deliberately smaller to prevent rapid overfit.
            direction = winner_value - alternative_value
            if not alternatives:
                if key.startswith("prompt:") or key.startswith("speech_act:"):
                    continue
                direction = winner_value * (0.65 if reward > 0.0 else 1.0)
            if direction == 0.0:
                continue
            delta = max(
                -0.06,
                min(0.06, self.learning_rate * reward * attribution * direction),
            )
            await self._database.execute(
                """INSERT INTO language_structural_feature_weights (
                       guild_id, feature_key, weight, observation_count,
                       positive_feedback, negative_feedback, updated_at
                   ) VALUES (?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT (guild_id, feature_key) DO UPDATE SET
                       weight = MAX(
                           -0.75,
                           MIN(0.75, language_structural_feature_weights.weight * 0.997 + ?)
                       ),
                       observation_count = observation_count + 1,
                       positive_feedback = positive_feedback + ?,
                       negative_feedback = negative_feedback + ?,
                       updated_at = excluded.updated_at""",
                (
                    guild_id,
                    key,
                    delta,
                    positive,
                    negative,
                    now,
                    delta,
                    positive,
                    negative,
                ),
            )

    async def _apply_construction_feedback(
        self,
        *,
        guild_id: int,
        decision_id: str,
        reward: float,
        attribution: float,
        observed_at: datetime,
    ) -> None:
        """Reinforce the abstract construction, not the exact answer text."""

        row = await self._database.fetchone(
            """SELECT learned.construction_id
               FROM language_response_candidates AS candidate
               JOIN language_candidate_constructions AS learned
                 ON learned.candidate_id = candidate.candidate_id
               WHERE candidate.decision_id = ? AND candidate.selected = 1""",
            (decision_id,),
        )
        if row is None:
            return
        construction_id = str(row["construction_id"])
        delta = max(-0.12, min(0.12, 0.10 * reward * attribution))
        positive = int(reward > 0.0)
        negative = int(reward < 0.0)
        await self._database.execute(
            """UPDATE language_productive_constructions
               SET positive_feedback = positive_feedback + ?,
                   negative_feedback = negative_feedback + ?,
                   utility = MAX(-1.0, MIN(1.0, utility + ?)),
                   confidence = MAX(
                       0.20,
                       MIN(0.99, confidence + CASE WHEN ? > 0 THEN 0.01 ELSE -0.02 END)
                   ),
                   updated_at = ?
               WHERE guild_id = ? AND construction_id = ?""",
            (
                positive,
                negative,
                delta,
                reward,
                observed_at.isoformat(),
                guild_id,
                construction_id,
            ),
        )

    async def _apply_feedback(
        self,
        *,
        guild_id: int,
        decision_id: str,
        feedback_event_id: str,
        reward: float,
        attribution: float,
        observed_at: datetime,
    ) -> None:
        rows = await self._database.fetchall(
            """SELECT source, features_json, selected, total_score
               FROM language_response_candidates
               WHERE decision_id = ? AND eligible = 1
               ORDER BY total_score DESC""",
            (decision_id,),
        )
        winner = next((row for row in rows if bool(row["selected"])), None)
        alternatives = tuple(row for row in rows if not bool(row["selected"]))[:4]
        if winner is None or not alternatives:
            return
        profile = await self.profile(guild_id)
        if profile.observations + 1 < self.minimum_feedback_observations:
            new_version = await self._store_profile(
                profile,
                weights=dict(profile.weights),
                source_biases=dict(profile.source_biases),
                observations=profile.observations + 1,
                observed_at=observed_at,
            )
            await self._database.execute(
                """UPDATE language_arbitration_feedback
                   SET new_version = ?, applied_at = ?
                   WHERE feedback_event_id = ?""",
                (new_version, observed_at.isoformat(), feedback_event_id),
            )
            return
        winner_features = {
            key: float(value)
            for key, value in json.loads(str(winner["features_json"])).items()
            if value is not None
        }
        alternative_features = [
            {
                key: float(value)
                for key, value in json.loads(str(row["features_json"])).items()
                if value is not None
            }
            for row in alternatives
        ]
        weights = dict(profile.weights)
        deltas: dict[str, float] = {}
        for name, current in weights.items():
            winner_value = winner_features.get(name)
            measured_alternatives = [
                item[name] for item in alternative_features if name in item
            ]
            if winner_value is None or not measured_alternatives:
                continue
            alternative = sum(measured_alternatives) / len(measured_alternatives)
            if name in _PENALTY_FEATURES:
                winner_value = -winner_value
                alternative = -alternative
            direction = winner_value - alternative
            raw_delta = self.learning_rate * reward * attribution * direction
            delta = max(-self.maximum_weight_delta, min(self.maximum_weight_delta, raw_delta))
            weights[name] = max(0.05, min(3.0, current + delta))
            if delta:
                deltas[name] = delta
        biases = dict(profile.source_biases)
        winner_source = str(winner["source"])
        bias_delta = max(
            -self.maximum_weight_delta,
            min(self.maximum_weight_delta, self.learning_rate * reward * attribution),
        )
        biases[winner_source] = max(-1.0, min(1.0, biases.get(winner_source, 0.0) + bias_delta))
        deltas[f"source:{winner_source}"] = bias_delta
        new_version = await self._store_profile(
            profile,
            weights=weights,
            source_biases=biases,
            observations=profile.observations + 1,
            observed_at=observed_at,
        )
        await self._database.execute(
            """UPDATE language_arbitration_feedback
               SET new_version = ?, weight_delta_json = ?, applied_at = ?
               WHERE feedback_event_id = ?""",
            (new_version, _json(deltas), observed_at.isoformat(), feedback_event_id),
        )

    async def _store_profile(
        self,
        profile: ArbitrationProfile,
        *,
        weights: dict[str, float],
        source_biases: dict[str, float],
        observations: int,
        observed_at: datetime,
        version: int | None = None,
    ) -> int:
        latest = await self._database.fetchone(
            """SELECT COALESCE(MAX(version), 0) AS version
               FROM language_arbitration_profile_versions
               WHERE guild_id = ? AND stage = 'surface'""",
            (profile.guild_id,),
        )
        latest_version = int(latest["version"]) if latest is not None else 0
        stored_version = max(version or 0, profile.version + 1, latest_version + 1)
        confidence = observations / (observations + 20.0)
        now = observed_at.isoformat()
        baseline = _json(
            {
                "weights": dict(DEFAULT_WEIGHTS),
                "source_biases": dict(DEFAULT_SOURCE_BIASES),
            }
        )
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT OR IGNORE INTO language_arbitration_profile_versions (
                       guild_id, stage, version, profile_id, weights_json,
                       source_biases_json, baseline_json, observations, confidence,
                       created_at
                   ) VALUES (?, 'surface', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile.guild_id,
                    profile.version,
                    profile.profile_id,
                    _json(dict(profile.weights)),
                    _json(dict(profile.source_biases)),
                    baseline,
                    profile.observations,
                    profile.confidence,
                    now,
                ),
            )
            await connection.execute(
                """INSERT INTO language_arbitration_profile_versions (
                       guild_id, stage, version, profile_id, weights_json,
                       source_biases_json, baseline_json, observations, confidence,
                       created_at
                   ) VALUES (?, 'surface', ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile.guild_id,
                    stored_version,
                    profile.profile_id,
                    _json(weights),
                    _json(source_biases),
                    baseline,
                    observations,
                    confidence,
                    now,
                ),
            )
            await connection.execute(
                """INSERT INTO language_arbitration_profile_pointers (
                       guild_id, stage, active_version, updated_at
                   ) VALUES (?, 'surface', ?, ?)
                   ON CONFLICT (guild_id, stage) DO UPDATE SET
                       active_version = excluded.active_version,
                       updated_at = excluded.updated_at""",
                (profile.guild_id, stored_version, now),
            )
        return stored_version


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()
