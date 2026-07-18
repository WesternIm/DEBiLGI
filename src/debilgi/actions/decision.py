"""Deterministic shadow decision scoring, independent from phrase generation."""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

from debilgi.cognition.models import (
    ActionIntent,
    ActionType,
    EntityModel,
    InternalState,
)
from debilgi.events.models import Event
from debilgi.personality.genome import DEBILGI_GENOME, PersonalityGenome


class ShadowDecisionEngine:
    """Score inspectable intentions; never dispatch anything to Discord."""

    def __init__(self, genome: PersonalityGenome = DEBILGI_GENOME) -> None:
        self.genome = genome

    def decide(
        self,
        event: Event,
        model: EntityModel,
        state: InternalState,
        *,
        active_expectations: int,
        evidence: tuple[str, ...] = (),
    ) -> ActionIntent:
        scores = {
            ActionType.STAY_SILENT: (
                0.25
                + (1.0 - model.familiarity) * 0.20
                + (1.0 - self.genome.sociability) * 0.10
                + self.genome.response_threshold * 0.08
            ),
            ActionType.ADD_REACTION: (
                0.12
                + model.familiarity * 0.16
                + model.predictability * 0.10
                + self.genome.expressiveness * 0.08
            ),
            ActionType.SHORT_REPLY: (
                0.08
                + model.familiarity * 0.34
                + state.interest * 0.16
                + self.genome.sociability * 0.14
                + self.genome.impulsivity * 0.06
            ),
            ActionType.ASK_QUESTION: (
                0.08
                + state.tension * 0.38
                + state.uncertainty * 0.30
                + self.genome.curiosity * 0.14
                + self.genome.suspicion * 0.05
            ),
            ActionType.CONTINUE_TOPIC: (
                0.05
                + min(0.30, model.unfinished_interactions * 0.10)
                + model.familiarity * 0.14
                + self.genome.memory_attachment * 0.10
                + self.genome.verbosity * 0.05
            ),
            ActionType.SAVE_OBSERVATION: (
                0.18
                + state.unexpectedness * 0.30
                + state.uncertainty * 0.16
                + self.genome.memory_attachment * 0.10
            ),
            ActionType.CHECK_HYPOTHESIS_LATER: (
                0.10
                + state.uncertainty * 0.34
                + state.unexpectedness * 0.24
                + self.genome.patience * 0.08
                + self.genome.suspicion * 0.08
            ),
        }
        if event.payload.get("bot_mentioned") is True and not event.payload.get(
            "author_is_bot", False
        ):
            scores[ActionType.SHORT_REPLY] += 0.45 + self.genome.sociability * 0.20
            scores[ActionType.ASK_QUESTION] += self.genome.curiosity * 0.12
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        selected, top_score = ordered[0]
        runner_up_score = ordered[1][1]
        margin = max(0.0, top_score - runner_up_score)
        confidence = _clamp(0.30 + margin + min(0.30, model.observations / 50.0))
        urgency = _clamp(max(state.tension, state.activation) * 0.65 + margin)
        created_at = event.utc_timestamp
        trace = (
            {
                "stage": "observation",
                "event_id": event.event_id,
                "event_type": event.event_type.value,
            },
            {
                "stage": "genome",
                "genome_id": self.genome.genome_id,
                "version": self.genome.version,
                "signature": self.genome.signature,
            },
            {
                "stage": "expectation_context",
                "active_expectations": active_expectations,
                "expected_control": round(state.expected_control, 6),
            },
            {
                "stage": "mismatch_state",
                "unexpectedness": round(state.unexpectedness, 6),
                "uncertainty": round(state.uncertainty, 6),
                "tension": round(state.tension, 6),
            },
            {
                "stage": "decision",
                "selected": selected.value,
                "scores": {action.value: round(score, 6) for action, score in ordered},
            },
        )
        return ActionIntent(
            intent_id=f"intent:{uuid4()}",
            guild_id=event.guild_id,
            action_type=selected,
            target_id=event.actor_id,
            channel_id=event.channel_id,
            urgency=urgency,
            confidence=confidence,
            expected_effect={
                "hypothesis": "improve future prediction evidence",
                "active_expectations": active_expectations,
            },
            reasoning_trace=trace,
            evidence=evidence,
            decision_context={
                "model_observations": model.observations,
                "familiarity": model.familiarity,
                "predictability": model.predictability,
                "genome_signature": self.genome.signature,
                "state_unexpectedness": state.unexpectedness,
                "state_uncertainty": state.uncertainty,
                "state_activation": state.activation,
                "state_tension": state.tension,
                "state_interest": state.interest,
                "state_updated_at": state.updated_at.isoformat(),
            },
            created_at=created_at,
            expires_at=created_at + timedelta(minutes=10),
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
