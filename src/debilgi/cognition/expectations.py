"""Expectation models and similarity-weighted categorical prediction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class OutcomeType(StrEnum):
    NO_RESPONSE = "no_response"
    OTHER_MESSAGE = "other_message"
    SAME_ACTOR_MESSAGE = "same_actor_message"
    REPLY = "reply"
    REACTION = "reaction"


OUTCOMES = tuple(OutcomeType)


class ExpectationStatus(StrEnum):
    PENDING = "pending"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class Situation:
    trigger_type: str
    author_is_bot: bool
    is_reply: bool
    has_mentions: bool
    channel_activity: str
    actor_charge: str
    message_length: str
    actor_message_rate: str
    actor_uppercase_ratio: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> Situation:
        data: dict[str, Any] = json.loads(value)
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PatternEvidence:
    situation: Situation
    outcome: OutcomeType
    observations: int


@dataclass(frozen=True, slots=True)
class Prediction:
    outcome: OutcomeType
    probability: float
    distribution: dict[OutcomeType, float]
    evidence_support: float


@dataclass(frozen=True, slots=True)
class Expectation:
    expectation_id: str
    guild_id: int
    channel_id: int
    actor_id: int | None
    trigger_event_id: str
    trigger_message_id: int | None
    situation: Situation
    prediction: Prediction
    created_at: datetime
    expires_at: datetime
    status: ExpectationStatus
    actual_outcome: OutcomeType | None = None
    difference: float | None = None
    resolved_by_event_id: str | None = None
    resolved_at: datetime | None = None
    source: str = "message_observation"
    observation_window_seconds: float = 120.0
    allowed_outcomes: tuple[OutcomeType, ...] = OUTCOMES


class ExpectationPredictor:
    _SIMILARITY_WEIGHTS = {
        "trigger_type": 3.0,
        "author_is_bot": 2.0,
        "is_reply": 1.0,
        "has_mentions": 1.0,
        "channel_activity": 1.0,
        "actor_charge": 1.0,
        "message_length": 1.0,
        "actor_message_rate": 1.0,
        "actor_uppercase_ratio": 1.0,
    }

    def __init__(self, *, minimum_similarity: float = 0.55, prior: float = 1.0) -> None:
        if not 0 <= minimum_similarity <= 1 or prior <= 0:
            raise ValueError("predictor parameters are outside their valid range")
        self.minimum_similarity = minimum_similarity
        self.prior = prior

    def predict(self, situation: Situation, evidence: list[PatternEvidence]) -> Prediction:
        scores = {outcome: self.prior for outcome in OUTCOMES}
        support = 0.0
        for item in evidence:
            similarity = self.similarity(situation, item.situation)
            if similarity < self.minimum_similarity:
                continue
            contribution = similarity * item.observations
            scores[item.outcome] += contribution
            support += contribution

        total = sum(scores.values())
        distribution = {outcome: score / total for outcome, score in scores.items()}
        predicted = max(OUTCOMES, key=distribution.__getitem__)
        return Prediction(
            outcome=predicted,
            probability=distribution[predicted],
            distribution=distribution,
            evidence_support=support,
        )

    def similarity(self, left: Situation, right: Situation) -> float:
        if left.trigger_type != right.trigger_type:
            return 0.0
        left_values = asdict(left)
        right_values = asdict(right)
        matched = sum(
            weight
            for field, weight in self._SIMILARITY_WEIGHTS.items()
            if left_values[field] == right_values[field]
        )
        return matched / sum(self._SIMILARITY_WEIGHTS.values())
