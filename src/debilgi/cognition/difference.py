"""Prediction error for categorical expectations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from debilgi.cognition.expectations import Expectation, OutcomeType


@dataclass(frozen=True, slots=True)
class DifferenceVector:
    """Independent, inspectable components of prediction error."""

    scalar: float
    outcome_mismatch: float
    participant_mismatch: float
    timing_mismatch: float
    intensity_mismatch: float
    duration_mismatch: float
    unexpectedness: float


_PARTICIPANT_CLASS = {
    OutcomeType.NO_RESPONSE: "none",
    OutcomeType.SAME_ACTOR_MESSAGE: "same",
    OutcomeType.OTHER_MESSAGE: "other",
    OutcomeType.REPLY: "other",
    OutcomeType.REACTION: "other",
}

_INTENSITY = {
    OutcomeType.NO_RESPONSE: 0.0,
    OutcomeType.REACTION: 0.35,
    OutcomeType.OTHER_MESSAGE: 0.65,
    OutcomeType.SAME_ACTOR_MESSAGE: 0.65,
    OutcomeType.REPLY: 1.0,
}


def categorical_difference(distribution: Mapping[OutcomeType, float], actual: OutcomeType) -> float:
    probability = float(distribution.get(actual, 0.0))
    if not 0.0 <= probability <= 1.0:
        raise ValueError("outcome probability must be between zero and one")
    return 1.0 - probability


def difference_vector(expectation: Expectation) -> DifferenceVector:
    """Expand the original scalar d without pretending unavailable signals exist."""

    actual = expectation.actual_outcome
    resolved_at = expectation.resolved_at
    if actual is None or expectation.difference is None or resolved_at is None:
        raise ValueError("expectation must be resolved before calculating its difference vector")

    predicted = expectation.prediction.outcome
    elapsed = max(0.0, (resolved_at - expectation.created_at).total_seconds())
    window = max(0.001, expectation.observation_window_seconds)
    normalized_time = min(1.0, elapsed / window)
    predicted_time = 1.0 if predicted is OutcomeType.NO_RESPONSE else 0.5
    timing = abs(normalized_time - predicted_time)
    outcome = 0.0 if predicted is actual else 1.0
    participant = float(_PARTICIPANT_CLASS[predicted] != _PARTICIPANT_CLASS[actual])
    intensity = abs(_INTENSITY[predicted] - _INTENSITY[actual])
    duration = timing if actual is not OutcomeType.NO_RESPONSE else max(0.0, 1.0 - normalized_time)
    scalar = max(0.0, min(1.0, expectation.difference))
    return DifferenceVector(
        scalar=scalar,
        outcome_mismatch=outcome,
        participant_mismatch=participant,
        timing_mismatch=timing,
        intensity_mismatch=intensity,
        duration_mismatch=duration,
        unexpectedness=scalar,
    )
