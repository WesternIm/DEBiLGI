"""Models for private post-outcome reflection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Reflection:
    reflection_id: str
    experience_id: str
    guild_id: int
    user_id: int | None
    outcome_type: str
    expected_reward: float
    actual_reward: float
    prediction_error: float
    surprise: float
    lesson_kind: str
    strategies: tuple[str, ...]
    evidence: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StrategyBelief:
    guild_id: int
    strategy: str
    observations: int
    mean_reward: float
    mean_surprise: float
    confidence: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ReflectionGuidance:
    length_scale: float = 1.0
    exploration: float = 0.0
    caution: float = 0.0
    confidence: float = 0.0
    strongest_strategy: str | None = None


@dataclass(frozen=True, slots=True)
class ReflectionStats:
    reflections: int
    revisions: int
    average_surprise: float
    strategy_beliefs: int
