"""Models for primitive and dynamically blended social reactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class PrimitiveVector:
    agreement: float
    resistance: float
    approval: float
    curiosity: float
    aversion: float
    trust: float
    arousal: float

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"reaction primitive {name} must be between 0 and 1")

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}

    def ranked(self) -> tuple[tuple[str, float], ...]:
        return tuple(sorted(self.as_dict().items(), key=lambda item: (-item[1], item[0])))

    def blended(self, other: PrimitiveVector, other_weight: float) -> PrimitiveVector:
        weight = max(0.0, min(1.0, other_weight))
        left = self.as_dict()
        right = other.as_dict()
        return PrimitiveVector(
            **{name: _clamp(left[name] * (1.0 - weight) + right[name] * weight) for name in left}
        )


@dataclass(frozen=True, slots=True)
class SocialReaction:
    reaction_id: str
    event_id: str
    guild_id: int
    actor_id: int | None
    vector: PrimitiveVector
    primary_component: str
    secondary_component: str | None
    blend_id: str
    confidence: float
    signals: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class RelationState:
    guild_id: int
    actor_id: int
    vector: PrimitiveVector
    observations: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ReactionStats:
    reactions: int
    composite_reactions: int
    relation_models: int


def composition(vector: PrimitiveVector) -> tuple[str, str | None, str]:
    ranked = vector.ranked()
    primary, primary_value = ranked[0]
    secondary, secondary_value = ranked[1]
    include_secondary = secondary_value >= 0.36 and primary_value - secondary_value <= 0.30
    secondary_component = secondary if include_secondary else None
    blend_id = f"{primary}+{secondary}" if secondary_component else primary
    return primary, secondary_component, blend_id


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
