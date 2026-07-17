"""Persistent models produced by Cognitive Core v0.2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class EpisodeStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class Episode:
    episode_id: str
    guild_id: int
    channel_id: int
    started_at: datetime
    ended_at: datetime
    status: EpisodeStatus
    participants: tuple[int, ...]
    key_event_types: tuple[str, ...]
    observed_outcome: str | None
    unexpectedness: float
    importance: float
    confidence: float
    signature: str
    event_ids: tuple[str, ...] = ()
    consolidated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SemanticMemory:
    semantic_id: str
    guild_id: int
    scope_type: str
    scope_id: int
    feature: str
    estimate: float
    confidence: float
    confirmations: int
    contradictions: int
    updated_at: datetime
    decay_rate: float
    provenance: tuple[str, ...]


class EntityType(StrEnum):
    SERVER = "server"
    CHANNEL = "channel"
    USER = "user"
    SELF = "self"


@dataclass(frozen=True, slots=True)
class EntityModel:
    guild_id: int
    entity_type: EntityType
    entity_id: int
    familiarity: float
    interaction_intensity: float
    predictability: float
    reply_probability: float
    average_reply_delay: float | None
    pattern_stability: float
    expectation_reliability: float
    unfinished_interactions: int
    bot_action_response: float
    observations: int
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class InternalState:
    guild_id: int
    subject_id: int
    unexpectedness: float
    uncertainty: float
    activation: float
    tension: float
    interest: float
    familiarity: float
    expected_control: float
    goal_alignment: float
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class Assessment:
    assessment_id: str
    guild_id: int
    object_type: str
    object_id: int
    dimensions: dict[str, float]
    confidence: float
    supporting_episode_ids: tuple[str, ...]
    contradicting_episode_ids: tuple[str, ...]
    alternative_explanations: tuple[str, ...]
    data_gap: float
    next_review_at: datetime
    created_at: datetime


class ActionType(StrEnum):
    STAY_SILENT = "stay_silent"
    ADD_REACTION = "add_reaction"
    SHORT_REPLY = "short_reply"
    ASK_QUESTION = "ask_question"
    CONTINUE_TOPIC = "continue_topic"
    SAVE_OBSERVATION = "save_observation"
    CHECK_HYPOTHESIS_LATER = "check_hypothesis_later"
    REWARD_BRAINS = "reward_brains"
    REMOVE_BRAINS = "remove_brains"
    GIVE_ITEM = "give_item"


@dataclass(frozen=True, slots=True)
class ActionIntent:
    intent_id: str
    guild_id: int
    action_type: ActionType
    target_id: int | None
    channel_id: int | None
    urgency: float
    confidence: float
    expected_effect: dict[str, Any]
    reasoning_trace: tuple[dict[str, Any], ...]
    evidence: tuple[str, ...]
    decision_context: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    shadow_only: bool = True
    executed_at: datetime | None = None
    actual_outcome: str | None = None
    evaluated_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.shadow_only or self.executed_at is not None:
            raise ValueError("Cognitive Core v0.2 intents must remain unexecuted shadow records")
