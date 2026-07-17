"""Models for autonomous goals and initiative decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class InitiativeState:
    guild_id: int
    channel_id: int
    evidence_balance: float
    observed_human_events: int
    initiatives_started: int
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class AutonomyGoal:
    goal_id: str
    guild_id: int
    channel_id: int
    target_user_id: int | None
    trigger_event_id: str
    intent_id: str
    response_message_id: int
    drive: float
    reasoning: tuple[dict[str, object], ...]
    status: str
    events_waited: int
    created_at: datetime
    resolved_at: datetime | None
    outcome_event_id: str | None


@dataclass(frozen=True, slots=True)
class InitiativeDecision:
    should_initiate: bool
    drive: float
    goal_id: str | None
    reasoning: tuple[dict[str, object], ...]
