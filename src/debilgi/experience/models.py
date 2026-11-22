"""Persistent experience records and prepared utterance metadata."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from debilgi.language.semantic_reply_plan import ProcessingOutcome


@dataclass(frozen=True, slots=True)
class PreparedReply:
    text: str
    intent_id: str
    tokens: tuple[str, ...]
    genome_signature: str
    candidate_count: int
    initiative: bool = False
    goal_id: str | None = None
    initiative_drive: float = 0.0
    initiative_reasoning: tuple[dict[str, object], ...] = ()
    fallback: bool = False
    pending_concept: str | None = None
    contextual_learning: bool = False
    response_source: str = "graph"
    arbitration_decision_id: str | None = None
    winner_candidate_id: str | None = None
    arbitration_weight_version: int = 0
    semantic_reply_plan_id: str | None = None
    processing_outcome: ProcessingOutcome | None = None
    assertion_experience: dict[str, object] | None = None
    meaning_intention_id: str | None = None
    dialogue_task_id: str | None = None
    dialogue_task_progress: float | None = None
    deliberation_steps: int = 0


@dataclass(frozen=True, slots=True)
class UtteranceExperience:
    experience_id: str
    guild_id: int
    channel_id: int
    user_id: int | None
    prompt_event_id: str
    intent_id: str
    response_message_id: int
    response_tokens: tuple[str, ...]
    genome_signature: str
    candidate_count: int
    status: str
    outcome_type: str | None
    reward: float | None
    created_at: datetime
    resolved_at: datetime | None
    initiative: bool = False
    goal_id: str | None = None
    expected_reward: float = 0.0


@dataclass(frozen=True, slots=True)
class SelfModel:
    guild_id: int
    utterances_observed: int
    positive_outcomes: int
    negative_outcomes: int
    average_reward: float
    initiative_attempts: int
    initiative_successes: int
    efficacy: float
    updated_at: datetime | None
    # Canonical evidence adapter; efficacy remains an autonomy statistic.
    claim_ids: tuple[str, ...] = ()
