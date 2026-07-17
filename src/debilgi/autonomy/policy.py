"""Expected-information initiative policy with no time-based cooldown."""

from __future__ import annotations

import hashlib
from uuid import uuid4

from debilgi.autonomy.models import InitiativeDecision
from debilgi.autonomy.repository import AutonomyRepository
from debilgi.cognition.models import ActionIntent
from debilgi.events.models import Event, EventType
from debilgi.experience.repository import ExperienceRepository
from debilgi.language.tokenizer import tokenize
from debilgi.personality.genome import PersonalityGenome


class InitiativePolicy:
    def __init__(
        self,
        genome: PersonalityGenome,
        autonomy: AutonomyRepository,
        experience: ExperienceRepository,
        *,
        enabled: bool = False,
        minimum_balance: float = 0.58,
        decision_threshold: float = 0.58,
    ) -> None:
        self.genome = genome
        self.autonomy = autonomy
        self.experience = experience
        self.enabled = enabled
        self.minimum_balance = minimum_balance
        self.decision_threshold = decision_threshold

    async def evaluate(self, event: Event, intent: ActionIntent) -> InitiativeDecision:
        denied = InitiativeDecision(False, 0.0, None, ())
        if (
            not self.enabled
            or event.channel_id is None
            or event.actor_id is None
            or event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
            or event.payload.get("author_is_bot") is True
            or event.payload.get("bot_mentioned") is True
        ):
            return denied
        content = event.payload.get("content")
        if not isinstance(content, str) or len(tokenize(content)) < 2:
            return denied
        state = await self.autonomy.state(event.guild_id, event.channel_id)
        if state.evidence_balance < self.minimum_balance:
            return denied
        if await self.autonomy.open_goal(event.guild_id, event.channel_id) is not None:
            return denied
        self_model = await self.experience.self_model(event.guild_id)
        context = intent.decision_context
        uncertainty = _number(context, "state_uncertainty")
        interest = _number(context, "state_interest")
        unexpectedness = _number(context, "state_unexpectedness")
        familiarity = _number(context, "familiarity")
        information_gain = _clamp(uncertainty * 0.45 + interest * 0.35 + unexpectedness * 0.20)
        social_grounding = _clamp(
            familiarity * 0.45 + self.genome.sociability * 0.35 + self.genome.curiosity * 0.20
        )
        drive = _clamp(
            self.genome.initiative * 0.25
            + self.genome.curiosity * 0.15
            + information_gain * 0.24
            + state.evidence_balance * 0.14
            + self_model.efficacy * 0.10
            + social_grounding * 0.12
        )
        jitter = (_stable_fraction(event.event_id) - 0.5) * 0.10
        effective_drive = _clamp(drive + jitter)
        reasoning: tuple[dict[str, object], ...] = (
            {
                "stage": "initiative_state",
                "evidence_balance": round(state.evidence_balance, 6),
                "open_goal": False,
            },
            {
                "stage": "expected_information_gain",
                "uncertainty": round(uncertainty, 6),
                "interest": round(interest, 6),
                "unexpectedness": round(unexpectedness, 6),
                "value": round(information_gain, 6),
            },
            {
                "stage": "self_model",
                "efficacy": round(self_model.efficacy, 6),
                "initiative_attempts": self_model.initiative_attempts,
                "initiative_successes": self_model.initiative_successes,
            },
            {
                "stage": "initiative_decision",
                "drive": round(effective_drive, 6),
                "threshold": self.decision_threshold,
            },
        )
        should = effective_drive >= self.decision_threshold
        return InitiativeDecision(
            should_initiate=should,
            drive=effective_drive,
            goal_id=f"goal:{uuid4()}" if should else None,
            reasoning=reasoning,
        )


def _number(context: dict[str, object], key: str) -> float:
    value = context.get(key, 0.0)
    return (
        _clamp(float(value))
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )


def _stable_fraction(value: str) -> float:
    integer = int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
    return integer / ((1 << 64) - 1)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
