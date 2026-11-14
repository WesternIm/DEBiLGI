"""Compose primitive social drives from text, state, personality and relationship history."""

from __future__ import annotations

from uuid import uuid4

from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.models import ActionIntent
from debilgi.events.models import Event, EventType
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.tokenizer import tokenize
from debilgi.personality.genome import PersonalityGenome
from debilgi.social.models import PrimitiveVector, SocialReaction, composition
from debilgi.social.repository import SocialReactionRepository
from debilgi.social.signals import evidence_counts

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


class SocialReactionEngine:
    def __init__(
        self,
        repository: SocialReactionRepository,
        cognitive_repository: CognitiveRepository,
        genome: PersonalityGenome,
    ) -> None:
        self.repository = repository
        self._cognitive_repository = cognitive_repository
        self.genome = genome

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        if await self.repository.reaction_for_event(event.event_id) is not None:
            return
        intent = await self._cognitive_repository.intent_for_event(event.event_id)
        prior = (
            await self.repository.relation_state(event.guild_id, event.actor_id)
            if event.actor_id is not None
            else None
        )
        reaction = assess_reaction(event, intent, self.genome, prior.vector if prior else None)
        await self.repository.save(reaction)


def assess_reaction(
    event: Event,
    intent: ActionIntent | None,
    genome: PersonalityGenome,
    prior: PrimitiveVector | None = None,
) -> SocialReaction:
    content = event.payload.get("content")
    text = content if isinstance(content, str) else ""
    tokens = tokenize(text)
    counts = evidence_counts(tokens)
    pragmatics = analyze_pragmatics(text)
    context = intent.decision_context if intent else {}
    familiarity = _number(context, "familiarity")
    predictability = _number(context, "predictability")
    uncertainty = _number(context, "state_uncertainty")
    interest = _number(context, "state_interest")
    tension = _number(context, "state_tension")
    activation = _number(context, "state_activation")
    unexpectedness = _number(context, "state_unexpectedness")
    question_signal = max(float("?" in tokens), pragmatics.question_force)
    exclamation_signal = min(1.0, tokens.count("!") * 0.5)
    uncertainty_signal = pragmatics.uncertainty
    emotional_signal = pragmatics.emotional_intensity
    if question_signal or {"или", "либо"}.intersection(tokens):
        # Alternatives named in a question are not the speaker's current stance.
        counts = {**counts, "affirmation": 0, "refusal": 0}

    vector = PrimitiveVector(
        agreement=_clamp(
            0.08
            + counts["affirmation"] * 0.48
            + counts["approval"] * 0.12
            + familiarity * 0.12
            - counts["refusal"] * 0.30
        ),
        resistance=_clamp(
            0.08
            + counts["refusal"] * 0.46
            + counts["disapproval"] * 0.18
            + tension * 0.16
            + genome.suspicion * 0.08
        ),
        approval=_clamp(
            0.08
            + counts["approval"] * 0.48
            + counts["affirmation"] * 0.14
            + predictability * 0.10
            - counts["disapproval"] * 0.24
        ),
        curiosity=_clamp(
            0.10
            + counts["curiosity"] * 0.24
            + question_signal * 0.24
            + uncertainty * 0.16
            + uncertainty_signal * 0.12
            + interest * 0.14
            + genome.curiosity * 0.10
        ),
        aversion=_clamp(
            0.05
            + counts["aversion"] * 0.52
            + counts["disapproval"] * 0.16
            + tension * 0.12
            + unexpectedness * 0.08
        ),
        trust=_clamp(
            0.08
            + familiarity * 0.34
            + predictability * 0.30
            + counts["affirmation"] * 0.08
            - counts["refusal"] * 0.08
        ),
        arousal=_clamp(
            0.08
            + activation * 0.36
            + tension * 0.26
            + exclamation_signal * 0.22
            + emotional_signal * 0.12
            + genome.expressiveness * 0.08
        ),
    )
    if prior is not None:
        vector = vector.blended(prior, 0.14)
    primary, secondary, blend_id = composition(vector)
    signal_names = tuple(name for name, count in counts.items() if count > 0)
    signal_strength = min(1.0, sum(counts.values()) / 4.0)
    confidence = _clamp(0.30 + signal_strength * 0.45 + familiarity * 0.15 + 0.10 * bool(intent))
    return SocialReaction(
        reaction_id=f"reaction:{uuid4()}",
        event_id=event.event_id,
        guild_id=event.guild_id,
        actor_id=event.actor_id,
        vector=vector,
        primary_component=primary,
        secondary_component=secondary,
        blend_id=blend_id,
        confidence=confidence,
        signals=signal_names,
        created_at=event.utc_timestamp,
    )


def _number(context: dict[str, object], key: str) -> float:
    value = context.get(key, 0.0)
    return (
        _clamp(float(value))
        if isinstance(value, int | float) and not isinstance(value, bool)
        else 0.0
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
