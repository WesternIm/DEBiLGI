"""Phenotype-like personality expression derived from genome, state, and experience."""

from __future__ import annotations

from dataclasses import dataclass

from debilgi.cognition.models import ActionIntent
from debilgi.personality.genome import PersonalityGenome


@dataclass(frozen=True, slots=True)
class PersonalityExpression:
    genome_signature: str
    chaos: float
    curiosity: float
    impulsivity: float
    expressiveness: float
    memory_attachment: float
    novelty_drive: float
    coherence_drive: float
    target_tokens: int
    candidate_count: int


class PersonalityExpressionEngine:
    def __init__(self, genome: PersonalityGenome) -> None:
        self.genome = genome

    def express(self, intent: ActionIntent) -> PersonalityExpression:
        context = intent.decision_context
        activation = _number(context, "state_activation")
        tension = _number(context, "state_tension")
        interest = _number(context, "state_interest")
        uncertainty = _number(context, "state_uncertainty")
        familiarity = _number(context, "familiarity")
        chaos = _clamp(self.genome.chaos * 0.78 + activation * 0.12 + tension * 0.10)
        curiosity = _clamp(self.genome.curiosity * 0.72 + interest * 0.18 + uncertainty * 0.10)
        impulsivity = _clamp(self.genome.impulsivity * 0.82 + activation * 0.18)
        expressiveness = _clamp(self.genome.expressiveness * 0.80 + tension * 0.20)
        memory_attachment = _clamp(self.genome.memory_attachment * 0.75 + familiarity * 0.25)
        novelty = _clamp(self.genome.novelty_seeking * 0.80 + interest * 0.20)
        coherence = _clamp(0.88 - chaos * 0.30 + familiarity * 0.08)
        target_tokens = round(
            8
            + self.genome.verbosity * 22
            + interest * 10
            + expressiveness * 6
            - self.genome.impulsivity * 2
        )
        candidate_count = round(24 + curiosity * 22 + novelty * 16)
        return PersonalityExpression(
            genome_signature=self.genome.signature,
            chaos=chaos,
            curiosity=curiosity,
            impulsivity=impulsivity,
            expressiveness=expressiveness,
            memory_attachment=memory_attachment,
            novelty_drive=novelty,
            coherence_drive=coherence,
            target_tokens=max(8, min(target_tokens, 48)),
            candidate_count=max(20, min(candidate_count, 80)),
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
