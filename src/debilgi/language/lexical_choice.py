"""Choose a surface lexeme for an already selected concept.

This module deliberately sits between semantic planning and morphology. A
planner supplies a concept and admissible lexical alternatives; this layer
scores alternatives by sense evidence, register and discourse repetition. It
never chooses a proposition and never stores answer phrases.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LexicalOption:
    lemma: str
    surface: str
    meaning: str
    sense_id: str | None = None
    register: str = "neutral"
    evidence: tuple[str, ...] = ()
    salience: float = 0.5


@dataclass(frozen=True, slots=True)
class LexicalChoice:
    concept: str
    option: LexicalOption
    alternatives_considered: tuple[str, ...]
    reason: tuple[str, ...]


def choose_lexeme(
    concept: str,
    options: tuple[LexicalOption, ...],
    *,
    preferred_register: str | None = None,
    recently_used: tuple[str, ...] = (),
) -> LexicalChoice | None:
    """Select among words that express the same grounded concept.

    A word without evidence is not allowed to win merely because it sounds
    fluent. Repetition is a tie-breaker, never a reason to change meaning.
    """
    eligible = tuple(option for option in options if option.meaning.strip() and option.evidence)
    if not eligible:
        return None
    recent = {item.casefold() for item in recently_used}

    def score(option: LexicalOption) -> tuple[float, float, float, str]:
        register_fit = 1.0 if preferred_register and option.register == preferred_register else 0.0
        novelty = 0.15 if option.surface.casefold() not in recent else -0.15
        return (register_fit, option.salience + novelty, -len(option.surface), option.lemma)

    selected = max(eligible, key=score)
    reasons = ["shared_grounded_meaning"]
    if preferred_register and selected.register == preferred_register:
        reasons.append("register_match")
    if selected.surface.casefold() not in recent:
        reasons.append("discourse_novelty")
    return LexicalChoice(
        concept,
        selected,
        tuple(option.lemma for option in eligible),
        tuple(reasons),
    )
