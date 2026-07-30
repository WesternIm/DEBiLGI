from dataclasses import replace
from datetime import UTC, datetime

from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.position_reasoning import PositionReasoningService
from debilgi.personality.expression import PersonalityExpression
from debilgi.personality.perspective import (
    build_personality_perspective,
    form_personal_opinion,
)
from debilgi.social.models import PrimitiveVector, RelationState

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=UTC)


def _expression() -> PersonalityExpression:
    return PersonalityExpression(
        genome_signature="genome:test",
        chaos=0.72,
        curiosity=0.76,
        impulsivity=0.78,
        expressiveness=0.82,
        memory_attachment=0.70,
        novelty_drive=0.80,
        coherence_drive=0.68,
        target_tokens=30,
        candidate_count=58,
        sociability=0.66,
        suspicion=0.48,
        patience=0.34,
        assertiveness=0.75,
        warmth=0.62,
        playfulness=0.78,
        uncertainty=0.16,
    )


def _relation(*, supportive: bool) -> RelationState:
    vector = (
        PrimitiveVector(0.78, 0.12, 0.82, 0.48, 0.08, 0.84, 0.42)
        if supportive
        else PrimitiveVector(0.10, 0.82, 0.08, 0.28, 0.76, 0.12, 0.66)
    )
    return RelationState(1, 2, vector, 18, NOW)


def _opinion_frame() -> DialogueActFrame:
    return DialogueActFrame(
        kind="opinion_question",
        response_obligation="express_position",
        subject="германия",
        predicate="быть",
        objects=("германия", "крутой"),
        polarity=0.0,
        register="neutral",
        complete=True,
        confidence=0.86,
        evidence=("opinion_request",),
    )


def test_relationship_history_changes_the_way_personality_is_expressed() -> None:
    supportive = build_personality_perspective(
        _expression(),
        relation=_relation(supportive=True),
    )
    hostile = build_personality_perspective(
        _expression(),
        relation=_relation(supportive=False),
    )

    assert supportive.trust > hostile.trust
    assert supportive.warmth > hostile.warmth
    assert hostile.resistance > supportive.resistance
    assert supportive.candidate_fit("acknowledge") > hostile.candidate_fit(
        "acknowledge"
    )
    assert "relation:observations:18" in supportive.evidence


def test_personal_values_form_an_opinion_before_any_sentence_exists() -> None:
    perspective = build_personality_perspective(_expression())
    opinion = form_personal_opinion(
        "self",
        "крутой",
        perspective=perspective,
    )

    assert opinion.stance == "support"
    assert opinion.score > 0.2
    assert {"expressiveness", "assertiveness"} <= set(opinion.value_dimensions)
    assert any(item.startswith("personality:") for item in opinion.evidence)


async def test_position_reasoning_uses_personality_but_keeps_it_revisable() -> None:
    perspective = build_personality_perspective(_expression())
    service = PositionReasoningService()

    personal = await service.resolve(
        1,
        _opinion_frame(),
        perspective=perspective,
    )
    grounded_opposition = await service.resolve(
        1,
        _opinion_frame(),
        facts=("Германия не крутая.",),
        perspective=perspective,
    )

    assert personal is not None
    assert personal.stance == "support"
    assert "personality_value_appraisal" in personal.evidence
    assert grounded_opposition is not None
    assert grounded_opposition.stance == "oppose"
    assert "grounded_fact" in grounded_opposition.evidence


def test_current_phenotype_changes_opinion_strength_without_changing_identity() -> None:
    expressive = build_personality_perspective(_expression())
    restrained = build_personality_perspective(
        replace(
            _expression(),
            expressiveness=0.12,
            assertiveness=0.18,
            playfulness=0.10,
        )
    )

    expressive_opinion = form_personal_opinion(
        "self",
        "крутой",
        perspective=expressive,
    )
    restrained_opinion = form_personal_opinion(
        "self",
        "крутой",
        perspective=restrained,
    )

    assert expressive.genome_signature == restrained.genome_signature
    assert expressive_opinion.score > restrained_opinion.score
