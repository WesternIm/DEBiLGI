from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME, PersonalityGenome

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def make_intent(**decision_context: object) -> ActionIntent:
    return ActionIntent(
        intent_id="intent:personality",
        guild_id=1,
        action_type=ActionType.SHORT_REPLY,
        target_id=2,
        channel_id=3,
        urgency=0.5,
        confidence=0.7,
        expected_effect={},
        reasoning_trace=(),
        evidence=(),
        decision_context=dict(decision_context),
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )


def test_genome_is_valid_immutable_and_signed() -> None:
    assert len(DEBILGI_GENOME.signature) == 16
    assert DEBILGI_GENOME.signature == DEBILGI_GENOME.signature
    assert all(0.0 <= value <= 1.0 for value in DEBILGI_GENOME.traits().values())

    with pytest.raises(FrozenInstanceError):
        DEBILGI_GENOME.chaos = 0.0  # type: ignore[misc]


def test_invalid_trait_is_rejected() -> None:
    traits = DEBILGI_GENOME.traits()
    traits["chaos"] = 2.0
    with pytest.raises(ValueError, match="chaos"):
        PersonalityGenome("broken", 1, **traits)


def test_state_and_experience_change_expression_without_changing_genome() -> None:
    engine = PersonalityExpressionEngine(DEBILGI_GENOME)
    calm = engine.express(make_intent())
    activated = engine.express(
        make_intent(
            state_activation=1.0,
            state_tension=1.0,
            state_interest=1.0,
            state_uncertainty=1.0,
            familiarity=1.0,
        )
    )

    assert activated.chaos > calm.chaos
    assert activated.curiosity > calm.curiosity
    assert activated.memory_attachment > calm.memory_attachment
    assert activated.target_tokens > calm.target_tokens
    assert activated.assertiveness != calm.assertiveness
    assert activated.suspicion > calm.suspicion
    assert activated.playfulness != calm.playfulness
    assert activated.genome_signature == calm.genome_signature == DEBILGI_GENOME.signature
