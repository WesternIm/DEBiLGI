from dataclasses import replace
from datetime import UTC, datetime

from debilgi.actions.decision import ShadowDecisionEngine
from debilgi.cognition.models import ActionType, EntityModel, EntityType, InternalState
from debilgi.events.models import Event, EventType
from debilgi.personality.genome import DEBILGI_GENOME

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def model(*, familiarity: float, observations: int, predictability: float) -> EntityModel:
    return EntityModel(
        guild_id=1,
        entity_type=EntityType.USER,
        entity_id=100,
        familiarity=familiarity,
        interaction_intensity=0.5,
        predictability=predictability,
        reply_probability=0.5,
        average_reply_delay=None,
        pattern_stability=predictability,
        expectation_reliability=predictability,
        unfinished_interactions=0,
        bot_action_response=0.5,
        observations=observations,
        updated_at=NOW,
    )


def state(*, familiarity: float, uncertainty: float, interest: float) -> InternalState:
    return InternalState(
        guild_id=1,
        subject_id=100,
        unexpectedness=0.1,
        uncertainty=uncertainty,
        activation=0.4,
        tension=0.1,
        interest=interest,
        familiarity=familiarity,
        expected_control=0.5,
        goal_alignment=1.0,
        updated_at=NOW,
    )


def test_same_observation_can_produce_different_intents_from_different_histories() -> None:
    event = Event(
        event_id="same-situation",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={},
        source="test",
    )
    engine = ShadowDecisionEngine()

    novice = engine.decide(
        event,
        model(familiarity=0.0, observations=1, predictability=0.5),
        state(familiarity=0.0, uncertainty=0.5, interest=0.2),
        active_expectations=1,
    )
    familiar = engine.decide(
        event,
        model(familiarity=0.9, observations=30, predictability=0.9),
        state(familiarity=0.9, uncertainty=0.1, interest=0.9),
        active_expectations=1,
    )

    assert novice.action_type != familiar.action_type
    assert novice.shadow_only and familiar.shadow_only
    assert novice.executed_at is None and familiar.executed_at is None


def test_direct_mention_selects_a_short_reply() -> None:
    event = Event(
        event_id="mention",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"bot_mentioned": True, "author_is_bot": False},
        source="test",
    )

    intent = ShadowDecisionEngine().decide(
        event,
        model(familiarity=0.0, observations=1, predictability=0.5),
        state(familiarity=0.0, uncertainty=0.5, interest=0.2),
        active_expectations=1,
    )

    assert intent.action_type is ActionType.SHORT_REPLY
    assert intent.decision_context["genome_signature"]
    assert any(step["stage"] == "genome" for step in intent.reasoning_trace)


def test_innate_genome_changes_decision_in_the_same_situation() -> None:
    event = Event(
        event_id="genome-choice",
        guild_id=1,
        channel_id=10,
        actor_id=100,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={},
        source="test",
    )
    social = replace(
        DEBILGI_GENOME,
        sociability=1.0,
        impulsivity=1.0,
        response_threshold=0.0,
    )
    withdrawn = replace(
        DEBILGI_GENOME,
        sociability=0.0,
        impulsivity=0.0,
        response_threshold=1.0,
    )
    situation_model = model(familiarity=0.5, observations=10, predictability=0.5)
    situation_state = state(familiarity=0.5, uncertainty=0.2, interest=0.2)

    social_intent = ShadowDecisionEngine(social).decide(
        event, situation_model, situation_state, active_expectations=0
    )
    withdrawn_intent = ShadowDecisionEngine(withdrawn).decide(
        event, situation_model, situation_state, active_expectations=0
    )

    assert social_intent.action_type is ActionType.SHORT_REPLY
    assert withdrawn_intent.action_type is ActionType.STAY_SILENT
    assert (
        social_intent.decision_context["genome_signature"]
        != withdrawn_intent.decision_context["genome_signature"]
    )
