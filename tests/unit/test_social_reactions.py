from datetime import UTC, datetime

from debilgi.cognition.models import ActionType
from debilgi.events.models import Event, EventType
from debilgi.language.planning import ReplyContext, StructuredThought, utterance_plan
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.social.engine import assess_reaction
from debilgi.social.models import PrimitiveVector, SocialReaction, composition

NOW = datetime(2026, 7, 18, 14, 0, tzinfo=UTC)


def event(content: str) -> Event:
    return Event(
        event_id=f"social:{content}",
        guild_id=1,
        channel_id=2,
        actor_id=3,
        target_id=4,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={"content": content, "author_is_bot": False},
        source="test",
    )


def reaction(vector: PrimitiveVector) -> SocialReaction:
    primary, secondary, blend_id = composition(vector)
    return SocialReaction(
        reaction_id="reaction:test",
        event_id="event:test",
        guild_id=1,
        actor_id=3,
        vector=vector,
        primary_component=primary,
        secondary_component=secondary,
        blend_id=blend_id,
        confidence=0.8,
        signals=(),
        created_at=NOW,
    )


def test_affirmation_and_approval_emerge_from_positive_evidence() -> None:
    result = assess_reaction(event("да, это отлично"), None, DEBILGI_GENOME)

    assert result.vector.agreement > 0.5
    assert result.vector.approval > 0.5
    assert {result.primary_component, result.secondary_component} == {"agreement", "approval"}


def test_curiosity_and_aversion_form_unlisted_composite() -> None:
    result = assess_reaction(event("почему это мерзко?"), None, DEBILGI_GENOME)

    assert result.vector.curiosity > 0.5
    assert result.vector.aversion > 0.5
    assert set(result.blend_id.split("+")) == {"curiosity", "aversion"}


def test_resistance_changes_statement_into_boundary_act() -> None:
    vector = PrimitiveVector(
        agreement=0.1,
        resistance=0.82,
        approval=0.1,
        curiosity=0.2,
        aversion=0.55,
        trust=0.2,
        arousal=0.6,
    )
    context = ReplyContext(reaction=reaction(vector))
    thought = StructuredThought(ActionType.SHORT_REPLY, 3, "respond", 0.8, ())

    plan = utterance_plan(thought, context=context)

    assert plan is not None
    assert plan.speech_act == "boundary_statement"
