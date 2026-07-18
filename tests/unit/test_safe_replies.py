from datetime import UTC, datetime, timedelta

from debilgi.autonomy.models import InitiativeDecision
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.discord.replies import SafeMentionReplyService
from debilgi.events.models import Event, EventType
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def intent(action_type: ActionType = ActionType.SHORT_REPLY) -> ActionIntent:
    return ActionIntent(
        intent_id="intent:test",
        guild_id=1,
        action_type=action_type,
        target_id=100,
        channel_id=10,
        urgency=0.5,
        confidence=0.8,
        expected_effect={"hypothesis": "acknowledge direct mention"},
        reasoning_trace=(),
        evidence=("discord:message:1",),
        decision_context={"model_observations": 1, "familiarity": 0.02},
        created_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
    )


class FakeRepository:
    def __init__(self, stored_intent: ActionIntent | None) -> None:
        self.stored_intent = stored_intent

    async def intent_for_event(self, event_id: str) -> ActionIntent | None:
        assert event_id == "discord:message:1"
        return self.stored_intent


class FakeMemory:
    def __init__(
        self,
        *,
        user_events: tuple[Event, ...] = (),
        channel_events: tuple[Event, ...] = (),
    ) -> None:
        self.user_events = user_events
        self.channel_events = channel_events

    async def user_context(self, guild_id: int, user_id: int) -> tuple[Event, ...]:
        assert (guild_id, user_id) == (1, 100)
        return self.user_events

    async def channel_context(self, guild_id: int, channel_id: int) -> tuple[Event, ...]:
        assert (guild_id, channel_id) == (1, 10)
        return self.channel_events


class FakeArchiveRepository:
    def __init__(
        self,
        *,
        texts: tuple[str, ...] = (),
        reply_heads: tuple[tuple[str, ...], ...] = (),
        ngrams: tuple[tuple[tuple[str, ...], int], ...] = (),
    ) -> None:
        self.texts = texts
        self.reply_heads = reply_heads
        self.ngrams = ngrams

    async def recent_texts(
        self, guild_id: int, channel_id: int, limit: int = 50
    ) -> tuple[str, ...]:
        assert (guild_id, channel_id, limit) == (1, 10, 50)
        return self.texts

    async def matching_reply_heads(
        self, guild_id: int, prompt_tokens: tuple[str, ...], limit: int = 8
    ) -> tuple[tuple[str, ...], ...]:
        assert guild_id == 1
        assert prompt_tokens
        assert limit == 8
        return self.reply_heads

    async def common_ngrams(
        self, guild_id: int, limit: int = 1000
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        assert (guild_id, limit) == (1, 1000)
        return self.ngrams


class FakeExperienceConsolidator:
    async def transition_adjustments(
        self, guild_id: int
    ) -> tuple[tuple[tuple[str, ...], float], ...]:
        assert guild_id == 1
        return ()


class FakeInitiativePolicy:
    def __init__(self, decision: InitiativeDecision | None = None) -> None:
        self.decision = decision or InitiativeDecision(False, 0.0, None, ())

    async def evaluate(self, event: Event, stored_intent: ActionIntent) -> InitiativeDecision:
        assert event.event_id == "discord:message:1"
        assert stored_intent.intent_id == "intent:test"
        return self.decision


EXPRESSION_ENGINE = PersonalityExpressionEngine(DEBILGI_GENOME)
EXPERIENCE = FakeExperienceConsolidator()
INITIATIVE = FakeInitiativePolicy()


def event(
    *,
    mentioned: bool,
    author_is_bot: bool = False,
    content: str = "<@123> che",
    event_id: str = "discord:message:1",
    actor_id: int = 100,
) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=actor_id,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=NOW,
        payload={
            "bot_mentioned": mentioned,
            "author_is_bot": author_is_bot,
            "content": content,
        },
        source="test",
    )


async def test_enabled_service_renders_the_stored_reply_intent() -> None:
    current = event(mentioned=True)
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response == "che che che"


async def test_service_synthesizes_words_from_previous_memory() -> None:
    previous = event(
        mentioned=True,
        content="<@123> red button drops server",
        event_id="discord:message:0",
    )
    current = event(mentioned=True, content="<@123> continue")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(previous, current), channel_events=(previous, current)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response is not None
    assert set(response.split()) <= {"red", "button", "drops", "server", "continue"}
    assert {"red", "button", "drops", "server"} & set(response.split())


async def test_service_uses_learned_archive_reply_head() -> None:
    current = event(mentioned=True, content="<@123> learned archive")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(
            texts=("learned archive answer continues topic",),
            reply_heads=(("learned", "archive", "answer"),),
            ngrams=(
                (("learned", "archive", "answer"), 5),
                (("archive", "answer", "continues"), 5),
            ),
        ),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response is not None
    assert "answer" in response


async def test_service_ignores_plain_messages_and_bot_authors() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    assert await service.response_for(event(mentioned=False)) is None
    assert await service.response_for(event(mentioned=True, author_is_bot=True)) is None


async def test_nonverbal_intent_remains_silent() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent(ActionType.SAVE_OBSERVATION)),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    assert await service.response_for(event(mentioned=True)) is None


async def test_initiative_is_generated_as_a_question_from_memory() -> None:
    current = event(mentioned=False, content="server behaves strangely again")
    initiative = FakeInitiativePolicy(
        InitiativeDecision(
            should_initiate=True,
            drive=0.81,
            goal_id="goal:test",
            reasoning=({"stage": "initiative_decision"},),
        )
    )
    service = SafeMentionReplyService(
        FakeRepository(intent(ActionType.SAVE_OBSERVATION)),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(
            texts=(
                "server behaves strangely after update",
                "why server falls again tonight",
            )
        ),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        initiative,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    prepared_reply = await service.prepare_initiative(current)

    assert prepared_reply is not None
    assert prepared_reply.initiative
    assert prepared_reply.goal_id == "goal:test"
    assert prepared_reply.text.endswith("?")
