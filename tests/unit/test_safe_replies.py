import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import debilgi.discord.replies as reply_module
from debilgi.autonomy.models import InitiativeDecision
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.discord.replies import SafeMentionReplyService, _stable_response_seed
from debilgi.events.models import Event, EventType
from debilgi.experience.models import PreparedReply
from debilgi.language.causal_reasoning import infer_compositional_consequence
from debilgi.language.conversation import ResponsePlan
from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.open_questions import ExplanationResolution
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.russian_reference import RemoteLexeme
from debilgi.language.semantic_consultant import ConsultationResult
from debilgi.language.sentence_understanding import UtteranceUnderstanding
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


def test_research_surface_gate_separates_answers_from_dictionary_metadata() -> None:
    agent_question = SimpleNamespace(
        subject="addressee",
        response_obligation="answer_question",
        complete=True,
    )
    world_question = SimpleNamespace(
        subject=None,
        response_obligation="answer_question",
        complete=True,
    )

    assert reply_module._looks_like_lexical_definition("Хотеть — глагол, обозначающий желание.")
    assert not reply_module._looks_like_lexical_definition("У осьминога три сердца.")
    assert not reply_module._research_facts_may_surface(
        dialogue_act=agent_question,
        structure=None,
        explicit_lookup=False,
    )
    assert reply_module._research_facts_may_surface(
        dialogue_act=world_question,
        structure=None,
        explicit_lookup=False,
    )
    assert reply_module._research_facts_may_surface(
        dialogue_act=agent_question,
        structure=None,
        explicit_lookup=True,
    )


def test_preference_thought_only_answers_polar_or_explicit_opinion_requests() -> None:
    understanding = UtteranceUnderstanding(
        content_terms=("почему", "любить", "небо"),
        known_terms=("любить", "небо"),
        unknown_terms=(),
        relations=(),
        context_links=(),
        comprehension_score=1.0,
        sufficient=True,
    )

    assert (
        reply_module._preference_question_thought("Почему ты не любишь небо?", understanding)
        is None
    )
    assert (
        reply_module._preference_question_thought(
            "Изучи, что такое небо, и скажи своё мнение", understanding
        )
        is not None
    )
    reason = reply_module._preference_reason(
        "Почему ты не любишь небо?",
        understanding,
        facts=("Небо связано с панорамой пространства и космосом.",),
    )
    assert reason is not None
    assert realize_thought(reason)[0].text.startswith("Нет, ")


def test_response_seed_depends_on_semantics_not_discord_event_id() -> None:
    first = _stable_response_seed("Что это?", ("мой компьютер решил",))
    repeated = _stable_response_seed("Что это?", ("мой компьютер решил",))
    changed_context = _stable_response_seed("Что это?", ("ворон достал ключ",))

    assert first == repeated
    assert first != changed_context


def test_fallback_never_prints_raw_plan_fact_without_quote_request() -> None:
    copied_fact = "Ответ — реплика, вызванная заданным вопросом."
    plan = ResponsePlan(
        plan_id="plan:test",
        source_event_id="discord:message:1",
        conversation_id=None,
        author_id=100,
        topic_ids=(),
        message_intent="statement",
        response_goal="continue_topic",
        key_points=(copied_fact,),
        tone="neutral",
        unknowns=(),
        confidence=0.8,
    )

    result = SafeMentionReplyService._fallback_for_plan(
        plan,
        (),
        prompt="Это работает.",
    )

    assert result != copied_fact
    assert "вызванная заданным вопросом" not in result.casefold()


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


class FailingRepository:
    async def intent_for_event(self, event_id: str) -> ActionIntent | None:
        raise RuntimeError(f"broken repository for {event_id}")


class FakeMemory:
    def __init__(
        self,
        *,
        user_events: tuple[Event, ...] = (),
        channel_events: tuple[Event, ...] = (),
    ) -> None:
        self.user_events = user_events
        self.channel_events = channel_events
        self.limits = SimpleNamespace(max_context_events=24)

    async def relevant_context(
        self,
        guild_id: int,
        *,
        channel_id: int | None,
        user_id: int | None,
        prompt: str,
        limit: int | None = None,
    ) -> tuple[Event, ...]:
        assert (guild_id, channel_id, user_id) == (1, 10, 100)
        assert prompt
        del limit
        combined = (*self.channel_events, *self.user_events)
        return tuple(
            event
            for index, event in enumerate(combined)
            if event.event_id not in {item.event_id for item in combined[:index]}
        )

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
        learned_replies: tuple[str, ...] = (),
        exact_reply: str | None = None,
        ngrams: tuple[tuple[tuple[str, ...], int], ...] = (),
    ) -> None:
        self.texts = texts
        self.reply_heads = reply_heads
        self.learned_replies = learned_replies
        self.exact_reply = exact_reply
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

    async def matching_replies(
        self, guild_id: int, prompt_tokens: tuple[str, ...], limit: int = 8
    ) -> tuple[str, ...]:
        del guild_id, prompt_tokens, limit
        raise AssertionError("complete archived replies must not enter reply context")

    async def exact_learned_reply(
        self, guild_id: int, prompt_tokens: tuple[str, ...]
    ) -> str | None:
        del guild_id, prompt_tokens
        raise AssertionError("exact archived replies must not become answers")

    async def common_ngrams(
        self, guild_id: int, limit: int = 1000
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        assert (guild_id, limit) == (1, 1000)
        return self.ngrams


class FixedGenerator:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, plan) -> str:
        del plan
        return self.text


class GroundedOnlyGenerator:
    def __init__(self, text: str) -> None:
        self.text = text

    def generate_candidates(self, plan, *, limit: int = 8):
        del plan, limit
        return (
            SimpleNamespace(
                text=self.text,
                source="grounded_surface",
                renderer="grounded_local_fact",
                variant=0,
            ),
        )


class FakeSemanticConsultant:
    trigger_threshold = 0.72

    def __init__(self) -> None:
        self.calls = []

    def should_consult(
        self,
        *,
        local_confidence: float,
        unknown_terms,
        has_bound_answer: bool,
    ) -> bool:
        del local_confidence, unknown_terms
        return not has_bound_answer

    async def consult(self, **kwargs) -> ConsultationResult:
        self.calls.append(kwargs)
        return ConsultationResult.from_payload(
            {
                "interpretation": "Пользователь спрашивает значение сленгового слова.",
                "intent": "definition_question",
                "entities": [{"surface": "база", "lemma": "база", "type": "abstract"}],
                "relations": [],
                "ambiguities": [],
                "facts": ["В сленге «база» выражает согласие или одобрение."],
                "response_text": "В этом контексте «база» — это согласие или одобрение.",
                "confidence": 0.82,
            },
            provider="fake-puter",
            model="fake-model",
        )


class InterpretationOnlySemanticConsultant(FakeSemanticConsultant):
    async def consult(self, **kwargs) -> ConsultationResult:
        self.calls.append(kwargs)
        return ConsultationResult.from_payload(
            {
                "interpretation": (
                    "Пользователь настойчиво интересуется, находится ли агент "
                    "в Кваргонии или происходит ли он оттуда."
                ),
                "intent": "origin_question",
                "entities": [],
                "relations": [],
                "ambiguities": [],
                "facts": [],
                "confidence": 0.82,
            },
            provider="fake-puter",
            model="fake-model",
        )


class FakeExperienceConsolidator:
    async def transition_adjustments(
        self, guild_id: int
    ) -> tuple[tuple[tuple[str, ...], float], ...]:
        assert guild_id == 1
        return ()


class FakeKnowledgeService:
    async def grounding_for_event(
        self, current: Event, prompt: str
    ) -> tuple[tuple[str, ...], str | None]:
        assert current.event_id == "discord:message:1"
        assert "мороженое" in prompt
        answer = "Мороженое — замороженный сладкий десерт."
        return (answer,), answer


class RelationKnowledgeService:
    def __init__(self) -> None:
        self.study_calls: list[tuple[str, ...]] = []

    async def grounding_for_event(
        self, current: Event, prompt: str
    ) -> tuple[tuple[str, ...], str | None]:
        del current, prompt
        return (), None

    async def internalize_for_reasoning(
        self,
        current: Event,
        terms: tuple[str, ...],
        *,
        limit: int = 6,
    ) -> tuple[SimpleNamespace, ...]:
        del current, limit
        self.study_calls.append(terms)
        definitions = {
            "носка": "действие или процесс ношения чего-либо",
            "носок": "предмет одежды, короткое изделие из ткани",
            "куртка": "предмет верхней одежды, изделие из ткани",
            "память": "способность сохранять и использовать информацию",
        }
        return tuple(
            SimpleNamespace(title=term, definition=definitions[term])
            for term in terms
            if term in definitions
        )


class RelationLexicon:
    def __init__(self) -> None:
        self.background_queue: list[tuple[str, ...]] = []

    async def analyze_word(self, word: str, **kwargs) -> WordAnalysis:
        del kwargs
        normalized = word.casefold()
        lemmas = {
            "носки": "носка",
            "куртка": "куртка",
            "память": "память",
            "хочу": "хотеть",
            "узнать": "узнать",
        }
        parts = {"хочу": "VERB", "узнать": "INFN"}
        return WordAnalysis(
            surface_form=word,
            normalized_form=normalized,
            lemma=lemmas.get(normalized, normalized),
            part_of_speech=parts.get(normalized, "NOUN"),
            grammatical_features={},
            dictionary_meanings=(),
            style_labels=(),
            source="opencorpora",
            confidence=ConfidenceProfile(0.82, 0.48, 0.42, 0.50, 0.42),
        )

    async def queue_semantic_study(
        self,
        current: Event,
        terms: tuple[str, ...],
        *,
        reason: str,
        priority: float = 0.72,
    ) -> int:
        del current, reason, priority
        self.background_queue.append(terms)
        return len(terms)


class FakeRussianReference:
    async def refresh_word(self, word: str) -> RemoteLexeme | None:
        assert word == "кринж"
        return RemoteLexeme(
            "кринж",
            "NOUN",
            ("чувство стыда за чужие действия",),
            "https://ru.wiktionary.org/wiki/кринж",
            ("молодёжное",),
        )


class FakeOpenQuestions:
    def __init__(self, resolution: ExplanationResolution | None = None) -> None:
        self.resolution = resolution
        self.opened: list[tuple[str, int]] = []

    async def open_for(
        self,
        prompt: Event,
        *,
        concept: str,
        question_message_id: int,
        asked_at: datetime,
    ) -> str:
        del prompt, asked_at
        self.opened.append((concept, question_message_id))
        return "open:test"

    async def resolution_for_event(self, event_id: str) -> ExplanationResolution | None:
        assert event_id == "discord:message:1"
        return self.resolution


class UnknownLexicon:
    def __init__(self) -> None:
        self.queued: list[str] = []

    async def analyze_word(self, word: str, **kwargs) -> WordAnalysis:
        del kwargs
        return WordAnalysis(
            surface_form=word,
            normalized_form=word.casefold(),
            lemma=None,
            part_of_speech=None,
            grammatical_features={},
            dictionary_meanings=(),
            style_labels=(),
            source="unknown",
            confidence=ConfidenceProfile(0.1, 0.0, 0.0, 0.0, 0.0),
            unknown_kind=UnknownKind.UNKNOWN_FORM,
        )

    async def semantic_terms(self, text: str, *, guild_id: int) -> tuple[str, ...]:
        del text, guild_id
        return ()

    async def find_similar_words(self, word: str, limit: int = 3) -> tuple[str, ...]:
        del word, limit
        return ()

    async def ensure_study_queue(self, current: Event, analysis: WordAnalysis) -> bool:
        del current
        self.queued.append(analysis.normalized_form)
        return True


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

    assert response is not None
    assert response != "che che che"
    assert "che" in response


async def test_direct_mentions_use_only_one_transport_fallback_when_generation_fails() -> None:
    current = event(mentioned=True, content="<@123> объясни штуку")
    service = SafeMentionReplyService(
        FailingRepository(),  # type: ignore[arg-type]
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )

    first = await service.prepare_guaranteed_reply(current)
    second = await service.prepare_guaranteed_reply(current)

    assert first is not None and first.fallback
    assert second is not None and second.fallback
    assert first.text == second.text == "Повтори запрос: сейчас ответ не отправился."
    assert first.text and second.text


def test_topic_continuity_rejects_an_unrelated_candidate() -> None:
    check = SafeMentionReplyService._topic_continuity_issue

    assert (
        check(
            "Бухта приключений зовёт корабли к далёкому острову.",
            prompt="Кто твоя мама?",
            dialogue_messages=("Мы говорим о семье.",),
            required_facts=(),
        )
        == "topic_discontinuity"
    )
    assert (
        check(
            "В человеческом смысле родителей у меня нет.",
            prompt="Кто твоя мама?",
            dialogue_messages=("Мы говорим о семье.",),
            required_facts=("Мама относится к родителям.",),
        )
        is None
    )


async def test_sent_identity_answer_is_logged_without_prescribing_its_name(monkeypatch) -> None:
    records: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        reply_module._identity_logger,
        "info",
        lambda *args, **kwargs: records.append(args),
    )
    current = event(mentioned=True, content="<@123> как тебя зовут?")
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
    reply = PreparedReply(
        text="Имя, полученное из генерации.",
        intent_id="fallback:test",
        tokens=("имя",),
        genome_signature="test",
        candidate_count=0,
        fallback=True,
    )

    await service.record_sent(
        current,
        response_message_id=7,
        reply=reply,
        created_at=NOW,
    )

    assert records
    assert "Имя, полученное из генерации." in records[0]


async def test_sent_unknown_fallback_opens_a_per_user_question_marker() -> None:
    questions = FakeOpenQuestions()
    current = event(mentioned=True, content="<@123> что такое стромба")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        open_questions=questions,  # type: ignore[arg-type]
        enabled=True,
    )
    reply = PreparedReply(
        text="«стромба» мне пока незнакомо.",
        intent_id="fallback:test",
        tokens=("стромба",),
        genome_signature="test",
        candidate_count=0,
        fallback=True,
        pending_concept="стромба",
    )

    await service.record_sent(
        current,
        response_message_id=700,
        reply=reply,
        created_at=NOW,
    )

    assert questions.opened == [("стромба", 700)]


async def test_incoherent_reply_is_not_saved_as_successful_language_experience() -> None:
    class RecordingExperience(FakeExperienceConsolidator):
        def __init__(self) -> None:
            self.registered = 0

        async def register_sent(self, *args, **kwargs) -> None:
            del args, kwargs
            self.registered += 1

    experience = RecordingExperience()
    current = event(mentioned=True, content="<@123> продолжай")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        experience,  # type: ignore[arg-type]
        INITIATIVE,
        MemoryPhraseGenerator(),
        enabled=True,
    )
    reply = PreparedReply(
        text=". затем ошибку разобрать?.",
        intent_id="intent:test",
        tokens=("затем", "ошибку", "разобрать"),
        genome_signature="test",
        candidate_count=1,
        response_source="arbitrated:graph",
    )

    await service.record_sent(
        current,
        response_message_id=701,
        reply=reply,
        created_at=NOW,
    )

    assert experience.registered == 0


async def test_unknown_generation_queues_learning_without_opening_a_user_question() -> None:
    lexicon = UnknownLexicon()
    current = event(mentioned=True, content="<@123> стромба")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        lexicon=lexicon,  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_reply(current)

    assert prepared is not None
    assert prepared.pending_concept is None
    assert "стромба" in prepared.text
    assert "самостоятельный разбор" not in prepared.text
    assert "очеред" not in prepared.text
    assert lexicon.queued == ["стромба"]


async def test_explicit_definition_request_can_still_open_a_word_question() -> None:
    lexicon = UnknownLexicon()
    current = event(mentioned=True, content="<@123> что такое стромба")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        lexicon=lexicon,  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_reply(current)

    assert prepared is not None
    assert prepared.pending_concept == "стромба"
    assert lexicon.queued == ["стромба"]


async def test_sentence_gap_searches_and_queues_multiple_terms_before_fallback() -> None:
    class TrackingReference:
        def __init__(self) -> None:
            self.lookups: list[str] = []

        async def refresh_word(self, word: str) -> None:
            self.lookups.append(word)
            return None

    lexicon = UnknownLexicon()
    reference = TrackingReference()
    current = event(mentioned=True, content="<@123> фырба крутит стромбу")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        lexicon=lexicon,  # type: ignore[arg-type]
        russian_reference=reference,  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_reply(current)

    assert prepared is not None
    assert prepared.pending_concept is None
    assert len(reference.lookups) == 3
    assert set(reference.lookups) == set(lexicon.queued)
    assert "самостоятельный разбор" not in prepared.text
    assert "очеред" not in prepared.text


async def test_bound_explanation_gets_an_explicit_contextual_confirmation() -> None:
    resolution = ExplanationResolution(
        question_id="question:1",
        concept_key="сужу",
        explanation="сужу — это от слова суд",
        relevance_reason="explanation_trigger",
        confirmation="Понял: «сужу» связано со словом «суд».",
    )
    service = SafeMentionReplyService(
        FakeRepository(None),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        open_questions=FakeOpenQuestions(resolution),  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_bound_explanation(
        event(mentioned=False, content="сужу — это от слова суд")
    )

    assert prepared is not None
    assert prepared.contextual_learning
    assert prepared.text == resolution.confirmation


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
    assert "red button drops server" not in response
    assert "continue" in response


async def test_service_uses_learned_reply_head_without_loading_raw_archive() -> None:
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
    assert "learned" in response
    assert "continues topic" not in response


async def test_service_blocks_internal_diagnostic_language() -> None:
    current = event(mentioned=True, content="<@123> ты разумный?")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Вижу удачный исход и повышаю доверие."),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response is not None
    assert "несвязным" not in response
    assert "разум" in response


async def test_service_does_not_repeat_its_last_reply_verbatim() -> None:
    previous_bot = event(
        mentioned=False,
        author_is_bot=True,
        content="Одинаковый ответ.",
        event_id="discord:message:bot",
        actor_id=999,
    )
    current = event(mentioned=True, content="<@123> а почему?")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(
            user_events=(current,),
            channel_events=(previous_bot, current),
        ),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Одинаковый ответ."),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response != "Одинаковый ответ."
    assert "несвязным" not in response


async def test_explicit_knowledge_answer_beats_unrelated_learned_fragments() -> None:
    current = event(mentioned=True, content="<@123> что такое мороженое")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(learned_replies=("это сказать", "нехватке говорю")),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        knowledge_service=FakeKnowledgeService(),  # type: ignore[arg-type]
        enabled=True,
    )

    response = await service.response_for(current)

    assert response == "Мороженое — замороженный сладкий десерт."


async def test_explicit_word_meaning_uses_reference_before_surface_generation() -> None:
    current = event(mentioned=True, content="<@123> узнай что такое кринж")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Я понял, что речь о кринже, но связь неоднозначна."),
        russian_reference=FakeRussianReference(),  # type: ignore[arg-type]
        enabled=True,
    )

    response = await service.response_for(current)

    assert response == "«кринж» (молодёжное) — чувство стыда за чужие действия."


async def test_explicit_lookup_rejects_a_poisoned_personal_placeholder() -> None:
    class ClownessReference:
        async def refresh_word(self, word: str) -> RemoteLexeme | None:
            assert word == "клоунесса"
            return RemoteLexeme(
                word="клоунесса",
                part_of_speech="NOUN",
                meanings=(
                    "женское название лица, соответствующее слову «клоун»; "
                    "базовое словарное значение: комический артист",
                ),
                source_url="https://ru.wiktionary.org/wiki/клоун",
                style_labels=("словообразовательный вывод",),
                derived_from="клоун",
            )

    poisoned = WordAnalysis(
        surface_form="клоунесса",
        normalized_form="клоунесса",
        lemma="клоунесса",
        part_of_speech=None,
        grammatical_features={},
        dictionary_meanings=(),
        style_labels=(),
        source="personal_lexicon",
        confidence=ConfidenceProfile(0.72, 0.72, 0.55, 0.72, 0.52),
        selected_meaning="Есть отдельное значение",
        personal_interpretation="Есть отдельное значение",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        russian_reference=ClownessReference(),  # type: ignore[arg-type]
        enabled=True,
    )

    response = await service._lexical_reference_reply("Что такое клоунесса", (poisoned,))

    assert response is not None
    assert "словообразовательный вывод" in response
    assert "комический артист" in response
    assert "Есть отдельное значение" not in response


async def test_cached_derived_reference_keeps_its_public_source_label() -> None:
    cached = WordAnalysis(
        surface_form="клоунесса",
        normalized_form="клоунесса",
        lemma="клоунесса",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=("женское название лица по слову «клоун»",),
        style_labels=("wiktionary_reference", "словообразовательный вывод"),
        source="dictionary_cache",
        confidence=ConfidenceProfile(0.95, 0.76, 0.8, 0.75, 0.76),
        selected_meaning="женское название лица по слову «клоун»",
    )
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

    response = await service._lexical_reference_reply("Что такое клоунесса", (cached,))

    assert response is not None
    assert "словообразовательный вывод" in response
    assert "wiktionary_reference" not in response


def test_surface_gate_rejects_technical_tokens_prompt_echo_and_fragments() -> None:
    check = SafeMentionReplyService._surface_quality_issue

    assert (
        check("Я понял. <user> узнай умник.", prompt="узнай умник", recent_self_replies=())
        == "technical_token_leak"
    )
    assert (
        check("Ну хотя бы предположи.", prompt="хотя бы предположи", recent_self_replies=())
        == "prompt_echo"
    )
    assert (
        check("это сказать. нехватке говорю.", prompt="что такое мороженое", recent_self_replies=())
        == "broken_fragment"
    )
    assert (
        check("), если стороны девушка перед хорошо.", prompt="или что", recent_self_replies=())
        == "broken_opening"
    )
    assert (
        check(
            "темы. уточни, что такое мама.",
            prompt="ты сверил маму?",
            recent_self_replies=(),
        )
        == "broken_opening_fragment"
    )
    for primitive in ("Да.", "Нет!", "Сейчас будет", "Не хочу."):
        assert (
            check(primitive, prompt="что ты решил", recent_self_replies=())
            == "bare_stance_without_reason"
        )
    assert (
        check("ты думаешь", prompt="Почему ты так думаешь", recent_self_replies=())
        == "partial_prompt_echo"
    )
    assert (
        check("трощкэ закидываешь", prompt="закинул трощкэ?", recent_self_replies=())
        == "partial_prompt_echo"
    )


async def test_social_routine_bypasses_stalled_surface_generation() -> None:
    current = event(mentioned=True, content="<@123> как дела?")
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Ответ про дело получился несвязным."),
        enabled=True,
    )

    response = await service.response_for(current)

    assert response is not None
    assert response == "Я в работе. А ты как?"
    assert "несвязным" not in response


async def test_second_person_predicates_are_answered_without_public_parse_failure() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Не совсем понял, скажи подробнее."),
        enabled=True,
    )

    smart = await service.response_for(event(mentioned=True, content="<@123> ты ведь умный?"))
    reversed_label = await service.response_for(event(mentioned=True, content="<@123> мордобой ты"))

    assert smart is not None and "оцениваешь меня" in smart
    assert reversed_label is not None and "оцениваешь меня" in reversed_label
    assert "на каком ответе" in smart.casefold()
    assert "на каком ответе" in reversed_label.casefold()
    assert "не совсем понял" not in smart.casefold()
    assert "не совсем понял" not in reversed_label.casefold()


async def test_active_arbitration_answers_preference_instead_of_requesting_an_action() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(exact_reply="Футбол — командная игра с мячом."),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("По «футбол» не хватает самого действия или вопроса."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> А футбол любишь?"))

    assert prepared is not None
    assert "футбол" in prepared.text.casefold()
    assert "не хватает" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_thought"
    assert prepared.arbitration_decision_id is not None
    assert prepared.candidate_count > 1


async def test_complete_elliptical_statement_does_not_trigger_clarification() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Просто ключи взял бы")
    )

    assert prepared is not None
    assert prepared.text == "Я понял тебя."
    assert prepared.response_source == "arbitrated:dialogue_act"
    assert prepared.fallback is False


async def test_relevant_grounded_fact_can_answer_a_question() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator("Ключ лежит в ящике."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Где лежит ключ?"))

    assert prepared is not None
    assert prepared.text == "Ключ лежит в ящике."
    assert prepared.response_source == "arbitrated:grounded_surface"
    assert prepared.fallback is False


async def test_dialogue_fragment_does_not_dump_a_dictionary_fact_from_memory() -> None:
    definition = "Ответ — реплика, вызванная заданным вопросом или реакция на какое-либо событие."
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator(definition),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Твой ответ"))

    assert prepared is not None
    assert prepared.text == "Я понял тебя."
    assert prepared.response_source == "arbitrated:dialogue_act"
    assert definition not in prepared.text


async def test_statement_does_not_replay_an_unrelated_memory_surface() -> None:
    memory_fragment = "Большая база даст тебе."
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator(memory_fragment),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Так ты же уже дал оценку")
    )

    assert prepared is not None
    assert prepared.text == "Я понял тебя."
    assert prepared.response_source == "arbitrated:dialogue_act"
    assert memory_fragment not in prepared.text


async def test_compound_study_and_opinion_request_returns_both_results() -> None:
    class SkyReference:
        async def refresh_word(self, word: str) -> RemoteLexeme | None:
            assert word == "небо"
            return RemoteLexeme(
                "небо",
                "NOUN",
                ("пространство над поверхностью Земли",),
                "https://ru.wiktionary.org/wiki/небо",
                (),
            )

    current = event(
        mentioned=True,
        content="<@123> изучи, что такое небо, и скажи своё мнение",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Небо находится сверху."),
        russian_reference=SkyReference(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(current)

    assert prepared is not None
    assert "пространство над поверхностью земли" in prepared.text.casefold()
    assert "люблю небо" in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_synthesis"


async def test_active_arbitration_does_not_turn_a_personal_label_into_a_tv_series() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(exact_reply="Умник — американский телесериал 1987 года."),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Умник — американский телесериал 1987 года."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Ты что умник теперь?")
    )

    assert prepared is not None
    assert "оцениваешь меня" in prepared.text.casefold()
    assert "на каком ответе" in prepared.text.casefold()
    assert "телесериал" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:dialogue_rule"


async def test_active_arbitration_does_not_invent_a_named_subject_for_answer_reference() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("умник ответил на вопрос."),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Ответил на вопрос")
    )

    assert prepared is not None
    assert "ответ" in prepared.text.casefold()
    assert "вопрос" in prepared.text.casefold()
    assert "умник" not in prepared.text.casefold()
    assert "борис" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:dialogue_rule"


async def test_active_arbitration_rejects_logged_graph_garbage_and_uses_fallback() -> None:
    garbage = "узнай?. затем ошибку разобрать. значение слова зависит от контекста: оно может быть."
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(garbage),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> продолжай разговор")
    )

    assert prepared is not None
    assert prepared.text != garbage
    assert prepared.fallback
    assert prepared.response_source == "arbitrated:fallback"


async def test_causal_question_rejects_remembered_topic_phrase_and_composes_effect() -> None:
    class CompositionalCausalReasoning:
        async def infer(self, guild_id: int, query):
            assert guild_id == 1
            return infer_compositional_consequence(query)

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("книга принадлежит борису"),
        causal_reasoning=CompositionalCausalReasoning(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> что будет, если толкнуть книгу со стола?",
        )
    )

    assert prepared is not None
    assert prepared.text == "Книга, скорее всего, упадёт."
    assert "борис" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:causal_reasoning"


async def test_self_location_is_composed_without_reusing_unrelated_location_fact() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("медный ключ находится у ворона"),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> где ты находишься?")
    )

    assert prepared is not None
    assert prepared.text == "Я работаю в Discord."
    assert "ворон" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_affective_state_is_composed_without_a_no_data_fallback() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Я не могу подтвердить это."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> что ты чувствуешь?")
    )

    assert prepared is not None
    assert prepared.text == "Я чувствую интерес к разговору."
    assert "не могу" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_threat_response_never_falls_back_to_missing_monster_data() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Достаточных данных нет у меня о монстре."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Не боишься болотного монстра?")
    )

    assert prepared is not None
    assert prepared.text in {
        "Да, я не боюсь болотного монстра.",
        "Нет, я не боюсь болотного монстра.",
    }
    assert "достаточ" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_evaluation_of_previous_answer_cannot_leak_definition_through_fallback() -> None:
    definition = (
        "Ответ — реплика, вызванная вопросом или реакция на событие. Если нужно подробнее — уточни."
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(exact_reply=definition),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(definition),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> твой ответ логичный, прикинь")
    )

    assert prepared is not None
    assert prepared.text == "Да, я стараюсь."
    assert "реплика" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:dialogue_act"


async def test_active_arbitration_handles_attention_call_before_graph_generation() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайную ошибку разобрать?."),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> ауу"))

    assert prepared is not None
    assert prepared.text == "Я здесь. Говори."
    assert prepared.response_source == "arbitrated:dialogue_rule"


async def test_active_arbitration_can_answer_from_grounded_meaning_plan() -> None:
    class FactOnlyKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            assert current.event_id == "discord:message:1"
            assert "лиса" in prompt
            return ("Лиса взяла красный ключ со стола.",), None

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайный обрывок разобрать?."),
        knowledge_service=FactOnlyKnowledge(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> что сделала лиса?")
    )

    assert prepared is not None
    assert prepared.text == "Лиса взяла красный ключ со стола."
    assert prepared.response_source == "arbitrated:meaning_plan"
    assert not prepared.fallback


async def test_active_arbitration_keeps_coherent_multi_fact_meaning_plan() -> None:
    class FactOnlyKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            assert current.event_id == "discord:message:1"
            assert "дракон" in prompt
            return (
                (
                    "Дракон — мифическое существо с крыльями.",
                    "Дракон живёт в горах.",
                ),
                None,
            )

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайный обрывок разобрать?."),
        knowledge_service=FactOnlyKnowledge(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> расскажи о драконе")
    )

    assert prepared is not None
    assert prepared.text == ("Дракон — мифическое существо с крыльями. Он живёт в горах.")
    assert prepared.response_source == "arbitrated:meaning_plan"
    assert not prepared.fallback


async def test_context_reference_expands_retrieval_without_rewriting_user_prompt() -> None:
    class ContextAwareKnowledge:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            assert current.event_id == "discord:message:1"
            self.prompts.append(prompt)
            if "волк" in prompt:
                return ("Волк живёт в лесу.",), None
            return (), None

    previous = event(
        mentioned=False,
        content="Лиса увидела волка.",
        event_id="discord:message:0",
    )
    knowledge = ContextAwareKnowledge()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайный обрывок разобрать?."),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> где он живёт?"))

    assert prepared is not None
    assert knowledge.prompts == ["где он живёт?", "где он живёт? волк"]
    assert prepared.text == "Волк живёт в лесу."
    assert prepared.response_source == "arbitrated:meaning_plan"


async def test_ambiguous_reference_competes_as_a_clarification_candidate() -> None:
    previous = event(
        mentioned=False,
        content="Дракон встретил рыцаря.",
        event_id="discord:message:0",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайный обрывок разобрать?."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> он ушёл?"))

    assert prepared is not None
    assert prepared.text == "Под «он» имеется в виду «дракон» или «рыцарь»?"
    assert prepared.response_source == "arbitrated:dialogue_context"


async def test_graph_cannot_replace_human_capability_subject_with_bot() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("я умею смысл."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> как думаешь? астрономы умеют летать?")
    )

    assert prepared is not None
    assert prepared.text == "Нет, астрономы не могут летать."
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_bot_physical_capability_question_gets_grounded_self_answer() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("я умею смысл."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> а ты умеешь летать?")
    )

    assert prepared is not None
    assert prepared.text == "Нет, я не умею летать."
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_bot_external_capability_generalizes_beyond_seeded_actions() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("я умею смысл."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> а ты умеешь копать землю?")
    )

    assert prepared is not None
    assert prepared.text in {
        "Нет, я не умею копать землю.",
        "Нет, я не умею землю копать.",
    }
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_unknown_capability_awaits_ai_research_before_arbitration() -> None:
    class CapabilityTeacher:
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.completed_terms: list[str] = []

        async def research_lexeme(self, **kwargs) -> tuple[str, ...]:
            term = kwargs["term"]
            assert "ты умеешь устраивать рейды?" in kwargs["context_messages"][-1]
            await asyncio.sleep(0.01)
            self.completed_terms.append(term)
            return ("Рейд — организованная групповая атака с участниками.",)

        def should_consult(self, **kwargs) -> bool:
            assert {"устраивать", "рейд"} <= set(self.completed_terms)
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs):
            del kwargs
            raise AssertionError("The awaited research must bind a local answer")

    teacher = CapabilityTeacher()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("У меня недостаточно данных."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        semantic_consultant=teacher,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты умеешь устраивать рейды?")
    )

    assert prepared is not None
    assert set(teacher.completed_terms) == {"устраивать", "рейд"}
    assert prepared.text == "Нет, я не умею устраивать рейды."
    assert "недостаточно" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_nonmodal_self_activity_awaits_research_instead_of_missing_data() -> None:
    class ActivityTeacher:
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.completed_terms: list[str] = []

        async def research_lexeme(self, **kwargs) -> tuple[str, ...]:
            term = kwargs["term"]
            assert "ты у себя пасёшь овечки?" in kwargs["context_messages"][-1]
            await asyncio.sleep(0.01)
            self.completed_terms.append(term)
            return ("Пасти овец — присматривать за стадом животных во время выпаса.",)

        def should_consult(self, **kwargs) -> bool:
            assert {"пасти", "овечка"} <= set(self.completed_terms)
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs):
            del kwargs
            raise AssertionError("Research must bind the self-activity answer")

    teacher = ActivityTeacher()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Достаточных данных нет у меня об овечке."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        semantic_consultant=teacher,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты у себя пасёшь овечки?")
    )

    assert prepared is not None
    assert set(teacher.completed_terms) == {"пасти", "овечка"}
    assert prepared.text == "Нет, я не пасу овечки."
    assert "достаточ" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_reasoning"


@pytest.mark.parametrize(
    ("prompt", "fact"),
    (
        ("<@123> Сколько сердец у осьминога?", "У осьминога три сердца."),
        (
            "<@123> Чем кварк отличается от электрона?",
            "Кварк участвует в сильном взаимодействии, а электрон является лептоном.",
        ),
        (
            "<@123> Зачем пчёлы исполняют танец?",
            "Пчёлы танцем передают направление и расстояние до источника пищи.",
        ),
    ),
)
async def test_any_unbound_question_runs_whole_request_research_before_fallback(
    prompt: str,
    fact: str,
) -> None:
    class WholeRequestTeacher:
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.researched = False

        def should_consult(self, **kwargs) -> bool:
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs) -> ConsultationResult:
            del kwargs
            return ConsultationResult.from_payload(
                {
                    "interpretation": "Пользователь задаёт фактический вопрос.",
                    "intent": "fact_question",
                    "facts": [],
                    "confidence": 0.82,
                },
                provider="fake-puter",
                model="fake-model",
            )

        async def research_question(self, **kwargs) -> tuple[str, ...]:
            assert kwargs["prompt"].casefold() in prompt.casefold()
            await asyncio.sleep(0.01)
            self.researched = True
            return (fact,)

    teacher = WholeRequestTeacher()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("У меня недостаточно данных."),
        semantic_consultant=teacher,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))

    assert prepared is not None
    assert teacher.researched
    assert prepared.text == fact
    assert "недостаточ" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_synthesis"


async def test_consulted_opinion_reframes_the_contract_instead_of_surfacing_definitions() -> None:
    class OpinionTeacher:
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs) -> ConsultationResult:
            del kwargs
            return ConsultationResult.from_payload(
                {
                    "interpretation": "Пользователь просит оценить страну.",
                    "intent": "opinion_question",
                    "expected_response_acts": ["answer", "answer_first"],
                    "social_frame": {
                        "prompt_act": "question",
                        "speaker_stance": "neutral",
                        "emotion": "curious",
                        "response_obligation": "answer",
                        "recommended_moves": ["answer_first"],
                        "avoid_moves": ["define_isolated_word"],
                        "directness": 1.0,
                        "warmth": 0.0,
                        "playfulness": 0.0,
                    },
                    "facts": [],
                    "confidence": 0.82,
                },
                provider="fake-puter",
                model="fake-model",
            )

        async def research_question(self, **kwargs) -> tuple[str, ...]:
            del kwargs
            return (
                "Германия — государство в Центральной Европе.",
                "Крутой — разговорное обозначение высокой оценки.",
            )

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("У меня недостаточно данных о стране."),
        semantic_consultant=OpinionTeacher(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Германия крутая страна?")
    )

    assert prepared is not None
    assert prepared.text == "Нет, я пока не считаю Германию крутой."
    assert "недостаточ" not in prepared.text.casefold()
    assert "государство" not in prepared.text.casefold()
    assert "обозначение" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:position_reasoning"


async def test_generic_no_data_surface_is_rejected_for_any_unresolved_question() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("У меня нет достаточных данных о квазаре."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Почему у квазара нет тени?")
    )

    assert prepared is not None
    assert prepared.text in {
        "Причина остаётся загадкой.",
        "Я не могу ответить на этот вопрос.",
    }
    assert "достаточ" not in prepared.text.casefold()
    assert "данн" not in prepared.text.casefold()


async def test_bot_tool_use_question_beats_missing_topic_data_fallback() -> None:
    class ToolLexicon:
        def __init__(self) -> None:
            self.learned = False
            self.queued: list[tuple[str, ...]] = []

        async def analyze_word(self, word: str, **kwargs) -> WordAnalysis:
            del kwargs
            normalized = word.casefold()
            is_object = normalized == "кусачками"
            return WordAnalysis(
                surface_form=word,
                normalized_form=normalized,
                lemma="кусачки" if is_object else "пользоваться",
                part_of_speech="NOUN" if is_object else "VERB",
                grammatical_features=(
                    {"case": "instrumental", "number": "plural"}
                    if is_object
                    else {"tense": "present", "person": "second"}
                ),
                dictionary_meanings=(
                    ("ручной шарнирно-губцевый инструмент для резки материала",)
                    if is_object and self.learned
                    else ("использовать некоторый ресурс",)
                    if not is_object
                    else ()
                ),
                style_labels=(),
                source="dictionary_cache" if self.learned else "opencorpora",
                confidence=ConfidenceProfile(0.9, 0.9, 0.9, 0.9, 0.9),
                unknown_kind=(
                    UnknownKind.UNKNOWN_MEANING if is_object and not self.learned else None
                ),
            )

        async def queue_semantic_study(
            self,
            current: Event,
            terms: tuple[str, ...],
            *,
            reason: str,
            priority: float,
        ) -> int:
            del current, reason, priority
            self.queued.append(terms)
            return len(terms)

    class ToolKnowledgeService:
        def __init__(self, lexicon: ToolLexicon) -> None:
            self.lexicon = lexicon
            self.study_calls: list[tuple[str, ...]] = []

        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            del current, prompt
            return (), None

        async def internalize_for_reasoning(
            self,
            current: Event,
            terms: tuple[str, ...],
            *,
            limit: int = 6,
        ) -> tuple[SimpleNamespace, ...]:
            del current, limit
            self.study_calls.append(terms)
            self.lexicon.learned = True
            return (
                SimpleNamespace(
                    title="кусачки",
                    definition=("ручной шарнирно-губцевый инструмент для резки материала"),
                ),
            )

    lexicon = ToolLexicon()
    knowledge = ToolKnowledgeService(lexicon)
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Достаточных данных нет у меня о кусачках."),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        lexicon=lexicon,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты кусачками пользуешься?")
    )

    assert prepared is not None
    assert prepared.text == "Нет, я не пользуюсь кусачками."
    assert prepared.response_source == "arbitrated:semantic_reasoning"
    assert "достаточ" not in prepared.text.casefold()
    assert knowledge.study_calls == [("кусачки",)]


async def test_self_experience_contract_beats_a_shallow_memory_match() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator("свои игрульки играешь"),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты играл в гта 5?")
    )

    assert prepared is not None
    assert prepared.text == "Нет, я не играл в гта 5."
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_hypothetical_desire_uses_previous_topic_instead_of_missing_data() -> None:
    previous = event(
        mentioned=False,
        content="Играешь в гта 5?",
        event_id="discord:message:0",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Интересно. А хотел бы сыграть?")
    )

    assert prepared is not None
    assert prepared.text == "Да, я хочу сыграть в гта 5."
    assert prepared.response_source == "arbitrated:semantic_reasoning"


async def test_elliptical_desire_does_not_surface_a_dictionary_definition() -> None:
    previous_question = event(
        mentioned=False,
        content="ты занимался кунг-фу?",
        event_id="discord:message:0",
    )
    previous_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Нет, я не занимался.",
        event_id="discord:message:00",
        actor_id=123,
    )

    class DictionaryTeacher:
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.research_calls = 0

        def should_consult(self, **kwargs) -> bool:
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs) -> ConsultationResult:
            del kwargs
            raise AssertionError("a locally bound contextual answer must win")

        async def research_question(self, **kwargs) -> tuple[str, ...]:
            del kwargs
            self.research_calls += 1
            return ("Хотеть — глагол, обозначающий желание.",)

    teacher = DictionaryTeacher()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_question, previous_answer)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("У меня недостаточно данных."),
        semantic_consultant=teacher,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> А хотел бы?"))

    assert prepared is not None
    assert prepared.text == "Да, я хочу заниматься кунг-фу."
    assert prepared.response_source == "arbitrated:semantic_reasoning"
    assert teacher.research_calls == 0


async def test_study_directive_uses_retrieved_answer_not_an_old_phrase() -> None:
    class GtaKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            del current
            assert "гта 5" in prompt.casefold()
            answer = "Grand Theft Auto V — компьютерная игра."
            return (answer,), answer

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator("ты изучи такое небо"),
        knowledge_service=GtaKnowledge(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> Изучи что такое гта 5 и ответь",
        )
    )

    assert prepared is not None
    assert prepared.text == "Grand Theft Auto V — компьютерная игра."
    assert prepared.response_source == "arbitrated:knowledge_retrieval"


async def test_failed_references_fall_back_to_ai_research_and_store_the_fact() -> None:
    class EmptyKnowledge:
        def __init__(self) -> None:
            self.persisted: list[tuple[str, str]] = []

        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            del current, prompt
            return (), None

        async def internalize_for_reasoning(
            self,
            current: Event,
            terms: tuple[str, ...],
            *,
            limit: int = 6,
        ) -> tuple[SimpleNamespace, ...]:
            del current, terms, limit
            return ()

        async def internalize_consulted_fact(
            self,
            current: Event,
            *,
            query: str,
            definition: str,
            confidence: float,
        ) -> None:
            del current, confidence
            self.persisted.append((query, definition))

    class ResearchTeacher:
        trigger_threshold = 0.72

        async def research_lexeme(self, **kwargs) -> tuple[str, ...]:
            assert kwargs["term"] == "рэпер"
            return ("Рэпер — исполнитель рэпа, работающий с ритмическим речитативом.",)

        def should_consult(self, **kwargs) -> bool:
            return not kwargs["has_bound_answer"]

        async def consult(self, **kwargs):
            raise AssertionError("The researched fact must bind the answer locally")

    knowledge = EmptyKnowledge()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        GroundedOnlyGenerator("Что ты хочешь сказать про что?"),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        semantic_consultant=ResearchTeacher(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Узнай что такое рэпер")
    )

    assert prepared is not None
    assert "исполнитель рэпа" in prepared.text
    assert "недостаточ" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_synthesis"
    assert knowledge.persisted == [
        (
            "рэпер",
            "Рэпер — исполнитель рэпа, работающий с ритмическим речитативом.",
        )
    ]


async def test_abbreviation_meaning_answers_when_external_lookup_is_unavailable() -> None:
    class AbbreviationLexicon:
        async def analyze_word(self, word: str, **kwargs) -> WordAnalysis:
            del kwargs
            normalized = word.casefold()
            abbreviation = normalized == "гта"
            meaning = "сокращение от «Grand Theft Auto»"
            return WordAnalysis(
                surface_form=word,
                normalized_form=normalized,
                lemma=normalized,
                part_of_speech="NOUN" if abbreviation else None,
                grammatical_features=({"abbreviation": "yes"} if abbreviation else {}),
                dictionary_meanings=(meaning,) if abbreviation else (),
                style_labels=("abbreviation",) if abbreviation else (),
                source="builtin_abbreviation" if abbreviation else "opencorpora",
                confidence=ConfidenceProfile(0.98, 0.96, 0.9, 0.9, 0.9),
                selected_meaning=meaning if abbreviation else None,
            )

        async def queue_semantic_study(self, *args, **kwargs) -> int:
            del args, kwargs
            return 0

    for prompt in (
        "<@123> Ты знаешь что такое гта 5?",
        "<@123> Изучи что такое гта 5",
    ):
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            GroundedOnlyGenerator("Что ты хочешь сказать про что?"),
            lexicon=AbbreviationLexicon(),  # type: ignore[arg-type]
            arbitration_mode="active",
            arbitration_minimum_score=0.0,
            enabled=True,
        )

        prepared = await service.prepare_reply(event(mentioned=True, content=prompt))

        assert prepared is not None
        assert "Grand Theft Auto" in prepared.text
        assert "недостаточ" not in prepared.text.casefold()
        assert prepared.response_source == "arbitrated:lexical_reference"


async def test_answer_gap_arbitrates_meaning_moves_instead_of_one_stock_sentence() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("удаления скажи ключ"),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> кто тут шалава?"))

    assert prepared is not None
    assert "шалав" in prepared.text.casefold()
    assert "данн" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:fallback"
    assert prepared.candidate_count >= 3


async def test_relation_question_studies_ambiguous_word_then_answers_from_subject_model() -> None:
    knowledge = RelationKnowledgeService()
    lexicon = RelationLexicon()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("команды узнай названного объекта."),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        lexicon=lexicon,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Хочу узнать, есть ли у тебя носки")
    )

    assert prepared is not None
    assert prepared.text.startswith(("Нет,", "Нет."))
    assert "у меня нет носков" in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:relation_reasoning"
    assert knowledge.study_calls
    assert {"носка", "носок"} <= set(knowledge.study_calls[0])


async def test_relation_reasoning_generalizes_to_another_physical_object() -> None:
    knowledge = RelationKnowledgeService()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("значения и памяти. но не хватает фактов."),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> У тебя имеется куртка?")
    )

    assert prepared is not None
    assert prepared.text.startswith("Нет,")
    assert prepared.response_source == "arbitrated:relation_reasoning"


async def test_relation_reasoning_distinguishes_internal_from_physical_resources() -> None:
    knowledge = RelationKnowledgeService()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("тема в памяти пока не хватает фактов."),
        knowledge_service=knowledge,  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Есть ли у тебя память?")
    )

    assert prepared is not None
    assert prepared.text.startswith("Да,")
    assert "у меня есть память" in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:relation_reasoning"


async def test_repeated_relation_question_keeps_the_bound_answer() -> None:
    previous_question = event(
        mentioned=True,
        content="<@123> У тебя имеется куртка?",
        event_id="discord:message:previous-question",
    )
    previous_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Нет, у меня нет куртки.",
        event_id="discord:message:previous-answer",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_question, previous_answer)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("случайный графовый обрывок."),
        knowledge_service=RelationKnowledgeService(),  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> У тебя имеется куртка?")
    )

    assert prepared is not None
    assert prepared.text in {
        "Нет, у меня нет куртки.",
        "Нет. У меня нет куртки.",
    }
    assert prepared.response_source == "arbitrated:relation_reasoning"


async def test_self_state_question_is_answered_from_local_state_without_reply_template() -> None:
    previous = event(
        mentioned=True,
        content="<@123> Идея в том, чтобы сначала собрать мысль, а потом строить фразу.",
        event_id="discord:message:previous-idea",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Вопрос по форме понятен, но ответ не собрался."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> А ты понял идею?"))

    assert prepared is not None
    assert prepared.text in {"Да, я понял идею.", "Да, идею я понял."}
    assert prepared.response_source == "arbitrated:semantic_thought"


@pytest.mark.parametrize(
    "prompt",
    (
        "<@123> какой у тебя уровень мышления?",
        "<@123> Так какой уровень мышления у тебя",
    ),
)
async def test_cognitive_level_question_never_falls_back_to_missing_data(
    prompt: str,
) -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))

    assert prepared is not None
    assert prepared.text == "Я считаю своё мышление развивающимся."
    assert prepared.response_source == "arbitrated:semantic_thought"
    assert prepared.fallback is False
    assert "недостаточ" not in prepared.text.casefold()
    assert "не могу" not in prepared.text.casefold()


@pytest.mark.parametrize(
    ("prompt", "expected", "source"),
    (
        (
            "<@123> а ты считаешь себя умным?",
            "Да, я считаю себя умным.",
            "arbitrated:semantic_thought",
        ),
        (
            "<@123> ты что мозгом поперхнулся?",
            "Я понял предыдущий вопрос неправильно.",
            "arbitrated:dialogue_act",
        ),
        (
            "<@123> А что тогда?",
            "Я понял предыдущий вопрос неправильно.",
            "arbitrated:dialogue_act",
        ),
        (
            "<@123> Бобун лох или нет? Да или нет? Как думаешь?",
            "Нет, я пока не считаю его лохом.",
            "arbitrated:position_reasoning",
        ),
        (
            "<@123> Лохи хорошие люди?",
            "Нет, я пока не считаю лохов хорошими.",
            "arbitrated:position_reasoning",
        ),
        (
            "<@123> Откроешь мне бутылку?",
            "Я не могу открыть бутылку.",
            "arbitrated:semantic_reasoning",
        ),
        (
            "<@123> Ты разбил зеркало?",
            "Нет, я не разбил зеркало.",
            "arbitrated:semantic_reasoning",
        ),
        (
            "<@123> Ты можешь без тела иметь мнение и позицию?",
            "Да, я умею иметь мнение.",
            "arbitrated:semantic_reasoning",
        ),
    ),
)
async def test_recent_conversational_failures_are_resolved_by_semantic_roles(
    prompt: str,
    expected: str,
    source: str,
) -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("случайный графовый обрывок."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))

    assert prepared is not None
    assert prepared.text == expected
    assert prepared.response_source == source


async def test_counterevidence_revises_previous_internal_capability_answer() -> None:
    previous_bad_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Нет, я не умею иметь мнение.",
        event_id="discord:message:previous-capability-answer",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_bad_answer,)),
        FakeArchiveRepository(learned_replies=("ты отвечай на вопрос",)),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("ты отвечай на вопрос"),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> Почему? Ты сейчас оцениваешь и отвечаешь мне",
        )
    )

    assert prepared is not None
    assert prepared.text == "Да, я могу иметь мнение."
    assert prepared.response_source == "arbitrated:dialogue_context"


async def test_epistemic_cause_identifies_overcautious_threshold() -> None:
    previous = event(
        mentioned=False,
        author_is_bot=True,
        content="Не определил я отношение к имени.",
        event_id="discord:message:previous-epistemic-answer",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Тела нет у меня."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Почему не определил?")
    )

    assert prepared is not None
    assert prepared.text == ("Я слишком осторожно отвечаю из-за завышенного порога.")
    assert prepared.response_source == "arbitrated:dialogue_context"


async def test_criticism_after_no_data_reply_explains_overcautious_threshold() -> None:
    previous = event(
        mentioned=False,
        author_is_bot=True,
        content="У меня нет достаточных данных.",
        event_id="discord:message:previous-no-data-answer",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Достаточных данных нет у меня о ответах."),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> Да почему ты такой пугливый в ответах?",
        )
    )

    assert prepared is not None
    assert prepared.text == ("Я слишком осторожно отвечаю из-за завышенного порога.")
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert "недостаточ" not in prepared.text.casefold()


async def test_misunderstanding_report_blocks_opposite_grounded_statement() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(learned_replies=("ты понял идею",)),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("ты понял идею"),
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Я не понял"))

    assert prepared is not None
    assert prepared.text == "Что осталось непонятным?"
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_meta_question_reprocesses_the_previous_question_semantically() -> None:
    previous_question = event(
        mentioned=True,
        content="<@123> У тебя имеется куртка?",
        event_id="discord:message:previous-question",
    )
    previous_bad_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Куртка — предмет верхней одежды.",
        event_id="discord:message:previous-answer",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_question, previous_bad_answer)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("По слову не хватает вопроса."),
        knowledge_service=RelationKnowledgeService(),  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Блять, але, я тебя спрашиваю")
    )

    assert prepared is not None
    assert prepared.text in {
        "Нет, у меня нет куртки.",
        "Нет. У меня нет куртки.",
    }
    assert prepared.response_source == "arbitrated:relation_reasoning"


async def test_last_resort_consultant_never_surfaces_its_ready_response() -> None:
    consultant = FakeSemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("команды узнай названного объекта."),
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Это база?"))

    assert prepared is not None
    assert prepared.text != "В этом контексте «база» — это согласие или одобрение."
    assert "пользователь" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_synthesis"
    assert len(consultant.calls) == 1


async def test_semantic_teacher_receives_role_labelled_dialogue_context() -> None:
    previous_user = event(
        mentioned=False,
        content="сервер опять сломался",
        event_id="discord:message:previous-user",
    )
    previous_agent = event(
        mentioned=False,
        author_is_bot=True,
        actor_id=123,
        content="Похоже, сервер недоступен.",
        event_id="discord:message:previous-agent",
    )
    consultant = FakeSemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_user, previous_agent)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("команды узнай названного объекта."),
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    await service.prepare_reply(event(mentioned=True, content="<@123> Это база?"))

    assert len(consultant.calls) == 1
    request = consultant.calls[0]["request"]
    assert request.context_messages == (
        "user: сервер опять сломался",
        "agent: похоже, сервер недоступен.",
    )


async def test_interpretation_without_facts_never_becomes_a_response_candidate() -> None:
    consultant = InterpretationOnlySemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Пользователь переспрашивает или просит."),
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Ты из Кваргонии?"))

    assert prepared is not None
    assert "пользователь" not in prepared.text.casefold()
    assert "агент" not in prepared.text.casefold()
    assert prepared.response_source != "arbitrated:semantic_synthesis"
    assert len(consultant.calls) == 1


async def test_consultant_is_not_called_when_local_relation_answer_is_bound() -> None:
    consultant = FakeSemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("команды узнай названного объекта."),
        knowledge_service=RelationKnowledgeService(),  # type: ignore[arg-type]
        lexicon=RelationLexicon(),  # type: ignore[arg-type]
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> У тебя есть носки?")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:relation_reasoning"
    assert consultant.calls == []


async def test_complete_alternative_question_is_answered_from_structure() -> None:
    consultant = FakeSemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("По «краб» не хватает самого действия или вопроса."),
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Краба выбираешь или рака?")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:semantic_reasoning"
    assert prepared.text in {"Я выбираю краба.", "Я выбираю рака."}
    assert consultant.calls == []


async def test_social_evaluation_is_bound_before_word_level_fallback() -> None:
    consultant = FakeSemanticConsultant()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("По «молодец» не хватает самого действия или вопроса."),
        semantic_consultant=consultant,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Молодец хуле"))

    assert prepared is not None
    assert prepared.response_source == "arbitrated:dialogue_act"
    assert prepared.text == "Да, я стараюсь."
    assert consultant.calls == []


async def test_social_understanding_challenge_never_uses_an_epistemic_gap() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> Ой бля, не делай вид, что ты не понял, ахахахах",
        )
    )

    assert prepared is not None
    assert prepared.text == "Да, я понял намёк."
    assert prepared.response_source == "arbitrated:dialogue_act"
    assert "данн" not in prepared.text.casefold()
    assert "не знаю" not in prepared.text.casefold()


async def test_hypothetical_spatial_question_uses_a_separate_mental_model() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> Представь: шар внутри куба. Где шар?",
        )
    )

    assert prepared is not None
    assert prepared.text == ("В этой воображаемой ситуации шар находится внутри куба.")
    assert prepared.response_source == "arbitrated:abstract_reasoning"


async def test_understood_first_person_state_never_leaks_a_technical_fallback() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> Просто удивляюсь"))

    assert prepared is not None
    assert prepared.text == "Чему ты удивляешься?"
    assert "не хватает" not in prepared.text.casefold()
    assert "вопроса" not in prepared.text.casefold()


async def test_answer_failure_retries_the_previous_question_instead_of_commenting_on_it() -> None:
    previous = event(
        mentioned=True,
        content="<@123> насколько ты считаешь себя осознанным?",
        event_id="discord:message:0",
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("умник ответил на вопрос."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты не ответил на вопрос")
    )

    assert prepared is not None
    assert prepared.text == "Нет, я не считаю себя осознанным."
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_consciousness_question_rejects_fluent_unrelated_graph_answer() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("умник ответил на вопрос."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> насколько считаешь ты себя осознанным?",
        )
    )

    assert prepared is not None
    assert prepared.text == "Нет, я не считаю себя осознанным."
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_open_self_reflection_uses_semantic_state_not_grammar_memory() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Правильно: «думаю о новом смысле». Прилагательное согласовано."),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> что ты думаешь о себе?")
    )

    assert prepared is not None
    assert prepared.text == "Я считаю себя обучающимся собеседником."
    assert prepared.response_source == "arbitrated:semantic_thought"


async def test_shadow_arbitration_keeps_the_legacy_text() -> None:
    legacy_text = "По «футбол» не хватает самого действия или вопроса."
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(legacy_text),
        arbitration_mode="shadow",
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> А футбол любишь?"))

    assert prepared is not None
    assert prepared.text == legacy_text
    assert prepared.response_source == "graph"
    assert prepared.arbitration_decision_id is None


def test_unrelated_generated_sentence_is_trimmed_from_answer_tail() -> None:
    trim = SafeMentionReplyService._trim_unrelated_tail

    assert (
        trim(
            "Чего ты хочешь. Тогда ответ Бориса к словам.",
            "чего",
        )
        == "Чего ты хочешь."
    )
    assert (
        trim(
            "Хорошо, не буду к нему лезть. Он останется в покое.",
            "не лезь к нему",
        )
        == "Хорошо, не буду к нему лезть. Он останется в покое."
    )


def test_fallback_repairs_pragmatic_cues_and_personal_claims() -> None:
    fallback = SafeMentionReplyService._fallback_for_plan
    plan = SimpleNamespace(key_points=())

    assert fallback(plan, (), prompt="Хмм") == "Что тебя смутило?"
    assert fallback(plan, (), prompt="Хехехех") == "Что тебя рассмешило?"
    assert fallback(plan, (), prompt="Я умный") == "Почему ты так считаешь?"


def test_fallback_answers_contextual_question_before_generic_epistemic_gap() -> None:
    fallback = SafeMentionReplyService._fallback_for_plan
    plan = SimpleNamespace(key_points=())

    assert (
        fallback(
            plan,
            (),
            prompt="А как ты тогда думаешь?",
            dialogue_messages=("Нет, у меня нет сознания.",),
        )
        == "Я анализирую контекст."
    )
    assert fallback(plan, (), prompt="Где ключ?") == "Я не могу найти ключ."


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
            learned_replies=(
                "server behaves strangely after update",
                "why server falls again tonight?",
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
