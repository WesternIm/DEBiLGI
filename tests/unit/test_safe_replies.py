import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import debilgi.discord.replies as reply_module
from debilgi.autonomy.models import InitiativeDecision
from debilgi.cognition.dialogue_tasks import DialogueTaskRepository
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.discord.replies import SafeMentionReplyService, _stable_response_seed
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.models import PreparedReply
from debilgi.knowledge.models import WikipediaResult
from debilgi.knowledge.repository import KnowledgeRepository
from debilgi.language.arbitration import ArbitrationProfile
from debilgi.language.assertion_experience import ExperienceGuidance
from debilgi.language.causal_reasoning import infer_compositional_consequence
from debilgi.language.conversation import ResponsePlan
from debilgi.language.lexicon import ConfidenceProfile, LexiconService, UnknownKind, WordAnalysis
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.open_questions import ExplanationResolution, OpenLanguageQuestionService
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.research_wait import LexicalResearchPending
from debilgi.language.russian_reference import RemoteLexeme, RussianReferenceService
from debilgi.language.ruwordnet import RuWordNet
from debilgi.language.semantic_consultant import ConsultationResult
from debilgi.language.sentence_understanding import UtteranceUnderstanding
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.storage.database import Database

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


async def test_dialogue_task_executes_open_ended_requested_utterance_without_script(tmp_path):
    from debilgi.experience.consolidation import ExperienceConsolidator
    from debilgi.experience.repository import ExperienceRepository

    database = Database(
        tmp_path / "dialogue-task-reply.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    try:
        prompt = event(mentioned=True, content="скажи да")
        await EventRepository(database).append(prompt)
        generator = Mock()
        generator.generate.side_effect = AssertionError("script generator must not run")
        tasks = DialogueTaskRepository(database)
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            ExperienceConsolidator(ExperienceRepository(database)),
            INITIATIVE,
            generator,
            dialogue_tasks=tasks,
            meaning_first_replies=True,
            enabled=True,
        )
        prepared = await service.prepare_guaranteed_reply(prompt)
        assert prepared is not None
        assert prepared.text == "Да."
        assert prepared.response_source == "arbitrated:dialogue_task"
        assert prepared.dialogue_task_id is not None
        await service.record_sent(
            prompt,
            response_message_id=999,
            reply=prepared,
            created_at=NOW + timedelta(seconds=1),
        )
        task = await tasks.task(prepared.dialogue_task_id)
        assert task is not None and task.status == "completed"
        generator.generate.assert_not_called()
    finally:
        await database.close()


@pytest.mark.parametrize("valued", [False, True])
async def test_preference_answers_require_feedback_on_a_real_sent_action(tmp_path, valued):
    from debilgi.experience.consolidation import ExperienceConsolidator
    from debilgi.experience.meaning_intents import MeaningIntentRepository
    from debilgi.experience.repository import ExperienceRepository
    from debilgi.language.proposition_memory import PropositionMemory

    database = Database(
        tmp_path / "intent-replies.sqlite3", Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    try:
        events = EventRepository(database)
        memory = PropositionMemory(database)
        learned = MeaningIntentRepository(database, exploration_rate=0)
        definition = replace(event(mentioned=True, event_id="definition", content=(
            "Быть выключенным для тебя, это когда ты не можешь отвечать мне, "
            "даже думать ты не можешь")), timestamp=NOW - timedelta(days=1))
        await events.append(definition)
        await memory.observe(definition)
        generator = Mock()
        generator.generate.side_effect = AssertionError("script generator must not run")
        consultant = SimpleNamespace(
            autonomous_planning=True, should_consult=Mock(), consult=AsyncMock(),
            plan_response=AsyncMock(),
        )
        service = SafeMentionReplyService(
            SimpleNamespace(intent_for_event=AsyncMock(return_value=intent())),
            FakeMemory(), FakeArchiveRepository(), EXPRESSION_ENGINE,
            ExperienceConsolidator(ExperienceRepository(database)), INITIATIVE, generator,
            semantic_consultant=consultant, proposition_memory=memory, meaning_intents=learned,
            meaning_first_replies=True, enabled=True,
        )
        prompt = replace(event(mentioned=True, event_id="factual-request",
                               content="Что значит быть выключенным?"),
                         timestamp=NOW - timedelta(minutes=1))
        await events.append(prompt)
        prepared = await service.prepare_guaranteed_reply(prompt)
        assert prepared and prepared.meaning_intention_id
        assert "не могу отвечать тебе" in prepared.text
        row = await database.fetchone("SELECT * FROM language_meaning_intentions")
        assert row["status"] == "prepared"
        assert not await learned.learned_goals(event(mentioned=True))
        if valued:
            await service.record_sent(
                prompt, response_message_id=999, reply=prepared,
                created_at=NOW - timedelta(seconds=50),
            )
            feedback = replace(event(mentioned=False, event_id="feedback"),
                timestamp=NOW - timedelta(seconds=40), event_type=EventType.REACTION_ADDED,
                payload={"message_id": 999, "emoji": "👍"})
            await events.append(feedback)
            assert await learned.observe(feedback)
        desire_prompt = event(mentioned=True, event_id="desire-query",
                              content="Ты хочешь быть выключенным?")
        await events.append(desire_prompt)
        reply = await service.prepare_guaranteed_reply(desire_prompt)
        assert reply is not None
        if valued:
            assert reply.text.startswith("Я не хочу быть выключенным")
            assert reply.meaning_intention_id
            assert reply.response_source == "arbitrated:semantic_reasoning"
        else:
            assert not reply.meaning_intention_id
            assert not reply.text.startswith("Я не хочу быть выключенным")
        generator.generate.assert_not_called()
        consultant.should_consult.assert_not_called()
        consultant.consult.assert_not_awaited()
        consultant.plan_response.assert_not_awaited()
    finally:
        await database.close()


@pytest.mark.parametrize("prompt,expected", [
    ("Что значит быть выключенным?", "Если я выключен"),
    ("Что будет, если ты выключен?", "Если я выключен"),
    ("Кто тупой?", "Гигик тупой."),
])
async def test_learned_semantic_relations_reach_delivery_without_scripts_or_llm(
    tmp_path, prompt, expected,
):
    from debilgi.language.proposition_memory import PropositionMemory

    database = Database(
        tmp_path / "meanings.sqlite3", Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    try:
        previous = replace(event(mentioned=True, event_id="definition", content=(
            "Быть выключенным для тебя, это когда ты не можешь отвечать мне, "
            "даже думать ты не можешь")), timestamp=NOW - timedelta(days=1))
        await EventRepository(database).append(previous)
        memory = PropositionMemory(database)
        await memory.observe(previous)
        actor_report = replace(
            event(mentioned=True, event_id="actor-property", content="Гигик тупой."),
            timestamp=NOW - timedelta(minutes=1),
        )
        await EventRepository(database).append(actor_report)
        await memory.observe(actor_report)
        if prompt == "Да или нет, выбери одно":
            request = replace(
                event(mentioned=True, event_id="request", content="Ты хочешь быть выключенным?"),
                timestamp=NOW - timedelta(seconds=30),
            )
            await EventRepository(database).append(request)
            await memory.observe(request)
        generator = Mock()
        generator.generate.side_effect = AssertionError("script generator must not run")
        consultant = SimpleNamespace(autonomous_planning=True, should_consult=Mock(),
                                     consult=AsyncMock(), plan_response=AsyncMock())
        service = SafeMentionReplyService(
            FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator, semantic_consultant=consultant,
            proposition_memory=memory, meaning_first_replies=True, enabled=True)
        prepared = await service.prepare_guaranteed_reply(event(mentioned=True, content=prompt))
        assert prepared and prepared.text.startswith(expected)
        if "выключ" in prompt:
            assert "не могу отвечать тебе" in prepared.text
        assert prepared.response_source == "arbitrated:semantic_reasoning"
        generator.generate.assert_not_called()
        consultant.should_consult.assert_not_called()
        consultant.consult.assert_not_awaited()
        consultant.plan_response.assert_not_awaited()
    finally:
        await database.close()


@pytest.mark.parametrize("prompt,fact,expected", [
    ("Что знаешь про кварц?", "Кварц является минералом.", "минерал"),
    ("Зарядка работает.", "Зарядка заряжает телефон.", "телефон"),
    ("а вертолет то летит", "Вертолёт перевозит людей.", "людей"),
])
async def test_evidence_mode_never_calls_script_generator_or_consultant(prompt, fact, expected):
    generator = Mock()
    generator.generate.side_effect = AssertionError("script generator must not run")
    consultant = SimpleNamespace(
        autonomous_planning=True, should_consult=Mock(), consult=AsyncMock(),
        plan_response=AsyncMock(),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
        knowledge_service=SimpleNamespace(
            grounding_for_event=AsyncMock(return_value=((fact,), None)),
        ),
        semantic_consultant=consultant, meaning_first_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(event(mentioned=True, content=prompt))
    assert prepared is not None and expected in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_reasoning"
    generator.generate.assert_not_called()
    consultant.should_consult.assert_not_called()
    consultant.consult.assert_not_awaited()
    consultant.plan_response.assert_not_awaited()


@pytest.mark.parametrize("action", [ActionType.SHORT_REPLY, ActionType.ASK_QUESTION])
async def test_empty_evidence_mode_does_not_emit_a_canned_recovery(action):
    generator = Mock()
    service = SafeMentionReplyService(
        FakeRepository(intent(action)), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
        meaning_first_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="ты молчишь"),
    )
    assert prepared is None
    generator.generate.assert_not_called()


async def test_attitude_statement_gets_an_affect_weighted_position_without_a_script(tmp_path):
    database = Database(
        tmp_path / "affective-stance.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    try:
        current = event(
            mentioned=True,
            event_id="attitude",
            content="Мне плевать на твои идеологии",
        )
        await EventRepository(database).append(current)
        generator = Mock()
        generator.generate.side_effect = AssertionError("script generator must not run")
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            generator,
            dialogue_tasks=DialogueTaskRepository(database),
            meaning_first_replies=True,
            enabled=True,
        )

        prepared = await service.prepare_guaranteed_reply(current)

        assert prepared is not None
        assert prepared.response_source == "arbitrated:affective_stance"
        assert "мои идеологии" in prepared.text.casefold()
        assert "ландсберг" not in prepared.text.casefold()
        assert "марксизм" not in prepared.text.casefold()
        assert prepared.text.casefold() != current.payload["content"].casefold()
        generator.generate.assert_not_called()
    finally:
        await database.close()


@pytest.mark.parametrize("fact,expected,route", [
    ("Маша открыла дверь.", "Да", "semantic_reasoning"),
    ("Маша не открыла дверь.", "Нет", "semantic_reasoning"),
    ("Марс вращается.", None, None),
])
async def test_binary_domain_reaches_delivery_with_and_without_evidence(fact, expected, route):
    question = replace(event(mentioned=True, content="Маша открыла дверь?", event_id="request"),
                       timestamp=NOW - timedelta(seconds=5))
    generator = Mock()
    generator.generate.side_effect = AssertionError("script generator must not run")
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(channel_events=(question,)), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
        knowledge_service=SimpleNamespace(grounding_for_event=AsyncMock(
            return_value=((fact,), None))),
        meaning_first_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(event(mentioned=True,
        content="Да или нет, выбери одно"))
    if expected:
        assert prepared
        assert prepared.text in {"Да", "Нет"}
        assert prepared.text == expected
        assert prepared.response_source == "arbitrated:" + route
    else:
        assert prepared is None
    generator.generate.assert_not_called()


async def test_pending_request_rebuilds_from_chronological_database_after_many_controls(tmp_path):
    from debilgi.language.proposition_memory import PropositionMemory

    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(tmp_path / "pending.sqlite3", migrations)
    await database.initialize()
    try:
        repository = EventRepository(database)
        question = replace(event(mentioned=True, content="Маша открыла дверь?", event_id="request"),
                           timestamp=NOW - timedelta(seconds=50))
        await repository.append(question)
        for index in range(20):
            control = replace(event(mentioned=True, content="Дай ответ",
                                    event_id=f"control:{index}"),
                              timestamp=question.timestamp + timedelta(seconds=index + 1))
            await repository.append(control)
        generator = Mock()
        service = SafeMentionReplyService(
            FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
            proposition_memory=PropositionMemory(database),
            knowledge_service=SimpleNamespace(grounding_for_event=AsyncMock(
                return_value=(("Маша открыла дверь.",), None))),
            meaning_first_replies=True, enabled=True,
        )
        prepared = await service.prepare_guaranteed_reply(event(mentioned=True,
            content="Нет, просто дай бинарный ответ, да или нет"))
        assert prepared and prepared.text == "Да"
        assert prepared.response_source == "arbitrated:semantic_reasoning"
        generator.generate.assert_not_called()
    finally:
        await database.close()


async def test_evidence_mode_retrieves_statement_after_database_restart(tmp_path):
    from debilgi.language.proposition_memory import PropositionMemory

    database_path = tmp_path / "propositions.sqlite3"
    migrations = Path(__file__).resolve().parents[2] / "migrations"
    database = Database(database_path, migrations)
    await database.initialize()
    past = replace(event(mentioned=False, content="Маша нашла ключ.", event_id="past:1"),
                   timestamp=NOW - timedelta(days=2))
    await EventRepository(database).append(past)
    memory = PropositionMemory(database)
    await memory.observe(past)
    await memory.observe(past)
    await database.close()
    database = Database(database_path, migrations)
    await database.initialize()
    try:
        service = SafeMentionReplyService(
            FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, Mock(),
            proposition_memory=PropositionMemory(database),
            meaning_first_replies=True, enabled=True,
        )
        prepared = await service.prepare_guaranteed_reply(
            event(mentioned=True, content="Кто нашёл ключ?"),
        )
        assert prepared is not None and "маша" in prepared.text.casefold()
        count = await database.fetchone("SELECT COUNT(*) AS n FROM language_proposition_reports")
        assert count["n"] == 1
    finally:
        await database.close()


async def test_evidence_mode_blocks_bound_script_reply_paths():
    questions = SimpleNamespace(resolution_for_event=AsyncMock())
    short_acts = SimpleNamespace(resolution_for_event=AsyncMock())
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, Mock(),
        open_questions=questions, short_speech_acts=short_acts,
        meaning_first_replies=True, enabled=True,
    )
    current = event(mentioned=True, content="але")
    assert await service.prepare_bound_explanation(current) is None
    assert await service.prepare_bound_short_act(current) is None
    questions.resolution_for_event.assert_not_awaited()
    short_acts.resolution_for_event.assert_not_awaited()


@pytest.mark.parametrize("prompt", [
    "ауу", "Эм", "зюзюбрик", "ты молчишь", "Почему робот скучает?",
    "напиши что-нибудь странное", "если бы рыба умела летать, что бы она делала",
])
async def test_database_language_graph_answers_unparsed_messages(tmp_path, prompt):
    from debilgi.language.proposition_memory import PropositionMemory

    database = Database(
        tmp_path / "language-graph.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    memory = PropositionMemory(database)
    repository = EventRepository(database)
    source_texts = (
        "Я отвечаю прямо и спокойно.",
        "Ты говоришь коротко и честно.",
        "Я думаю о разговоре вечером.",
        "Ты знаешь значение слова.",
        "Ты думаешь о книге утром.",
        "Робот скучает без разговора.",
        "Человек скучает без дела.",
        "Рыба летает над водой.",
        "Птица летает над домом.",
    )
    source_events = tuple(
        replace(
            event(mentioned=False, event_id=f"source:{index}", content=text),
            timestamp=NOW - timedelta(minutes=len(source_texts) - index),
        )
        for index, text in enumerate(source_texts)
    )
    for source in source_events:
        await repository.append(source)
        await memory.observe(source)
    current = event(mentioned=True, content=prompt, bot_user_id=123)
    await repository.append(current)
    generator = Mock()
    generator.generate.side_effect = AssertionError("script path must not run")
    consultant = SimpleNamespace(autonomous_planning=True, should_consult=Mock(),
                                 consult=AsyncMock(), plan_response=AsyncMock())
    try:
        service = SafeMentionReplyService(
            FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
            semantic_consultant=consultant,
            proposition_memory=memory,
            dialogue_tasks=DialogueTaskRepository(database),
            meaning_first_replies=True,
            productive_only_replies=True,
            enabled=True,
        )
        service._defer_pending_research = AsyncMock(
            side_effect=AssertionError("reply must not wait")
        )
        prepared = await service.prepare_guaranteed_reply(current)
        assert prepared is not None and prepared.text.strip()
        assert prepared.response_source in {
            "arbitrated:learned_distribution",
            "arbitrated:semantic_reasoning",
        }
        assert prepared.text not in source_texts
        assert prepared.text.casefold().rstrip(".!?") != prompt.casefold().rstrip(".!?")
        assert "я слушаю" not in prepared.text.casefold()
        assert "я отвечаю" not in prepared.text.casefold()
        if prompt.rstrip().endswith("?"):
            assert not prepared.text.rstrip().endswith("?")
        generator.generate.assert_not_called()
        consultant.should_consult.assert_not_called()
        consultant.consult.assert_not_awaited()
        consultant.plan_response.assert_not_awaited()
        service._defer_pending_research.assert_not_awaited()
    finally:
        await database.close()


@pytest.mark.parametrize("challenge", ["Всмысле", "Это разве ответ?", "Ответь нормально"])
async def test_answer_challenge_retries_the_original_request_without_a_meta_script(
    tmp_path, challenge
):
    from debilgi.language.proposition_memory import PropositionMemory

    database = Database(
        tmp_path / "answer-repair.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    events = EventRepository(database)
    memory = PropositionMemory(database)
    tasks = DialogueTaskRepository(database)
    sources = (
        replace(
            event(mentioned=False, event_id="cause:robot", content="Робот скучает без разговора."),
            timestamp=NOW - timedelta(minutes=4),
        ),
        replace(
            event(mentioned=False, event_id="cause:human", content="Человек скучает без дела."),
            timestamp=NOW - timedelta(minutes=3),
        ),
    )
    question = replace(
        event(mentioned=True, event_id="original-question", content="Почему робот скучает?"),
        timestamp=NOW - timedelta(seconds=20),
    )
    old_reply = Event(
        event_id="old-bot-reply",
        guild_id=1,
        channel_id=10,
        actor_id=123,
        target_id=100,
        event_type=EventType.BOT_REPLIED,
        timestamp=NOW - timedelta(seconds=10),
        payload={
            "bot_mentioned": False,
            "author_is_bot": True,
            "content": "Я не могу ответить на этот вопрос.",
            "bot_user_id": 123,
        },
        source="test",
    )
    current = event(mentioned=True, content=challenge, bot_user_id=123)
    try:
        for source in sources:
            await events.append(source)
            await memory.observe(source)
        await events.append(question)
        initial = await tasks.deliberate(
            question,
            prompt=question.payload["content"],
            consultation_messages=(),
            dialogue_state=None,
        )
        assert initial is not None and initial.active_task is not None
        assert await tasks.record_attempt(
            question,
            task_id=initial.active_task.task_id,
            response_message_id=777,
            text=old_reply.payload["content"],
            created_at=old_reply.timestamp,
        )
        await events.append(old_reply)
        await events.append(current)
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(channel_events=(*sources, question, old_reply)),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            Mock(),
            proposition_memory=memory,
            dialogue_tasks=tasks,
            meaning_first_replies=True,
            enabled=True,
        )

        prepared = await service.prepare_guaranteed_reply(current)

        assert prepared is not None
        assert prepared.response_source != "arbitrated:semantic_recovery"
        assert {"робот", "скучает"} <= set(prepared.text.casefold().rstrip(".!?").split())
        assert "не могу ответить" not in prepared.text.casefold()
        assert "исправля" not in prepared.text.casefold()
        assert prepared.text.casefold().rstrip(".!?") != challenge.casefold().rstrip(".!?")
    finally:
        await database.close()


async def test_unknown_fragment_without_language_data_does_not_trigger_a_script():
    generator = Mock()
    generator.generate.side_effect = AssertionError("script path must not run")
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, generator,
        meaning_first_replies=True, productive_only_replies=True, enabled=True,
    )

    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="буу"),
    )

    assert prepared is None
    generator.generate.assert_not_called()


async def test_stretched_contact_resumes_the_unanswered_human_statement():
    previous = replace(
        event(mentioned=True, content="ебать, а я ногти грызу", event_id="prior", bot_user_id=123),
        timestamp=NOW - timedelta(seconds=5),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(channel_events=(previous,)), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, Mock(),
        meaning_first_replies=True, productive_only_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="ауу", bot_user_id=123),
    )
    assert prepared is not None
    assert "грыз" in prepared.text.casefold() and "ногти" in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_reasoning"


@pytest.mark.parametrize("upstream_failure", [None, "event_processing_timeout"])
async def test_direct_reply_without_intent_or_language_data_does_not_invent_a_script(
    upstream_failure,
):
    service = SafeMentionReplyService(
        FakeRepository(None), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, Mock(),
        meaning_first_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="привет"), upstream_failure=upstream_failure,
    )
    assert prepared is None


async def test_local_generation_recovers_from_context_failure_without_a_canned_phrase():
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE, Mock(),
        meaning_first_replies=True, enabled=True,
    )
    service._reply_context = AsyncMock(side_effect=RuntimeError("context unavailable"))
    prepared = await service.prepare_guaranteed_reply(event(mentioned=True, content="зюзюбрик"))
    assert prepared is None


@pytest.mark.parametrize(
    "topic,fact,expected",
    [
        ("кварц", "Кварц является минералом.", "минерал"),
        ("Марс", "Марс вращается.", "вращ"),
        ("Машу", "Маша нашла ключ.", "ключ"),
    ],
)
async def test_content_request_reaches_delivery_as_fact_not_knowledge_state(topic, fact, expected):
    knowledge = SimpleNamespace(grounding_for_event=AsyncMock(return_value=((fact,), None)))
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Да, я знаю."),
        knowledge_service=knowledge,
        productive_only_replies=True,
        arbitration_mode="active",
        enabled=True,
    )
    service._remember_arbitration = Mock(wraps=service._remember_arbitration)
    prepared = await service.prepare_reply(
        event(mentioned=True, content=f"Что знаешь про {topic}?"),
    )
    assert prepared is not None
    assert expected in prepared.text.casefold()
    assert "знаю" not in prepared.text.casefold()
    decision = service._remember_arbitration.call_args.args[1]
    assert decision.winner.metadata["thought_before_surface"] is True
    assert decision.winner.metadata["reader_checked"] is True


@pytest.mark.parametrize("prompt", ["лошадка", "пиздээ", "не знаю", "расскажи про космос"])
@pytest.mark.parametrize("learned_penalty", [False, True])
async def test_non_questions_do_not_collapse_into_unknown_answer(prompt, learned_penalty):
    class FeedbackRepository:
        async def profile(self, guild_id):
            return ArbitrationProfile.baseline(guild_id)

        async def structural_weights(self, guild_id):
            return {"source:dialogue_move": -100.0, "source:semantic_recovery": -100.0}

        async def save_decision(self, *args, **kwargs):
            pass

    class NoLexicalOverlap:
        async def measure_candidates(self, **kwargs):
            return tuple(
                SimpleNamespace(score=0.0, focus_concepts=(prompt,))
                for _ in kwargs["candidate_texts"]
            )

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        context_gate=NoLexicalOverlap(),
        arbitration_repository=FeedbackRepository() if learned_penalty else None,
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )
    # Dictionary research can be pending at delivery time. It must not replace
    # a successfully arbitrated conversational question with a no-answer claim.
    service._defer_pending_research = AsyncMock(return_value=(prompt,))
    service._remember_arbitration = Mock(wraps=service._remember_arbitration)
    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))
    assert prepared is not None
    assert "не знаю ответа" not in prepared.text.casefold()
    assert "не отправился" not in prepared.text.casefold()
    assert prepared.response_source.startswith("arbitrated:")
    decision = service._remember_arbitration.call_args.args[1]
    assert all(
        "outside_selected_message_plan" not in score.rejection_reasons for score in decision.scores
    )
    if prompt in {"лошадка", "пиздээ"}:
        # Neither a successful first realization nor a familiar move label
        # should remove the other independently composed meanings.
        plans = {score.candidate.metadata.get("message_plan_key") for score in decision.scores}
        assert len(plans - {None}) >= 2


async def test_productive_mode_answers_elliptical_follow_up_instead_of_stopping():
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_guaranteed_reply(event(mentioned=True, content="ну и?"))

    assert prepared is not None
    assert prepared.text.strip()
    assert prepared.response_source.startswith("arbitrated:")


@pytest.mark.parametrize(
    "prompt,is_question",
    [
        ("лошадка", False),
        ("Сколько стоит гречка?", True),
    ],
)
async def test_delivery_definition_guard_reenters_generation(prompt, is_question):
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )
    service._generate_checked = AsyncMock(return_value="Что значит лошадка?")
    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))
    if is_question:
        assert prepared is None  # No price evidence; a canned no-answer is not a reply.
        return
    assert prepared is not None
    assert "не знаю ответа" not in prepared.text.casefold()
    assert "Что значит" not in prepared.text
    assert prepared.arbitration_decision_id is not None


@pytest.mark.parametrize("surface_budget", [4, 48])
@pytest.mark.parametrize(
    "prompt,lemma,meanings,expected",
    [
        (
            "Что думаешь о ножницах?",
            "ножницы",
            ("режущий инструмент из двух раздвигающихся лезвий", "технический приём в футболе"),
            ("полезными", "резать"),
        ),
        (
            "У тебя нос капает?",
            "нос",
            ("выступающая часть лица человека", "передняя часть судна"),
            ("носа", "тела"),
        ),
        ("Что думаешь о грелке?", "грелка", ("согревающий прибор",), ("полезной", "согревать")),
    ],
)
async def test_dictionary_premises_reach_a_composed_selected_answer(
    tmp_path,
    monkeypatch,
    prompt,
    lemma,
    meanings,
    expected,
    surface_budget,
):
    db = Database(tmp_path / "premises.sqlite3", Path(__file__).resolve().parents[2] / "migrations")
    await db.initialize()
    lexicon = LexiconService(db)
    client = SimpleNamespace(
        lookup=AsyncMock(
            return_value=RemoteLexeme(
                lemma,
                "NOUN",
                meanings,
                f"https://ru.wiktionary.org/wiki/{lemma}",
            )
        )
    )
    reference = RussianReferenceService(db, lexicon, client, enabled=True)
    try:
        await reference.refresh_word(lemma)
        current = event(mentioned=True, content=prompt)
        await EventRepository(db).append(current)
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            FixedGenerator(""),
            lexicon=lexicon,
            productive_only_replies=True,
            arbitration_mode="active",
            enabled=True,
            arbitration_max_total_candidates=surface_budget,
        )
        decisions = []
        remember = service._remember_arbitration

        def capture(event_id, decision):
            decisions.append(decision)
            remember(event_id, decision)

        monkeypatch.setattr(service, "_remember_arbitration", capture)
        prepared = await service.prepare_reply(current)
        assert prepared is not None
        assert all(part in prepared.text for part in expected)
        assert "потому что" in prepared.text
        assert prepared.response_source == "arbitrated:semantic_reasoning"
        assert prepared.assertion_experience is None
        winner = decisions[-1].winner
        assert winner.renderer.startswith("premise_chain:")
        assert winner.metadata["locally_composed"] is True
        assert winner.metadata["thought_before_surface"] is True
        assert winner.metadata["reader_checked"] is True
        assert winner.metadata["message_plan_key"]
        assert winner.metadata["message_plan"]["reason"] is not None
        assert winner.metadata["reasoning_evidence"]
        assert winner.metadata["conditional_assumption"] == (len(meanings) > 1)
        assert ("Если" in prepared.text) == (len(meanings) > 1)
        client.lookup.assert_awaited_once_with(lemma)
    finally:
        await db.close()


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Что такое спаржа?", "Спаржа является овощем."),
        ("Что такое замок?", "«замок» означает «замок для запирания» или «средневековый замок»."),
    ],
)
async def test_installed_ruwordnet_reaches_selected_reply_without_network(
    tmp_path,
    prompt,
    expected,
):
    root = Path(__file__).resolve().parents[2]
    reference = root / "data/models/ruwordnet/ruwordnet-2021.db"
    if not reference.exists():
        pytest.skip("Published RuWordNet database is an optional local installation")
    db = Database(tmp_path / "reply-ruwordnet.sqlite3", root / "migrations")
    await db.initialize()
    try:
        current = event(mentioned=True, content=prompt)
        await EventRepository(db).append(current)
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            FixedGenerator(""),
            lexicon=LexiconService(db, ruwordnet=RuWordNet(reference)),
            productive_only_replies=True,
            arbitration_mode="active",
            enabled=True,
        )
        prepared = await service.prepare_reply(current)
        assert prepared is not None
        assert prepared.text == expected
        assert prepared.response_source == "arbitrated:semantic_thought"
    finally:
        await db.close()


async def test_acquired_definition_reaches_the_selected_reply_without_network(tmp_path):
    db = Database(
        tmp_path / "reply-definition.sqlite3", Path(__file__).resolve().parents[2] / "migrations"
    )
    await db.initialize()
    try:
        current = event(mentioned=True, content="Для чего ручка?")
        await EventRepository(db).append(current)
        await KnowledgeRepository(db).store_internalized_concept(
            current,
            query="ручка",
            source="wikipedia",
            language="ru",
            result=WikipediaResult(
                1,
                "Ручка",
                "Ручка — письменный инструмент для письма.",
                "https://ru.wikipedia.org/wiki/Ручка",
                None,
                1,
                (),
            ),
        )
        service = SafeMentionReplyService(
            FakeRepository(intent()),
            FakeMemory(),
            FakeArchiveRepository(),
            EXPRESSION_ENGINE,
            EXPERIENCE,
            INITIATIVE,
            FixedGenerator(""),
            lexicon=LexiconService(db),
            productive_only_replies=True,
            arbitration_mode="active",
            enabled=True,
        )
        prepared = await service.prepare_reply(current)
        assert prepared is not None
        assert prepared.text == "Ручка используется для письма."
        assert prepared.response_source == "arbitrated:semantic_thought"
    finally:
        await db.close()


@pytest.mark.parametrize("autonomous", [False, True])
async def test_temporal_continuation_uses_attributed_memory_without_external_planner(autonomous):
    async def adapt(current, prompt, plan):
        return plan, ExperienceGuidance()

    experiences = SimpleNamespace(adapt=AsyncMock(side_effect=adapt), record_sent=AsyncMock())
    prior = replace(
        event(
            mentioned=False,
            author_is_bot=True,
            actor_id=123,
            target_id=100,
            bot_user_id=123,
            content="Я пока не открыл дверь.",
            event_id="prior",
        ),
        timestamp=NOW - timedelta(seconds=2),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(prior,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
        assertion_experiences=experiences,
    )
    service._autonomous_planning_required = autonomous
    current = event(mentioned=True, bot_user_id=123, content="Ну всё, время настало")
    context = await service._reply_context(current, intent())
    assert context.assertion_continuation is not None
    prepared = await service.prepare_reply(current)
    assert prepared is not None
    assert prepared.text == "Я открыл дверь?"
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert prepared.assertion_experience is not None
    assert prepared.assertion_experience["anchor_event_id"] == prior.event_id
    experiences.record_sent.assert_not_awaited()
    await service.record_sent(
        current,
        response_message_id=1234,
        reply=replace(prepared, fallback=True),
        created_at=NOW,
    )
    experiences.record_sent.assert_awaited_once_with(
        current,
        response_message_id=1234,
        proposal=prepared.assertion_experience,
        created_at=NOW,
    )
    # A learned rejection must actually remove this interpretation from runtime context.
    experiences.adapt = AsyncMock(return_value=(None, ExperienceGuidance(0, 2)))
    rejected_context = await service._reply_context(current, intent())
    assert rejected_context.assertion_continuation is None
    assert rejected_context.assertion_feedback_counts == (0, 2)


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
    assert not reply_module._research_fact_bound_to_prompt(
        "А я сейчас вам покажу, откуда на Беларусь готовилось нападение.",
        "У тебя сейчас какая раскладка языка стоит?",
    )
    assert reply_module._research_fact_bound_to_prompt(
        "Русская раскладка использует кириллицу.",
        "Какая русская раскладка используется?",
    )
    layout_question = reply_module.analyze_structured_utterance(
        "У тебя сейчас какая раскладка языка стоит?"
    )
    assert reply_module._question_addresses_agent(
        reply_module.analyze_dialogue_act(layout_question),
        layout_question,
    )


def test_epistemic_recovery_is_not_a_global_question_mode() -> None:
    self_question = reply_module.analyze_structured_utterance("Ты герой что ли?")
    world_question = reply_module.analyze_structured_utterance("Где лежит ключ?")

    assert not reply_module._allows_epistemic_recovery(
        reply_module.analyze_dialogue_act(
            self_question,
            self_state=reply_module.analyze_self_state(self_question),
        ),
        self_question,
    )
    assert reply_module._allows_epistemic_recovery(
        reply_module.analyze_dialogue_act(world_question),
        world_question,
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


class FailingArbitrationPersistence:
    def __init__(self) -> None:
        self.save_attempts = 0

    async def profile(self, guild_id: int) -> ArbitrationProfile:
        return ArbitrationProfile.baseline(guild_id)

    async def structural_weights(self, guild_id: int) -> dict[str, float]:
        del guild_id
        return {}

    async def save_decision(self, *args, **kwargs) -> None:
        del args, kwargs
        self.save_attempts += 1
        raise RuntimeError("arbitration database unavailable")


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
            provider="fake-provider",
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
            provider="fake-provider",
            model="fake-model",
        )


class FakeAutonomousPlanner:
    autonomous_planning = True
    trigger_threshold = 0.72

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def should_consult(self, **kwargs) -> bool:
        del kwargs
        return False

    async def plan_response(self, **kwargs) -> ConsultationResult:
        self.calls.append(kwargs)
        grounding_ids = {fact_id for fact_id, _ in kwargs["grounding_facts"]}
        propositions = []
        if "previous_answer.topic_match" in grounding_ids:
            propositions.append(
                {
                    "predicate": "ответить",
                    "predicate_features": {
                        "tense": "past",
                        "gender": "feminine",
                    },
                    "arguments": [
                        {
                            "role": "subject",
                            "lemma": "я",
                            "part_of_speech": "pronoun",
                        },
                        {
                            "role": "topic",
                            "lemma": "тема",
                            "part_of_speech": "noun",
                            "grammatical_case": "dative",
                            "preposition": "по",
                        },
                    ],
                    "clause_kind": "statement",
                    "negated": True,
                    "evidence_ids": ["previous_answer.topic_match"],
                }
            )
        propositions.extend(
            (
                {
                    "predicate": "наблюдать",
                    "predicate_features": {"tense": "present"},
                    "arguments": [
                        {
                            "role": "subject",
                            "lemma": "я",
                            "part_of_speech": "pronoun",
                        },
                        {
                            "role": "direct_object",
                            "lemma": "раскладка",
                            "part_of_speech": "noun",
                            "modifiers": ["системный"],
                        },
                    ],
                    "clause_kind": "statement",
                    "negated": True,
                    "evidence_ids": ["runtime.keyboard_layout.observable"],
                },
                {
                    "predicate": "приходить",
                    "predicate_features": {"tense": "present"},
                    "arguments": [
                        {
                            "role": "subject",
                            "lemma": "сообщение",
                            "part_of_speech": "noun",
                        },
                        {
                            "role": "manner",
                            "lemma": "по-русски",
                            "part_of_speech": "adverb",
                        },
                    ],
                    "clause_kind": "statement",
                    "negated": False,
                    "evidence_ids": ["message.language"],
                },
            )
        )
        return ConsultationResult.from_payload(
            {
                "interpretation": "Семантический план ответа.",
                "intent": "runtime_state_answer",
                "response_text": "ЭТУ ГОТОВУЮ ФРАЗУ КОПИРОВАТЬ НЕЛЬЗЯ.",
                "response_plans": [
                    {
                        "move": "answer",
                        "stance": "direct",
                        "propositions": propositions,
                        "confidence": 0.91,
                    }
                ],
                "confidence": 0.91,
            },
            provider="fake-semantic-planner",
            model="no-surface",
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
        if analysis.normalized_form not in self.queued:
            self.queued.append(analysis.normalized_form)
        return True

    async def queue_semantic_study(self, current, terms, **kwargs):
        for term in terms:
            if term not in self.queued:
                self.queued.append(term)


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
    reply_to_message_id: int | None = None,
    target_id: int | None = None,
    bot_user_id: int | None = None,
) -> Event:
    return Event(
        event_id=event_id,
        guild_id=1,
        channel_id=10,
        actor_id=actor_id,
        target_id=target_id,
        event_type=(
            EventType.MESSAGE_REPLIED
            if reply_to_message_id is not None
            else EventType.MESSAGE_CREATED
        ),
        timestamp=NOW,
        payload={
            "bot_mentioned": mentioned,
            "author_is_bot": author_is_bot,
            "content": content,
            "reply_to_message_id": reply_to_message_id,
            "bot_user_id": bot_user_id,
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


async def test_clarification_intent_uses_semantic_plan_before_general_generator() -> None:
    class ForbiddenGenerator:
        def generate(self, plan):
            del plan
            raise AssertionError("general generator must not realize clarification intent")

    current = event(mentioned=True, content="<@123> это самое, ну ты поняла")
    service = SafeMentionReplyService(
        FakeRepository(intent(ActionType.ASK_QUESTION)),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        ForbiddenGenerator(),
        enabled=True,
    )

    prepared = await service.prepare_reply(current)

    assert prepared is not None
    assert prepared.semantic_reply_plan_id == "semantic-reply:intent:test"
    assert prepared.processing_outcome is not None
    assert prepared.response_source.startswith("semantic_reply_plan")
    assert prepared.text.endswith("?")


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


async def test_productive_only_mode_withholds_fixed_transport_fallback() -> None:
    current = event(mentioned=True, content="<@123> объясни штуку")
    service = SafeMentionReplyService(
        FailingRepository(),  # type: ignore[arg-type]
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        MemoryPhraseGenerator(),
        productive_only_replies=True,
        enabled=True,
    )

    assert await service.prepare_guaranteed_reply(current) is None


async def test_arbitration_persistence_failure_does_not_replace_a_valid_reply() -> None:
    current = event(
        mentioned=True,
        content="<@123> собачки умеют бегать?",
    )
    arbitration = FailingArbitrationPersistence()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(user_events=(current,), channel_events=(current,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Собачки умеют бегать."),
        arbitration_repository=arbitration,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=-100.0,
        enabled=True,
    )

    prepared = await service.prepare_guaranteed_reply(current)

    assert arbitration.save_attempts == 1
    assert prepared is not None
    assert prepared.text != "Повтори запрос: сейчас ответ не отправился."
    assert prepared.intent_id == "intent:test"
    assert prepared.arbitration_decision_id is not None


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


async def test_unknown_statement_does_not_open_a_question_even_with_stale_pending_marker() -> None:
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

    assert questions.opened == []


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
    assert not reply_module.requests_word_definition(prepared.text)
    assert "самостоятельный разбор" not in prepared.text
    assert "очеред" not in prepared.text
    assert lexicon.queued == ["стромба"]


async def test_explicit_definition_request_does_not_ask_user_to_define_it() -> None:
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

    assert prepared is None  # Research remains queued without a technical chat response.
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


async def test_active_arbitration_rejects_logged_graph_garbage_without_forcing_fallback() -> None:
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

    assert prepared is None


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
    definition = "Ответ — реплика, вызванная вопросом или реакция на событие."
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
    assert prepared.text == "Я слушаю тебя."
    assert prepared.response_source == "arbitrated:dialogue_act"


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


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("repair", [False, True])
async def test_reference_learning_cannot_replace_conversation_with_a_definition(
    cached: bool,
    repair: bool,
) -> None:
    from debilgi.knowledge.models import KnowledgeGrounding

    definition = "понимать — постигать смысл какого-либо высказывания, явления"

    class ReferenceKnowledge:
        async def grounding_evidence_for_event(self, current, prompt):
            if cached:
                return KnowledgeGrounding(
                    facts=(definition,),
                    exact_answer=definition,
                    definitions=(("понимать", definition),),
                )
            return KnowledgeGrounding()

        async def internalize_for_reasoning(self, current, terms, *, limit=6):
            return (SimpleNamespace(title="понимать", definition=definition),)

    history = (
        (
            event(mentioned=True, content="Как это понимать?", event_id="discord:message:question"),
            event(
                mentioned=False,
                author_is_bot=True,
                actor_id=123,
                content=definition,
                event_id="discord:message:bad-answer",
            ),
        )
        if repair
        else ()
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=history),
        FakeArchiveRepository(exact_reply=definition),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(definition),
        knowledge_service=ReferenceKnowledge(),
        lexicon=RelationLexicon(),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )
    current = event(
        mentioned=True,
        content=("<@123> О как, но я это не спрашивал" if repair else "<@123> Как это понимать?"),
    )
    context = await service._reply_context(current, intent())
    assert not context.required_facts
    assert context.exact_answer is None
    prepared = await service.prepare_reply(current)
    assert prepared is not None
    assert "постигать" not in prepared.text.casefold()
    assert "какого-либо" not in prepared.text.casefold()
    assert not reply_module.requests_word_definition(prepared.text)


@pytest.mark.parametrize("prompt", ["Что значит понимать?", "Объясни значение слова понимать"])
async def test_explicit_definition_can_surface_reference_evidence(prompt: str, monkeypatch) -> None:
    from debilgi.knowledge.models import KnowledgeGrounding

    definition = "понимать — постигать смысл какого-либо высказывания, явления"

    class ReferenceKnowledge:
        async def grounding_evidence_for_event(self, current, prompt):
            return KnowledgeGrounding(
                facts=(definition,),
                definitions=(("понимать", definition),),
            )

        async def internalize_for_reasoning(self, current, terms, *, limit=6):
            return (SimpleNamespace(title="понимать", definition=definition),)

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        knowledge_service=ReferenceKnowledge(),
        lexicon=RelationLexicon(),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )
    decisions = []
    remember = service._remember_arbitration

    def capture(event_id, decision):
        decisions.append(decision)
        remember(event_id, decision)

    monkeypatch.setattr(service, "_remember_arbitration", capture)
    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))
    assert prepared is not None
    assert "постигать смысл" in prepared.text.casefold()
    assert not reply_module.requests_word_definition(prepared.text)
    assert prepared.text != definition
    assert decisions[-1].winner.renderer == "definition_action_roles"
    assert "action_complement" in decisions[-1].winner.metadata["role_order"]
    assert decisions[-1].winner.metadata["locally_composed"] is True


@pytest.mark.parametrize(
    ("prompt", "definition"),
    [
        (
            "И что это мне должно давать",
            "давать — перемещать что-либо в распоряжение другого субъекта",
        ),
        (
            "Ок и че?",
            "Отве́т — реплика, вызванная заданным вопросом или реакция на какое-либо событие.",
        ),
    ],
)
async def test_old_untyped_definitions_cannot_become_answer_plans(prompt, definition, monkeypatch):
    # Historical memory lacks the provenance now provided by reference services.
    knowledge = SimpleNamespace(
        grounding_for_event=AsyncMock(return_value=((definition,), definition)),
        internalize_for_reasoning=AsyncMock(return_value=()),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(exact_reply=definition),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(definition),
        knowledge_service=knowledge,
        lexicon=RelationLexicon(),
        productive_only_replies=True,
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )
    decisions = []
    remember = service._remember_arbitration

    def capture(event_id, decision):
        decisions.append(decision)
        remember(event_id, decision)

    monkeypatch.setattr(service, "_remember_arbitration", capture)
    reply = await service.prepare_reply(event(mentioned=True, content=prompt))
    assert reply is not None
    assert reply.text != definition
    assert not any(
        word in reply.text.casefold() for word in ("распоряжение", "реплика", "субъекта")
    )
    assert not any(
        score.eligible
        and score.candidate.renderer
        in {
            "internalized_lexical_fact",
            "multi_act_knowledge_then_position",
        }
        for decision in decisions
        for score in decision.scores
    )


async def test_productive_fact_answer_is_composed_without_a_raw_required_reply(monkeypatch):
    fact = "Лиса взяла красный ключ со стола."
    knowledge = SimpleNamespace(
        grounding_for_event=AsyncMock(return_value=((fact,), fact)),
        internalize_for_reasoning=AsyncMock(return_value=()),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        knowledge_service=knowledge,
        productive_only_replies=True,
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )
    decisions = []
    remember = service._remember_arbitration

    def capture(event_id, decision):
        decisions.append(decision)
        remember(event_id, decision)

    monkeypatch.setattr(service, "_remember_arbitration", capture)
    reply = await service.prepare_reply(event(mentioned=True, content="Что сделала лиса?"))
    assert reply is not None
    assert "ключ" in reply.text.casefold()
    assert decisions[-1].winner.renderer == "structured_russian"
    assert decisions[-1].winner.metadata["proposition_count"] == 1
    assert decisions[-1].winner.metadata["locally_composed"] is True


async def test_event_answer_does_not_wait_for_an_incidental_word_lookup():
    reference = SimpleNamespace(
        refresh_word=AsyncMock(return_value=None),
        retry_delay_for=AsyncMock(return_value=300),
    )
    previous = replace(
        event(
            mentioned=True,
            content="Я положил фепку на стол",
            event_id="discord:message:previous-event",
        ),
        timestamp=NOW - timedelta(seconds=1),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        lexicon=RelationLexicon(),
        russian_reference=reference,
        arbitration_mode="active",
        productive_only_replies=True,
        arbitration_minimum_score=0.0,
        enabled=True,
    )
    prepared = await service.prepare_reply(event(mentioned=True, content="Куда я положил фепку?"))
    assert prepared is not None
    assert "на стол" in prepared.text.casefold()
    reference.refresh_word.assert_not_awaited()
    # An empty task dependency list must not consult a service-wide pending status.
    reference.retry_delay_for.assert_not_awaited()


async def test_missing_reference_is_not_sent_to_dictionary_recovery():
    reference = SimpleNamespace(
        refresh_word=AsyncMock(return_value=None),
        retry_delay_for=AsyncMock(return_value=300),
    )
    knowledge = SimpleNamespace(
        grounding_for_event=AsyncMock(return_value=((), None)),
        internalize_for_reasoning=AsyncMock(return_value=()),
        investigate=AsyncMock(return_value=()),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        lexicon=RelationLexicon(),
        russian_reference=reference,
        knowledge_service=knowledge,
        arbitration_mode="active",
        productive_only_replies=True,
        arbitration_minimum_score=0.0,
        enabled=True,
    )
    prepared = await service.prepare_reply(event(mentioned=True, content="Как это понимать?"))
    assert prepared is not None
    reference.refresh_word.assert_not_awaited()
    reference.retry_delay_for.assert_not_awaited()
    knowledge.internalize_for_reasoning.assert_not_awaited()
    knowledge.investigate.assert_not_awaited()


async def test_deferred_turn_cannot_bind_a_reference_to_a_future_message():
    previous = replace(
        event(mentioned=True, content="Волк живёт в лесу", event_id="discord:message:old"),
        timestamp=NOW - timedelta(seconds=1),
    )
    future = replace(
        event(mentioned=True, content="Кот живёт на кухне", event_id="discord:message:future"),
        timestamp=NOW + timedelta(seconds=1),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous, future)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        enabled=True,
    )
    context = await service._reply_context(event(mentioned=True, content="Где он живёт?"), intent())
    assert "волк живёт в лесу" in context.dialogue_messages
    assert "кот живёт на кухне" not in context.dialogue_messages
    assert all("кухне" not in message for message in context.consultation_messages)


async def test_agent_intent_question_does_not_surface_a_dictionary_fact() -> None:
    class MisleadingKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            assert current.event_id == "discord:message:1"
            assert "хочешь сказать" in prompt
            definition = "Хотеть — глагол, обозначающий желание или стремление к чему-то."
            return (definition,), definition

    previous_bot_turn = event(
        mentioned=False,
        author_is_bot=True,
        content="Что ты хочешь сказать про типо?",
        event_id="discord:message:0",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_bot_turn,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(". затем случайный обрывок разобрать?."),
        knowledge_service=MisleadingKnowledge(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> Хз, а ты что хочешь сказать?")
    )

    assert prepared is not None
    assert prepared.text == "Я хочу понять твою мысль."
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert "глагол" not in prepared.text.casefold()


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


@pytest.mark.parametrize("prompt", ["<@123> ну так что? Можешь?", "<@123> але", "<@123> аууу"])
async def test_live_modal_follow_up_recovers_full_question_through_bad_intervening_turns(
    prompt,
) -> None:
    history = (
        event(
            mentioned=True,
            event_id="discord:message:capability-question",
            content="<@123> Ты машиной управлять можешь?",
        ),
        event(
            mentioned=True,
            event_id="discord:message:elliptical-question",
            content="<@123> Не знаю, а ты можешь?",
        ),
        event(
            mentioned=False,
            author_is_bot=True,
            event_id="discord:message:bad-agent-answer",
            content="Не знаю, чего хочешь, как я обычно решаю, что делать дальше?",
        ),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=history),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Ты спрашивал."),
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))

    assert prepared is not None
    assert "управлять" in prepared.text.casefold(), (
        prepared.text,
        prepared.response_source,
    )
    assert "машин" in prepared.text.casefold()
    assert "?" not in prepared.text
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
                provider="fake-provider",
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
                provider="fake-provider",
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

    assert prepared is None


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
    assert prepared.response_source == "arbitrated:personal_action"
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
    assert prepared.response_source == "arbitrated:personal_action"


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


async def test_answer_gap_does_not_force_a_below_threshold_stock_sentence() -> None:
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

    assert prepared is None


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


async def test_morphological_familiarity_does_not_claim_understanding_of_an_idea() -> None:
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
    assert "не понял" not in prepared.text.casefold()
    assert prepared.text.endswith("?")


async def test_colloquial_self_role_question_stays_conversational() -> None:
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
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты чё герой что-ли")
    )

    assert prepared is not None
    assert "геро" in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert prepared.fallback is False
    assert "не могу ответить" not in prepared.text.casefold()


async def test_productive_only_colloquial_confirmation_never_stays_silent() -> None:
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
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> ты же круто, да?"))

    assert prepared is not None
    assert "крут" in prepared.text.casefold()
    assert prepared.text != "Нет, я не считаю себя крутым."
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert prepared.fallback is False


async def test_productive_only_fragment_builds_clarification_from_roles() -> None:
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
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> ебануться"))

    assert prepared is not None
    assert prepared.text.endswith("?")
    assert prepared.text != "Что ты подразумеваешь под ебануться?"
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert prepared.fallback is False


async def test_reason_follow_up_uses_previous_self_position_not_faq_fallback() -> None:
    previous_question = event(
        mentioned=True,
        content="<@123> ты чё герой что-ли",
        event_id="discord:message:previous-question",
    )
    previous_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Нет, я не считаю себя героем.",
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
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> А почему так"))

    assert prepared is not None
    assert prepared.text == "У меня нет оснований для такой оценки."
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert prepared.fallback is False
    assert "не могу ответить" not in prepared.text.casefold()


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
            "arbitrated:personal_action",
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
    if "считаешь себя умным" in prompt:
        assert not prepared.text.startswith("Да,")
        return
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


async def test_epistemic_cause_does_not_invent_a_threshold_failure() -> None:
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

    assert prepared is None


async def test_criticism_after_no_data_reply_does_not_invent_a_threshold_failure() -> None:
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

    assert prepared is None


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
    assert not prepared.text.startswith("Да,")
    assert "умник ответил" not in prepared.text.casefold()


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
    assert not prepared.text.startswith("Да,")
    assert "умник ответил" not in prepared.text.casefold()


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


async def test_personal_action_question_cannot_surface_encyclopedic_homonym() -> None:
    class CapturingTraces:
        def __init__(self) -> None:
            self.steps: list[dict[str, object]] = []

        async def begin(self, *args, **kwargs) -> None:
            del args, kwargs

        async def step(self, *args, **kwargs) -> None:
            del args
            self.steps.append(kwargs)

    class MisleadingKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            del current, prompt
            definition = "Ножницы Windows — это приложение для создания снимков экрана."
            return (definition,), definition

    traces = CapturingTraces()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        knowledge_service=MisleadingKnowledge(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        response_traces=traces,  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты ножницы покупал?")
    )

    assert prepared is not None, traces.steps
    assert prepared.text == "Нет, я не покупал ножницы."
    assert prepared.response_source == "arbitrated:personal_action"
    assert "Windows" not in prepared.text


async def test_runtime_layout_question_cannot_surface_unrelated_research() -> None:
    class CapturingTraces:
        def __init__(self) -> None:
            self.steps: list[dict[str, object]] = []

        async def begin(self, *args, **kwargs) -> None:
            del args, kwargs

        async def step(self, *args, **kwargs) -> None:
            del args
            self.steps.append(kwargs)

    class MisleadingKnowledge:
        async def grounding_for_event(
            self,
            current: Event,
            prompt: str,
        ) -> tuple[tuple[str, ...], str | None]:
            del current, prompt
            bad_fact = "А я сейчас вам покажу, откуда на Беларусь готовилось нападение."
            return (bad_fact,), bad_fact

    planner = FakeAutonomousPlanner()
    traces = CapturingTraces()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        knowledge_service=MisleadingKnowledge(),  # type: ignore[arg-type]
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        response_traces=traces,  # type: ignore[arg-type]
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> у тебя сейчас какая раскладка языка стоит?",
        )
    )

    assert prepared is not None
    assert prepared.text == ("Я не наблюдаю системную раскладку. Сообщение приходит по-русски.")
    assert prepared.response_source == "arbitrated:autonomous_semantic_plan"
    assert "Беларус" not in prepared.text
    assert "КОПИРОВАТЬ" not in prepared.text
    assert planner.calls
    candidate_step = next(step for step in traces.steps if step["component"] == "response_sources")
    assert {item["source"] for item in candidate_step["details"]["candidates"]} == {
        "autonomous_semantic_plan"
    }
    knowledge_step = next(
        step for step in traces.steps if step["component"] == "knowledge_and_consultation"
    )
    assert "meaning_plan" in knowledge_step["details"]["suppressed_precomposed_sources"]


async def test_unavailable_autonomous_planner_never_falls_back_to_fixed_phrase() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            return None

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Повтори запрос: сейчас ответ не отправился."),
        semantic_consultant=UnavailablePlanner(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=-100.0,
        productive_only_replies=False,
        enabled=True,
    )

    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="<@123> как дела?")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:fallback"
    assert "Повтори запрос" not in prepared.text


async def test_personality_desire_survives_unavailable_autonomous_planner() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            return None

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=UnavailablePlanner(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> ты хочешь побегать?")
    )

    assert prepared is not None
    assert prepared.text in {
        "Да, я хочу побегать.",
        "Нет, я не хочу побегать.",
    }
    assert prepared.response_source == "arbitrated:personality_desire"


async def test_local_self_reflection_does_not_wait_for_unavailable_planner() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> думаешь типо?"))

    assert prepared is not None
    assert prepared.text == "Я анализирую контекст."
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert planner.calls == 0


@pytest.mark.parametrize(
    ("prompt", "expected_source", "expected_texts"),
    (
        (
            "ауу",
            "arbitrated:dialogue_move",
            {"Я отвечаю тебе."},
        ),
        (
            "лети отсюда",
            "arbitrated:semantic_reasoning",
            {"Я не могу лететь."},
        ),
        (
            "сбегай за сушняком",
            "arbitrated:semantic_reasoning",
            {"Я не могу сбегать за сушняком."},
        ),
    ),
)
async def test_grounded_local_replies_survive_autonomous_provider_failure(
    prompt: str,
    expected_source: str,
    expected_texts: set[str],
) -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content=f"<@123> {prompt}"))

    assert prepared is not None
    assert prepared.response_source == expected_source
    assert prepared.text in expected_texts
    assert planner.calls == 0


async def test_scaffolded_statement_defers_to_autonomous_semantic_planner() -> None:
    class StatementPlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs) -> ConsultationResult:
            self.calls.append(kwargs)
            return ConsultationResult.from_payload(
                {
                    "interpretation": "Пользователь сообщил утверждение.",
                    "intent": "engage_statement",
                    "response_plans": [
                        {
                            "move": "engage",
                            "stance": "direct",
                            "propositions": [
                                {
                                    "predicate": "рассматривать",
                                    "predicate_features": {"tense": "present"},
                                    "arguments": [
                                        {
                                            "role": "subject",
                                            "lemma": "я",
                                            "part_of_speech": "pronoun",
                                        },
                                        {
                                            "role": "direct_object",
                                            "lemma": "версия",
                                            "part_of_speech": "noun",
                                        },
                                    ],
                                    "clause_kind": "statement",
                                    "evidence_ids": ["personality.current"],
                                }
                            ],
                            "confidence": 0.91,
                        }
                    ],
                    "confidence": 0.91,
                },
                provider="fake-semantic-planner",
                model="semantic-only",
            )

    planner = StatementPlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> завтра будет четверг")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:autonomous_semantic_plan"
    assert not prepared.text.endswith("?")
    assert planner.calls


async def test_colloquial_capability_follow_up_uses_structured_previous_answer() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    previous = event(
        mentioned=False,
        author_is_bot=True,
        actor_id=123,
        event_id="discord:message:previous-capability-boundary",
        content="Я не могу лететь.",
    )
    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> а чего не можешь то а?")
    )

    assert prepared is not None
    assert prepared.text == "У меня нет тела."
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert planner.calls == 0

    grounding = dict(reply_module._previous_answer_grounding(("agent: Я не могу лететь.",)))
    assert "previous_answer.statement" in grounding
    assert "previous_answer.capability_boundary" in grounding


async def test_contextual_intent_is_not_suppressed_by_external_planning_mode() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    prior_user = event(
        mentioned=False,
        content="дол",
        event_id="discord:message:prior-user",
    )
    prior_bot = event(
        mentioned=False,
        author_is_bot=True,
        content="Что ты подразумеваешь под дол?",
        event_id="discord:message:prior-bot",
        actor_id=123,
    )
    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(prior_user, prior_bot)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="А ты что хочешь сказать?")
    )

    assert prepared is not None
    assert prepared.text == "Я хочу понять твою мысль."
    assert prepared.response_source == "arbitrated:dialogue_context"
    assert planner.calls == 0


async def test_recovery_does_not_promote_scaffold_below_the_quality_threshold() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> ну?"))

    assert prepared is not None
    assert prepared.text.endswith("?")
    assert "сформулировать" not in prepared.text.casefold()
    assert prepared.response_source == "arbitrated:semantic_recovery"
    assert planner.calls == 1


async def test_local_productive_reasoning_survives_external_plan_failure() -> None:
    class CapturingTraces:
        def __init__(self) -> None:
            self.steps: list[dict[str, object]] = []

        async def begin(self, *args, **kwargs) -> None:
            del args, kwargs

        async def step(self, *args, **kwargs) -> None:
            del args
            self.steps.append(kwargs)

    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            return None

    traces = CapturingTraces()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=UnavailablePlanner(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        response_traces=traces,  # type: ignore[arg-type]
        enabled=True,
    )

    directive = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> мусорку выкинь",
        )
    )
    fragment = await service.prepare_reply(
        event(
            mentioned=True,
            content="<@123> алеу",
        )
    )

    assert directive is not None, [
        (step.get("component"), step.get("status"), step.get("details"))
        for step in traces.steps
        if step.get("component")
        in {"response_sources", "response_arbitrator", "safe_reply_service"}
    ]
    assert directive.text == "Я не могу выкинуть мусорку.", directive.text.encode("unicode_escape")
    assert directive.response_source == "arbitrated:semantic_reasoning"
    assert fragment is not None
    assert "подразумеваешь" not in fragment.text.casefold()
    assert fragment.response_source.startswith("arbitrated:")
    assert fragment.response_source != "arbitrated:semantic_recovery"


@pytest.mark.parametrize("prompt", ["але", "алло?", "эй", "ну ау", "привет"])
async def test_contact_does_not_enter_lexical_clarification(prompt):
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        productive_only_replies=True,
        arbitration_mode="active",
        enabled=True,
    )
    prepared = await service.prepare_reply(event(mentioned=True, content=prompt))
    assert prepared is not None
    assert not prepared.text.endswith("?")
    assert "отвеч" in prepared.text.casefold()


@pytest.mark.parametrize("autonomous", [False, True])
async def test_name_question_uses_gateway_fact_in_productive_mode(autonomous):
    current = event(mentioned=True, content="Как тебя зовут?")
    current = replace(
        current,
        payload={
            **current.payload,
            "bot_identity": {"id": 123, "display_name": "DEBiLGI"},
        },
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        productive_only_replies=True,
        arbitration_mode="active",
        enabled=True,
    )
    service._autonomous_planning_required = autonomous
    prepared = await service.prepare_reply(current)
    assert prepared is not None
    assert "DEBiLGI" in prepared.text
    assert "подразумеваешь" not in prepared.text
    assert prepared.response_source == "arbitrated:discord_identity"


async def test_generated_definition_question_registers_a_learning_obligation():
    questions = FakeOpenQuestions()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        open_questions=questions,
        enabled=True,
    )
    current = event(mentioned=True, content="кваргония")
    prepared = PreparedReply(
        text="Что ты подразумеваешь под «кваргония»?",
        intent_id="intent:test",
        tokens=(),
        genome_signature="test",
        candidate_count=1,
        fallback=True,
        response_source="arbitrated:semantic_recovery",
    )
    await service.record_sent(current, response_message_id=99, reply=prepared, created_at=NOW)
    assert questions.opened == [("кваргония", 99)]


async def test_voluntary_lesson_changes_future_reply_after_restart_without_question(tmp_path):
    db = Database(
        tmp_path / "learning-cycle.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await db.initialize()
    try:
        lexicon = LexiconService(db)
        questions = OpenLanguageQuestionService(db, lexicon, SelfDatasetRepository(db))
        experience = SimpleNamespace(
            transition_adjustments=EXPERIENCE.transition_adjustments,
            register_sent=AsyncMock(),
        )

        def service():
            return SafeMentionReplyService(
                SimpleNamespace(intent_for_event=AsyncMock(return_value=intent())),
                FakeMemory(),
                FakeArchiveRepository(),
                EXPRESSION_ENGINE,
                experience,
                INITIATIVE,
                FixedGenerator(""),
                lexicon=lexicon,
                open_questions=questions,
                productive_only_replies=True,
                arbitration_mode="active",
                enabled=True,
            )

        current = event(mentioned=True, content="кваргония")
        await EventRepository(db).append(current)
        await lexicon.observe(current)
        bot = service()
        question = await bot.prepare_reply(current)
        assert question is not None and not reply_module.requests_word_definition(question.text)
        assert await questions.active_for(1, 10, 100) is None
        lesson = event(mentioned=True, content="Кваргония это настольная игра", event_id="lesson")
        await EventRepository(db).append(lesson)
        await lexicon.observe(lesson)
        await db.close()
        await db.initialize()
        lexicon = LexiconService(db)
        questions = OpenLanguageQuestionService(db, lexicon, SelfDatasetRepository(db))
        followup = event(mentioned=True, content="Что такое кваргония?", event_id="followup")
        await EventRepository(db).append(followup)
        answer = await service().prepare_reply(followup)
        assert answer is not None
        assert "настольной игрой" in answer.text
        assert "подразумеваешь" not in answer.text
    finally:
        await db.close()


@pytest.mark.parametrize(
    "word,meaning,expected",
    [
        ("кваргония", "настольная игра", "настольной игрой"),
        ("спаржа", "овощ", "овощем"),
    ],
)
async def test_dictionary_outage_defers_original_reply_then_found_meaning_answers_it(
    tmp_path,
    word,
    meaning,
    expected,
):
    db = Database(tmp_path / "research.sqlite3", Path(__file__).resolve().parents[2] / "migrations")
    await db.initialize()

    class ReferenceClient:
        available = False

        async def lookup(self, term):
            if term != word:
                return None
            if not self.available:
                raise OSError("temporary reference outage")
            return RemoteLexeme(term, "NOUN", (meaning,), "https://example.test/dictionary")

        async def close(self):
            pass

    client = ReferenceClient()
    lexicon = LexiconService(db)
    reference = RussianReferenceService(db, lexicon, client, enabled=True)
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        lexicon=lexicon,
        russian_reference=reference,
        productive_only_replies=True,
        arbitration_mode="active",
        enabled=True,
    )
    current = event(mentioned=True, content=f"Что такое {word}?")
    try:
        await EventRepository(db).append(current)
        with pytest.raises(LexicalResearchPending) as wait:
            await service.prepare_guaranteed_reply(current)
        assert word in wait.value.terms
        assert 1 <= wait.value.retry_after <= 300
        client.available = True
        await db.execute(
            "UPDATE language_reference_lookups SET checked_at='2000-01-01T00:00:00+00:00'",
        )
        prepared = await service.prepare_guaranteed_reply(current)
        assert prepared is not None
        assert expected in prepared.text
        assert not reply_module.requests_word_definition(prepared.text)
        assert (await lexicon.analyze_word(word)).has_usable_meaning
    finally:
        await reference.stop()
        await db.close()


async def test_bare_reply_label_is_digested_as_evaluation_not_definition_probe() -> None:
    parent = event(
        mentioned=False,
        author_is_bot=True,
        content="Я ответил бессвязно.",
        event_id="discord:message:parent",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(parent,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="гений",
            reply_to_message_id=99,
            target_id=123,
            bot_user_id=123,
        )
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert "подразумеваешь" not in prepared.text.casefold()


async def test_negated_knowledge_reply_reaches_productive_self_state_answer() -> None:
    parent = event(
        mentioned=False,
        author_is_bot=True,
        content="Что ты подразумеваешь под гений?",
        event_id="discord:message:parent",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(parent,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="а ты будто не знаешь",
            reply_to_message_id=99,
            target_id=123,
            bot_user_id=123,
        )
    )

    assert prepared is None  # Familiar words do not establish an agent's knowledge state.


async def test_addressed_statement_gets_content_bound_continuation_at_live_threshold() -> None:
    parent = event(
        mentioned=False,
        author_is_bot=True,
        content="Что ты подразумеваешь под гений?",
        event_id="discord:message:parent",
        actor_id=123,
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(parent,)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content="завтра будет четверг",
            reply_to_message_id=99,
            target_id=123,
            bot_user_id=123,
        )
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:dialogue_move"
    assert "четверг" in prepared.text.casefold()
    assert "мысль" not in prepared.text.casefold()
    assert "тему" not in prepared.text.casefold()


@pytest.mark.parametrize("learned_penalty", [False, True])
async def test_grounded_clarification_survives_learned_failure_penalties(
    learned_penalty: bool,
) -> None:
    class NegativeFeedbackRepository:
        async def profile(self, guild_id):
            return ArbitrationProfile.baseline(guild_id)

        async def structural_weights(self, guild_id):
            return {"source:semantic_recovery": -100.0}

        async def save_decision(self, *args, **kwargs):
            pass

    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            return None

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator("Повтори запрос: сейчас ответ не отправился."),
        semantic_consultant=UnavailablePlanner(),  # type: ignore[arg-type]
        arbitration_repository=NegativeFeedbackRepository() if learned_penalty else None,
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> гречка сколько стоит?")
    )

    assert prepared is None


def test_question_loop_gate_rejects_evasive_questions_after_user_objection() -> None:
    issue = SafeMentionReplyService._question_response_issue(
        "Что ты думаешь о космосе?",
        response_obligation="continue_topic",
        dialogue_messages=("Ты что специально задаешь вопросы?",),
        recent_self_replies=("Что ты думаешь о времени?",),
        metadata={"response_move": "explore_assertion"},
    )

    assert issue == "question_loop_after_user_objection"


async def test_productive_motion_question_answers_instead_of_staying_silent() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            return None

    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=UnavailablePlanner(),  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content=("<@123> вагонетка мчится налево, откуда она появится на экране сперва"),
        )
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:spatial_dynamics"
    assert prepared.text == "Вагонетка появится справа."
    assert "не могу" not in prepared.text.casefold()


async def test_productive_motion_follow_up_explains_its_previous_inference() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    premise = event(
        mentioned=True,
        content=("<@123> вагонетка мчится налево, откуда она появится на экране сперва"),
        event_id="discord:message:motion-premise",
    )
    answer = event(
        mentioned=False,
        author_is_bot=True,
        content="Вагонетка появится справа.",
        event_id="discord:message:motion-answer",
        actor_id=123,
    )
    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(premise, answer)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> почему же оттуда поедет")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:spatial_dynamics"
    assert "потому что" in prepared.text.casefold()
    assert "налево" in prepared.text.casefold()
    assert "не могу" not in prepared.text.casefold()
    assert planner.calls == 0


async def test_productive_rate_question_is_solved_without_external_planner() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content=(
                "<@123> а если машина едет со скоростью 200 км/ч, когда она доедет 654 километра"
            ),
        )
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:quantitative_reasoning"
    assert prepared.text == "Машина доедет за 3 часа 16 минут 12 секунд."
    assert planner.calls == 0


async def test_productive_scene_inquiry_reasons_before_answering() -> None:
    class UnavailablePlanner:
        autonomous_planning = True
        trigger_threshold = 0.72

        def __init__(self) -> None:
            self.calls = 0

        def should_consult(self, **kwargs) -> bool:
            del kwargs
            return False

        async def plan_response(self, **kwargs):
            del kwargs
            self.calls += 1
            return None

    planner = UnavailablePlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(
        event(mentioned=True, content="<@123> волк бежит за зайцем")
    )

    assert prepared is not None
    assert prepared.response_source == "arbitrated:semantic_inquiry"
    assert "волк преследует зайца" in prepared.text.casefold()
    assert "заяц убегает от волка" in prepared.text.casefold()
    assert "похоже" in prepared.text.casefold()
    assert "вероятно" in prepared.text.casefold()
    assert planner.calls == 0


@pytest.mark.parametrize(
    "question,answer,expected",
    [
        (
            "Что ты подразумеваешь под «салфетка»?",
            "Это бумага такая для гигиены",
            "Салфетка является бумагой для гигиены?",
        ),
        ("Кто открыл дверь?", "Маша", "Маша открыла дверь?"),
        ("Что ты открыл?", "Дверь", "Ты открыл дверь?"),
        ("Где лежит ключ?", "На столе", "Ключ лежит на столе?"),
    ],
)
async def test_user_definition_answers_previous_lexical_question_productively(
    question,
    answer,
    expected,
) -> None:
    previous = event(
        mentioned=False,
        author_is_bot=True,
        actor_id=123,
        content=question,
        event_id="discord:message:definition-question",
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
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )
    prepared = await service.prepare_reply(
        event(
            mentioned=True,
            content=f"<@123> {answer}",
        )
    )
    assert prepared is not None
    assert prepared.text == expected


@pytest.mark.parametrize(
    "report,prompt,expected",
    [
        ("Я вчера потерял зюзюбрик", "Когда я его потерял?", "вчера"),
        ("Я потерял зюзюбрик в парке", "Где я его потерял?", "в парке"),
    ],
)
async def test_event_slot_answer_does_not_require_unknown_object_definition(
    report,
    prompt,
    expected,
):
    previous = replace(
        event(mentioned=True, content=report, event_id="prior", bot_user_id=123),
        timestamp=NOW - timedelta(seconds=5),
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
        productive_only_replies=True,
        enabled=True,
    )
    prepared = await service.prepare_reply(event(mentioned=True, content=prompt, bot_user_id=123))
    assert prepared is not None
    assert expected in prepared.text
    assert "зюзюбрик" in prepared.text
    assert not prepared.text.endswith("?")


async def test_unknown_event_object_does_not_trigger_foreground_dictionary_question():
    lexicon = UnknownLexicon()
    reference = SimpleNamespace(refresh_word=AsyncMock(return_value=None))
    current = event(mentioned=True, content="Я вчера потерял зюзюбрик", bot_user_id=123)
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        lexicon=lexicon,
        russian_reference=reference,
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )
    prepared = await service.prepare_reply(current)
    assert prepared is not None
    assert "подразумеваешь" not in prepared.text
    assert "зюзюбрик" in prepared.text
    reference.refresh_word.assert_not_awaited()


async def test_claim_explanation_and_challenge_keep_context_across_turns():
    class ConversationRepository(FakeRepository):
        async def intent_for_event(self, event_id):
            return self.stored_intent

    memory = FakeMemory(
        channel_events=(
            event(
                mentioned=False,
                author_is_bot=True,
                actor_id=123,
                content="Да, считаю я себя хищником.",
                event_id="discord:message:initial-claim",
            ),
        )
    )
    service = SafeMentionReplyService(
        ConversationRepository(intent()),
        memory,
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        arbitration_mode="active",
        productive_only_replies=True,
        enabled=True,
    )
    for index, prompt in enumerate(
        (
            "А почему так считаешь?",
            "Ну типо почему ты так себя представляешь",
            "Но ты сказал, что ты хищник",
        )
    ):
        current = event(
            mentioned=True, content=f"<@123> {prompt}", event_id=f"discord:message:followup-{index}"
        )
        prepared = await service.prepare_reply(current)
        if index < 2:
            assert prepared is None  # No stored reason for the earlier claim.
            memory.channel_events = (*memory.channel_events, current)
            continue
        assert prepared is not None
        assert "подразумеваешь" not in prepared.text
        assert not prepared.text.endswith("?")
        if index == 2:
            assert "хищником" in prepared.text
            assert "говорил" in prepared.text
        memory.channel_events = (
            *memory.channel_events,
            current,
            event(
                mentioned=False,
                author_is_bot=True,
                actor_id=123,
                content=prepared.text,
                event_id=f"discord:message:answer-{index}",
            ),
        )


@pytest.mark.parametrize(
    "prompt,expected_goal",
    [
        ("А почему так считаешь?", "reason"),
        ("Ну типо почему ты так себя представляешь", "reason"),
        ("Но ты сказал, что ты хищник", "review"),
    ],
)
@pytest.mark.parametrize("autonomous", [False, True])
async def test_own_claim_followups_do_not_degrade_to_word_definitions(
    prompt,
    expected_goal,
    autonomous,
):
    previous = event(
        mentioned=False,
        author_is_bot=True,
        actor_id=123,
        content="Да, считаю я себя хищником.",
        event_id="discord:message:own-claim",
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
        arbitration_minimum_score=0.45,
        productive_only_replies=True,
        enabled=True,
    )
    service._autonomous_planning_required = autonomous
    prepared = await service.prepare_reply(event(mentioned=True, content=f"<@123> {prompt}"))
    if expected_goal == "reason":
        assert prepared is None
        return
    assert prepared is not None
    assert "подразумеваешь" not in prepared.text
    assert not prepared.text.endswith("?")
    if expected_goal == "review":
        assert "хищником" in prepared.text
        assert "говорил" in prepared.text
        assert "обоснован" not in prepared.text  # Do not append a fabricated epistemic report.
    else:
        assert any(term in prepared.text for term in ("причин", "основан"))
        assert prepared.text != "Нет, я не представляю себя."


async def test_emotional_clarification_repairs_runtime_answer_instead_of_reusing_it() -> None:
    previous_question = event(
        mentioned=True,
        content="<@123> у тебя сейчас какая раскладка языка стоит?",
        event_id="discord:message:previous-layout-question",
    )
    previous_bad_answer = event(
        mentioned=False,
        author_is_bot=True,
        content="А я сейчас вам покажу, откуда на Беларусь готовилось нападение.",
        event_id="discord:message:previous-bad-layout-answer",
        actor_id=123,
    )
    planner = FakeAutonomousPlanner()
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(channel_events=(previous_question, previous_bad_answer)),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,
        FixedGenerator(""),
        semantic_consultant=planner,  # type: ignore[arg-type]
        arbitration_mode="active",
        arbitration_minimum_score=0.0,
        productive_only_replies=True,
        enabled=True,
    )

    prepared = await service.prepare_reply(event(mentioned=True, content="<@123> чего??))"))

    assert prepared is not None
    assert prepared.text == (
        "Я не ответила по теме. Я не наблюдаю системную раскладку. Сообщение приходит по-русски."
    )
    assert prepared.response_source == "arbitrated:autonomous_semantic_plan"
    assert "Беларус" not in prepared.text
    assert "КОПИРОВАТЬ" not in prepared.text
