from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from debilgi.autonomy.models import InitiativeDecision
from debilgi.cognition.dialogue_tasks import DialogueTaskRepository
from debilgi.cognition.models import ActionIntent, ActionType
from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.discord.replies import SafeMentionReplyService, _stable_response_seed
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.personality.expression import PersonalityExpressionEngine
from debilgi.personality.genome import DEBILGI_GENOME
from debilgi.storage.database import Database

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


async def test_observed_social_exchange_enables_meaning_first_without_a_script(tmp_path):
    from debilgi.language.discourse import DiscourseMemoryRepository
    from debilgi.language.semantic_assimilation import SemanticAssimilationService

    database = Database(tmp_path / "observed-social.sqlite3", Path("migrations"))
    await database.initialize()
    try:
        assimilation = SemanticAssimilationService(database, SelfDatasetRepository(database))
        generator = Mock()
        repository = FakeRepository(intent())
        repository.intent_for_event = AsyncMock(return_value=intent())
        service = SafeMentionReplyService(
            repository, FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            enabled=True,
            semantic_assimilation=assimilation,
            dialogue_tasks=DialogueTaskRepository(database),
        )
        query = event(mentioned=True, content="ку бро", event_id="query:empty")
        await EventRepository(database).append(query)
        assert await service.prepare_guaranteed_reply(query) is None
        for index, (actor, text) in enumerate((
            (100, "Привет!"), (200, "Здравствуй!"), (100, "привет"), (200, "ну привет!!!"),
        )):
            observed = replace(
                event(mentioned=False, actor_id=actor, content=text,
                      event_id=f"observed:{index}"),
                timestamp=NOW - timedelta(minutes=2) + timedelta(seconds=index),
            )
            observed = replace(observed, payload={**observed.payload, "message_id": index + 1})
            await EventRepository(database).append(observed)
            await DiscourseMemoryRepository(database).observe(observed)
            await assimilation.dialogue_learning.observe(observed)
        query = replace(query, event_id="query:learned", timestamp=NOW + timedelta(seconds=5))
        await EventRepository(database).append(query)
        prepared = await service.prepare_guaranteed_reply(query)
        assert prepared is not None
        assert "здравствуй" in prepared.text or "привет" in prepared.text
        assert "здорово" not in prepared.text.casefold()  # avoid praise/greeting ambiguity
        assert prepared.response_source == "arbitrated:learned_dialogue_structure"
        generator.generate.assert_not_called()
    finally:
        await database.close()


async def test_clear_question_does_not_use_a_learned_lexical_repair(tmp_path):
    from debilgi.experience.meaning_intents import MeaningIntentRepository
    from debilgi.language.discourse import DiscourseMemoryRepository
    from debilgi.language.semantic_assimilation import SemanticAssimilationService

    database = Database(tmp_path / "intent-repair.sqlite3", Path("migrations"))
    await database.initialize()
    try:
        assimilation = SemanticAssimilationService(database, SelfDatasetRepository(database))
        meanings = MeaningIntentRepository(database)
        repository = FakeRepository(intent())
        repository.intent_for_event = AsyncMock(return_value=intent())
        generator = Mock()
        generator.generate.side_effect = AssertionError("script generator must not run")
        service = SafeMentionReplyService(
            repository, FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            enabled=True,
            semantic_assimilation=assimilation, meaning_intents=meanings,
            dialogue_tasks=DialogueTaskRepository(database),
        )
        for index, (actor, text) in enumerate((
            (100, "Я читаю книгу."), (200, "В смысле, читаешь или пишешь?"),
        )):
            observed = replace(
                event(mentioned=False, actor_id=actor, content=text,
                      event_id=f"observed:{index}"),
                timestamp=NOW - timedelta(minutes=2) + timedelta(seconds=index),
            )
            observed = replace(observed, payload={**observed.payload, "message_id": index + 1})
            await EventRepository(database).append(observed)
            await DiscourseMemoryRepository(database).observe(observed)
            await assimilation.dialogue_learning.observe(observed)
        query = event(mentioned=True, content="Ты любишь собак?", event_id="query:repair")
        await EventRepository(database).append(query)
        prepared = await service.prepare_guaranteed_reply(query)
        # Low interlocutor confidence does not make a clear question ambiguous.
        # These observations teach only a lexical echo; they supply no own fact
        # and no grammatical epistemic construction for this minimal service.
        assert prepared is None
        for actor in (100, 200):
            bot_turn = event(mentioned=True, content="В смысле?", actor_id=actor,
                             author_is_bot=True)
            assert await service.prepare_guaranteed_reply(bot_turn) is None
            assert await service.prepare_initiative(bot_turn) is None
        generator.generate.assert_not_called()
    finally:
        await database.close()


async def test_disabled_reply_service_records_a_withheld_reason(tmp_path):
    from debilgi.language.response_trace import ResponseTraceRepository

    database = Database(tmp_path / "disabled.sqlite3", Path("migrations"))
    await database.initialize()
    try:
        current = event(mentioned=True, content="привет")
        await EventRepository(database).append(current)
        traces = ResponseTraceRepository(database)
        service = SafeMentionReplyService(
            FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            enabled=False, response_traces=traces,
        )
        assert await service.prepare_guaranteed_reply(current) is None
        steps = await traces.steps_for_event(current.event_id)
        assert steps[-1]["status"] == "withheld"
        assert steps[-1]["details"]["reason"] == "replies_disabled"
    finally:
        await database.close()


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

            dialogue_tasks=tasks,
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
            ExperienceConsolidator(ExperienceRepository(database)), INITIATIVE,
             proposition_memory=memory, meaning_intents=learned,
            enabled=True,
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
        # One approval cannot supply a desire or a dislike of being shut down.
        assert reply is None or reply.text == "Я не знаю, хочу ли я быть выключенным."
        assert not await learned.learned_goals(desire_prompt)
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
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            proposition_memory=memory, enabled=True)
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
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        knowledge_service=SimpleNamespace(
            grounding_for_event=AsyncMock(return_value=((fact,), None)),
        ),
         enabled=True,
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
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="ты молчишь"),
    )
    assert prepared is None
    generator.generate.assert_not_called()


async def test_attitude_statement_without_a_learned_exchange_is_withheld(tmp_path):
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

            dialogue_tasks=DialogueTaskRepository(database),
            enabled=True,
        )

        prepared = await service.prepare_guaranteed_reply(current)

        assert prepared is None
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
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        knowledge_service=SimpleNamespace(grounding_for_event=AsyncMock(
            return_value=((fact,), None))),
        enabled=True,
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
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            proposition_memory=PropositionMemory(database),
            knowledge_service=SimpleNamespace(grounding_for_event=AsyncMock(
                return_value=(("Маша открыла дверь.",), None))),
            enabled=True,
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
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
            proposition_memory=PropositionMemory(database),
            enabled=True,
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
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        open_questions=questions,
        enabled=True,
    )
    assert not hasattr(service, "prepare_bound_explanation")
    assert not hasattr(service, "prepare_bound_short_act")
    questions.resolution_for_event.assert_not_awaited()
    short_acts.resolution_for_event.assert_not_awaited()


@pytest.mark.parametrize("prompt", [
    "ауу", "Эм", "зюзюбрик", "ты молчишь", "Почему робот скучает?",
    "напиши что-нибудь странное", "если бы рыба умела летать, что бы она делала",
])
async def test_language_graph_without_reactive_pairs_withholds_social_content(tmp_path, prompt):
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
            EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,

            proposition_memory=memory,
            dialogue_tasks=DialogueTaskRepository(database),
            productive_only_replies=True,
            enabled=True,
        )
        service._defer_pending_research = AsyncMock(
            side_effect=AssertionError("reply must not wait")
        )
        prepared = await service.prepare_guaranteed_reply(current)
        # These are isolated corpus sentences, not demonstrations of how to
        # react. Their first-person content cannot become the agent's activity.
        assert prepared is None
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
            event(mentioned=False, event_id="cause:robot",
                  content="Робот скучает, потому что человек молчит."),
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

            proposition_memory=memory,
            dialogue_tasks=tasks,
            enabled=True,
        )

        prepared = await service.prepare_guaranteed_reply(current)

        assert prepared is not None
        assert prepared.response_source != "arbitrated:semantic_recovery"
        assert {"робот", "скучает"} <= {
            word.strip(".,!?") for word in prepared.text.casefold().split()
        }
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
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        productive_only_replies=True, enabled=True,
    )

    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="буу"),
    )

    assert prepared is None
    generator.generate.assert_not_called()


async def test_stretched_contact_opens_a_new_episode_without_inventing_language():
    previous = replace(
        event(mentioned=True, content="ебать, а я ногти грызу", event_id="prior", bot_user_id=123),
        timestamp=NOW - timedelta(seconds=5),
    )
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(channel_events=(previous,)), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        productive_only_replies=True, enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="ауу", bot_user_id=123),
    )
    # No observed contact response exists in this empty language memory.
    assert prepared is None


@pytest.mark.parametrize("upstream_failure", [None, "event_processing_timeout"])
async def test_direct_reply_without_intent_or_language_data_does_not_invent_a_script(
    upstream_failure,
):
    service = SafeMentionReplyService(
        FakeRepository(None), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        enabled=True,
    )
    prepared = await service.prepare_guaranteed_reply(
        event(mentioned=True, content="привет"), upstream_failure=upstream_failure,
    )
    assert prepared is None


async def test_local_generation_recovers_from_context_failure_without_a_canned_phrase():
    service = SafeMentionReplyService(
        FakeRepository(intent()), FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, EXPERIENCE, INITIATIVE,
        enabled=True,
    )
    service._reply_context = AsyncMock(side_effect=RuntimeError("context unavailable"))
    prepared = await service.prepare_guaranteed_reply(event(mentioned=True, content="зюзюбрик"))
    assert prepared is None


def test_response_seed_depends_on_semantics_not_discord_event_id() -> None:
    first = _stable_response_seed("Что это?", ("мой компьютер решил",))
    repeated = _stable_response_seed("Что это?", ("мой компьютер решил",))
    changed_context = _stable_response_seed("Что это?", ("ворон достал ключ",))

    assert first == repeated
    assert first != changed_context


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


class FakeExperienceConsolidator:
    pass


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


async def test_service_ignores_plain_messages_and_bot_authors() -> None:
    service = SafeMentionReplyService(
        FakeRepository(intent()),
        FakeMemory(),
        FakeArchiveRepository(),
        EXPRESSION_ENGINE,
        EXPERIENCE,
        INITIATIVE,

        enabled=True,
    )

    assert await service.response_for(event(mentioned=False)) is None
    assert await service.response_for(event(mentioned=True, author_is_bot=True)) is None


@pytest.fixture
async def reactive_service(tmp_path):
    from debilgi.experience.meaning_intents import MeaningIntentRepository
    from debilgi.language.discourse import DiscourseMemoryRepository
    from debilgi.language.response_trace import ResponseTraceRepository
    from debilgi.language.semantic_assimilation import SemanticAssimilationService

    db = Database(tmp_path / "reactive.sqlite3", Path("migrations"))
    await db.initialize()
    assimilation = SemanticAssimilationService(db, SelfDatasetRepository(db))
    generator = Mock()
    generator.generate.side_effect = AssertionError("unlearned replies are forbidden")
    repository = FakeRepository(intent())
    repository.intent_for_event = AsyncMock(return_value=intent())
    consolidator = FakeExperienceConsolidator()
    consolidator.register_sent = AsyncMock()
    service = SafeMentionReplyService(
        repository, FakeMemory(), FakeArchiveRepository(),
        EXPRESSION_ENGINE, consolidator, INITIATIVE,
        semantic_assimilation=assimilation, meaning_intents=MeaningIntentRepository(db),
        response_traces=ResponseTraceRepository(db), dialogue_tasks=DialogueTaskRepository(db),
        enabled=True,
    )

    async def demonstrate(prompt, answer, *, start=0):
        for i, (actor, text) in enumerate(((1000, prompt), (2000, answer)), start=start):
            observed = replace(
                event(mentioned=False, actor_id=actor, content=text, event_id=f"demo:{i}"),
                timestamp=NOW - timedelta(minutes=3) + timedelta(seconds=i),
            )
            observed = replace(observed, payload={**observed.payload, "message_id": i + 1})
            await EventRepository(db).append(observed)
            await DiscourseMemoryRepository(db).observe(observed)
            await assimilation.dialogue_learning.observe(observed)

    async def ask(text, *, n=0, actor=100, mentioned=True):
        query = replace(event(mentioned=mentioned, content=text, actor_id=actor,
                              event_id=f"query:{n}"), timestamp=NOW + timedelta(seconds=n * 20))
        await EventRepository(db).append(query)
        return query, await service.prepare_guaranteed_reply(query)

    yield service, assimilation.dialogue_learning, demonstrate, ask
    generator.generate.assert_not_called()
    await db.close()


@pytest.mark.parametrize("prompt,answer,query,expected", [
    ("лошара", "сам такой долбоеб", "лошара", "сам такой долбоеб"),
    ("ты тупой", "сам такой))", "ты тупой", "сам такой))"),
    ("ну ты бля крутой", "ну а то))", "ну ты бля крутой", "ну а то))"),
    ("красавчик", "ну а то))", "красавчик", "ну а то))"),
    ("я читаю книгу", "книгу? какую?", "я читаю журнал", "журнал? какой?"),
    ("мне нравится рисовать", "ого))", "мне нравится лепить", "ого))"),
    ("я сегодня работал весь день", "жиза", "я сегодня работал весь день", "жиза"),
    ("как дела", "а у тебя?", "как дела", "а у тебя?"),
    ("скучно", "эх))", "скучно", "эх))"),
    ("ахахах", "ахах))", "ахахах", "ахах))"),
])
async def test_reactive_move_requires_an_actual_human_exchange(
    reactive_service, prompt, answer, query, expected,
):
    service, _, demonstrate, ask = reactive_service
    _, cold = await ask(query)
    assert cold is None
    await demonstrate(prompt, answer)
    await demonstrate("я сегодня танцевал", "ну здорово!!!", start=10)
    event_, warm = await ask(query, n=1)
    from debilgi.language.dialogue_learning import normalized_response

    assert warm
    # Atomic interjections and laughter can coincide with a learned lexical act.
    # Contentful source utterances still cannot be copied wholesale.
    if query not in {"скучно", "ахахах", "мне нравится лепить"}:
        assert normalized_response(warm.text) not in {
            normalized_response(answer), normalized_response("ну здорово!!!"),
        }
    if "журнал" in expected:
        assert "журнал" in warm.text or "какой" in warm.text
        assert "какую" not in warm.text
    assert warm.response_source == "arbitrated:learned_dialogue_structure"
    diagnostic = await service._response_traces.diagnostics_for_event(
        event_.event_id, status="prepared", final_text=warm.text,
        selected_route=warm.response_source,
    )
    assert diagnostic["category"] == "social"
    assert "demo:1" in diagnostic["content"]["evidence_ids"]
    assert diagnostic["content"]["reaction_meaning"]
    assert diagnostic["content"]["reaction_parts"]
    assert diagnostic["content"]["exchange_pattern_id"].startswith("exchange-composition:")


@pytest.mark.parametrize("hook_text", ["какую?", "какую"])
async def test_reactive_repeat_hook_budget_actor_and_episode_isolation(reactive_service, hook_text):
    service, learning, demonstrate, ask = reactive_service
    await demonstrate("лошара", "сам такой долбоеб))")
    await demonstrate("лошара", "ну сам такой", start=10)
    first, reply = await ask("лошара")
    assert reply
    await service.record_sent(first, response_message_id=700, reply=reply,
                              created_at=NOW + timedelta(seconds=1))
    second, alternative = await ask("лошара", n=1)
    assert alternative and alternative.text != reply.text
    # Prepared text alone is not a sent form and spends no episode budget.
    history = await learning.recent_reactions(second)
    assert len(history) == 1 and history[0][0] == reply.text
    assert history[0][1].startswith("exchange-composition:")
    assert not await learning.recent_reactions(replace(second, actor_id=999))
    assert not await learning.recent_reactions(replace(second, channel_id=999))
    assert not await learning.recent_reactions(replace(second, timestamp=NOW + timedelta(hours=1)))

    await demonstrate("я читаю книгу", hook_text, start=20)
    hook_event, hook = await ask("я читаю книгу", n=2)
    assert hook and "какую" in hook.text
    await service.record_sent(hook_event, response_message_id=701, reply=hook,
                              created_at=hook_event.timestamp + timedelta(seconds=1))
    _, repeated_hook = await ask("я читаю книгу", n=3)
    assert repeated_hook is None
    _, unaddressed = await ask("лошара", n=4, mentioned=False)
    assert unaddressed is None
    _, nonsense = await ask("фырпырвззз", n=5)
    assert nonsense is None


async def test_sent_greeting_can_be_explained_only_to_its_recipient(reactive_service):
    service, learning, demonstrate, ask = reactive_service
    await demonstrate("привет", "ну привет!")
    await demonstrate("приветствую", "привет", start=10)
    first, greeting = await ask("але")
    assert greeting and "здорово" not in greeting.text.casefold()
    follow, _ = await ask("и к чему это?", n=1)
    assert await learning.prior_response_for(follow) is None
    await service.record_sent(first, response_message_id=700, reply=greeting,
                              created_at=first.timestamp + timedelta(seconds=1))
    prior = await learning.prior_response_for(follow)
    assert prior and prior["response_text"] == greeting.text
    assert not await learning.prior_response_for(replace(follow, actor_id=999))
    assert not await learning.prior_response_for(replace(follow, channel_id=999))
    assert not await learning.prior_response_for(replace(
        follow, timestamp=follow.timestamp + timedelta(hours=1)))
    _, explanation = await ask("и к чему это?", n=2)
    assert explanation and "приветствовал" in explanation.text.casefold()


async def test_explicit_reply_edge_finds_the_delivered_turn_without_the_time_window(
    reactive_service,
):
    service, learning, demonstrate, ask = reactive_service
    await demonstrate("привет", "ну привет!")
    await demonstrate("приветствую", "привет", start=10)
    first, greeting = await ask("але")
    await service.record_sent(first, response_message_id=700, reply=greeting,
                              created_at=first.timestamp + timedelta(seconds=1))
    follow, _ = await ask("и к чему это?", n=1)
    later = replace(follow, timestamp=follow.timestamp + timedelta(hours=16))
    assert await learning.prior_response_for(later) is None
    answered = replace(later, payload={**later.payload, "reply_to_message_id": 700})
    prior = await learning.prior_response_for(answered)
    assert prior and prior["response_message_id"] == 700
    other = replace(answered, payload={**answered.payload, "reply_to_message_id": 701})
    assert await learning.prior_response_for(other) is None
    assert not await learning.prior_response_for(replace(answered, actor_id=999))
    assert not await learning.prior_response_for(replace(answered, channel_id=999))


async def test_learned_question_keeps_reported_past_tense(reactive_service):
    service, _, demonstrate, ask = reactive_service
    await demonstrate("я читаю книгу", "какую книгу читаешь?")
    await demonstrate("я гулял", "ты гулял сегодня?", start=10)
    current, prepared = await ask("я сегодня читал журнал")
    assert prepared and "читал" in prepared.text and "читаешь" not in prepared.text
    assert "журнал" in prepared.text and "сегодня" in prepared.text
    diagnostic = await service._response_traces.diagnostics_for_event(
        current.event_id, status="prepared", final_text=prepared.text,
        selected_route=prepared.response_source)
    assert diagnostic["content"]["reverse_check"]["passed"]


async def test_why_after_sent_unknown_reuses_its_scoped_query(reactive_service):
    service, learning, demonstrate, ask = reactive_service
    from debilgi.language.proposition_memory import PropositionMemory

    memory = PropositionMemory(learning._database)
    service._proposition_memory = memory
    await demonstrate("ты что делаешь?", "я не помню, что я делаю.")
    await memory.observe(await EventRepository(learning._database).get("demo:1"))
    first, unknown = await ask("че делаешь?")
    assert unknown and "не знаю" in unknown.text.casefold()
    await service.record_sent(first, response_message_id=700, reply=unknown,
                              created_at=first.timestamp + timedelta(seconds=1))
    current, explanation = await ask("почему?", n=1)
    # Stage 4.22b (В5): the retelling keeps the verb that was actually said («не знаю»);
    # the learned «не помню» is not what the delivered limit used.
    assert explanation and "не знаю" in explanation.text.casefold()
    assert "не помню" not in explanation.text.casefold()
    assert "делаю" in explanation.text.casefold()
    prior = await learning.prior_response_for(current)
    assert prior["response_text"] == unknown.text


async def test_self_possession_unknown_has_an_explicit_subject(reactive_service):
    service, learning, demonstrate, ask = reactive_service
    from debilgi.language.proposition_memory import PropositionMemory

    memory = PropositionMemory(learning._database)
    service._proposition_memory = memory
    await demonstrate("ты что делаешь?", "я не помню, что я делаю.")
    await memory.observe(await EventRepository(learning._database).get("demo:1"))
    _, response = await ask("а что есть у тебя?")
    assert response and "не знаю" in response.text.casefold()
    assert "у меня" in response.text.casefold()


@pytest.mark.parametrize("answer", [
    "я читаю книгу", "у меня трое детей", "я на море", "ты работаешь врачом",
    "мой сын врач", "в Москве", "я люблю рисовать", "я сегодня работал весь день",
    "дома", "вчера", "здесь",
    "ого круто ответственная работа", "круто ты врач",
])
async def test_reaction_never_imports_a_demonstrators_biography(reactive_service, answer):
    _, _, demonstrate, ask = reactive_service
    await demonstrate("как дела", answer)
    _, reply = await ask("как дела")
    assert reply is None


async def test_unbound_acknowledgement_transfers_but_formality_does_not(reactive_service):
    service, _, demonstrate, ask = reactive_service
    await demonstrate("прикольно", "да да)")
    e, bored = await ask("скучно")
    # An affirmation learned from approval does not acknowledge boredom.
    assert bored is None
    await demonstrate("красавчик", "ну а то))", start=20)
    await demonstrate("молодец", "и вам", start=10)
    e, compliment = await ask("красавчик", n=1)
    assert compliment and "вам" not in compliment.text.casefold()
    diagnostic = await service._response_traces.diagnostics_for_event(
        e.event_id, status="prepared", final_text=compliment.text,
        selected_route=compliment.response_source,
    )
    assert "demo:11" not in diagnostic["content"]["evidence_ids"]


@pytest.mark.parametrize("prompt", ["объясни, почему небо голубое", "так да или нет?",
                                   "Маша встретила Олю."])
async def test_committed_form_mode_is_not_replaced_by_a_later_answer_requirement(
    reactive_service, prompt,
):
    service, _, demonstrate, ask = reactive_service
    await demonstrate("что ещё расскажешь?", "ну не знаю")
    await demonstrate("прикольно", "ну ага", start=10)
    e, reply = await ask(prompt)
    # Raw interlocutor ignorance does not establish the bot's factual Unknown.
    assert reply is None


async def test_compositional_chat_learns_new_coarse_moves_and_never_learns_its_output(
    reactive_service,
):
    from debilgi.language.dialogue_learning import normalized_response

    service, learning, demonstrate, ask = reactive_service
    pairs = (
        ("ты тупой", "сам такой долбоеб))"),
        ("ты дурак", "ну сам такой"),
        ("красавчик", "ну а то))"),
        ("ты крутой", "да да!!!"),
    )
    for i, (prompt, answer) in enumerate(pairs):
        await demonstrate(prompt, answer, start=i * 10)
    sources = {normalized_response(answer) for _, answer in pairs}
    replies = []
    for n, query in enumerate(("ты очень тупой", "ну ты дурак", "ну ты бля крутой", "ты молодец")):
        event_, reply = await ask(query, n=n)
        assert reply and normalized_response(reply.text) not in sources
        assert len(reply.tokens) <= 9
        diagnostic = await service._response_traces.diagnostics_for_event(
            event_.event_id, status="prepared", final_text=reply.text,
            selected_route=reply.response_source,
        )
        content = diagnostic["content"]
        assert content["reaction_meaning"] == ("retort" if n < 2 else "agree")
        assert content["reaction_parts"]
        assert any(p["component"] != "core" for p in content["reaction_parts"])
        if n < 2:
            assert "сам" in reply.text
        else:
            assert ")" in reply.text or "!" in reply.text
        await service.record_sent(event_, response_message_id=900 + n, reply=reply,
                                  created_at=event_.timestamp + timedelta(seconds=1))
        replies.append(reply.text)
    # Receipts affect the episode history, never human demonstration support.
    rows = await learning._database.fetchall(
        "SELECT answer_event_id FROM language_observed_exchanges")
    assert all(r["answer_event_id"].startswith("demo:") for r in rows)
    assert len(rows) == 7  # includes the adjacent human edges between mini-chat pairs
    assert len({normalized_response(t) for t in replies[:2]}) == 2


@pytest.mark.parametrize("answer", [
    "привет. дела отлично. у тебя как?",
    "привет, всё хорошо. у тебя как?",
])
async def test_smalltalk_cannot_import_a_state_from_a_mixed_answer(reactive_service, answer):
    service, _, demonstrate, ask = reactive_service
    await demonstrate("как дела", answer)
    await demonstrate("я танцую", "ну здорово!", start=10)
    event_, reply = await ask("как дела")
    assert reply and "отлично" not in reply.text and "дела" not in reply.text
    diagnostic = await service._response_traces.diagnostics_for_event(
        event_.event_id, status="prepared", final_text=reply.text,
        selected_route=reply.response_source,
    )
    assert diagnostic["content"]["reaction_meaning"] == "reciprocal_question"
    await service.record_sent(event_, response_message_id=950, reply=reply,
                              created_at=event_.timestamp + timedelta(seconds=1))
    _, greeting = await ask("как дела", n=1)
    # A prepared question without a subsequent human continuation is not a
    # second greeting cue. Repeating the question spends its existing budget.
    assert greeting is None


async def test_reactions_follow_event_valence_and_unknown_condition_without_cheering(
    reactive_service,
):
    from debilgi.language.dialogue_learning import normalized_response

    service, _, demonstrate, ask = reactive_service
    pairs = (
        ("я провалил экзамен", "бля жесть!"),
        ("я потерял кошелёк", "очень печально."),
        ("как дела", "и как ты?"),
        ("я выиграл приз", "ого круто!!!"),
    )
    for i, (prompt, answer) in enumerate(pairs):
        await demonstrate(prompt, answer, start=i * 10)
    for n, query in enumerate(("я вчера провалил экзамен", "я потерял телефон", "я вчера болел")):
        event_, reply = await ask(query, n=n)
        assert reply
        assert normalized_response(reply.text) not in {normalized_response(a) for _, a in pairs}
        assert "ого" not in reply.text and "круто" not in reply.text and "!!" not in reply.text
        diagnostic = await service._response_traces.diagnostics_for_event(
            event_.event_id, status="prepared", final_text=reply.text,
            selected_route=reply.response_source,
        )
        assert diagnostic["content"]["part_search"] == "observed_part_grammar"
        if n < 2:
            assert diagnostic["content"]["reaction_meaning"] == "empathize"
            assert diagnostic["content"]["event_valence"] < 0
        else:
            assert diagnostic["content"]["reaction_meaning"] == "check_condition"
            assert diagnostic["content"]["event_valence"] is None
        # Separate episodes in this check keep the consecutive-hook guard independent.


async def test_attribute_ellipsis_binds_the_argument_of_a_delivered_question(reactive_service):
    service, learning, demonstrate, ask = reactive_service
    await demonstrate('я читаю книгу', 'какую книгу читаешь?')
    await demonstrate('почему?', 'я не знаю', start=10)
    first, question = await ask('я читаю газету')
    assert question and 'газету' in question.text and question.text.endswith('?')
    await service.record_sent(first, response_message_id=700, reply=question,
                              created_at=first.timestamp + timedelta(seconds=1))
    current, answer = await ask('так какую?', n=1)
    assert answer and 'не знаю' in answer.text.casefold()
    assert 'какую газету ты читаешь' in answer.text.casefold()
    diagnostics = await service._response_traces.steps_for_event(current.event_id)
    plans = [s['details'] for s in diagnostics if s['stage'] == 'candidate_generation']
    assert any('receipt:10:700' in str(p) for p in plans)
    assert not await learning._database.fetchone(
        "SELECT 1 FROM language_observed_exchanges WHERE answer_event_id=?", (current.event_id,))


async def test_single_unknown_has_a_second_move_from_its_actual_receipt(reactive_service):
    from debilgi.language.proposition_memory import PropositionMemory

    service, learning, demonstrate, ask = reactive_service
    service._proposition_memory = PropositionMemory(learning._database)
    await demonstrate("я читаю книгу", "ого!")
    first, answer = await ask("а ты кто?", n=80)
    assert answer and "не знаю" in answer.text
    await service.record_sent(first, response_message_id=700, reply=answer,
                              created_at=first.timestamp + timedelta(seconds=1))
    _, second = await ask("а если прикинуть кто ты", n=81)
    assert second and second.text != answer.text
    assert second.candidate_count >= 2
    assert "не знаю" in second.text


async def test_meaning_first_never_enters_compatibility_generators(reactive_service):
    service, _, demonstrate, ask = reactive_service
    await demonstrate("привет", "ну привет!")
    assert not hasattr(service, "_generate_legacy_checked")
    assert not hasattr(service, "_generate_arbitrated")
    _, reply = await ask("але", n=90)
    assert reply


async def _why_setup(reactive_service):
    from debilgi.language.discourse import DiscourseMemoryRepository

    service, learning, demonstrate, ask = reactive_service
    await demonstrate("привет", "ну привет!")
    await demonstrate("приветствую", "привет", start=10)
    greeted, _ = await ask("привет")
    observed = replace(greeted, payload={**greeted.payload, "message_id": 77})
    await DiscourseMemoryRepository(learning._database).observe(observed)
    return learning, ask, greeted


async def _affect_row(db, source, actor):
    await db.execute(
        "INSERT INTO affect_observations VALUES (?,?,?,?,?,?,?)",
        (source.event_id, source.guild_id, actor, "{}", "{}", "{}", source.timestamp.isoformat()))


async def _why(learning, ask, basis, n):
    from debilgi.language.planning import ReplyContext

    query, _ = await ask("почему?", n=n)
    context = ReplyContext(prompt="почему?", turn_prompt="почему?")
    prior = {"basis": basis, "operations": await _operations(learning, query)}
    return query, await learning.explain_delivered_basis(query, context, prior)


async def _operations(learning, query):
    return (await learning.prior_response_for(query) or {}).get("operations") or {
        "greeting": ("приветствовать", "demo:2")}


async def test_why_about_own_delivered_turn_gives_the_event_that_moved_the_affect(
        reactive_service):
    from debilgi.language.communication import ResponseComposer
    from debilgi.language.structured_utterance import analyze_structured_utterance

    learning, ask, greeted = await _why_setup(reactive_service)
    await _affect_row(learning._database, greeted, greeted.actor_id)
    query, plan = await _why(learning, ask, {"affect_events": [greeted.event_id]}, 1)
    assert plan and plan.operation == "explain_delivered_basis" and plan.answer_slot == "почему"
    assert [p.evidence_id for p in plan.premises] == [greeted.event_id]
    text = ResponseComposer(analyze_structured_utterance("почему?")).compose(plan.plan)[0].text
    assert "приветствовал" in text.casefold()


async def test_why_never_uses_corpus_donors_other_people_or_personal_facts_of_others(
        reactive_service):
    from dataclasses import asdict

    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    learning, ask, greeted = await _why_setup(reactive_service)
    await _affect_row(learning._database, greeted, 999)  # someone else's affect
    personal = ClauseThought(
        ConceptLexeme("любить", "любить", "verb"),
        (SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
         SemanticDependency("direct_object", ConceptArgument(ConceptLexeme("чай", "чай", "noun")))))
    donors = {"premises": [
        {"evidence_id": "seed:toloka:1:2", "kind": "human_answer_operation", "text": "x",
         "thought": asdict(personal)},
        {"evidence_id": "k:1", "kind": "knowledge", "text": "я люблю чай",
         "thought": asdict(personal)}],
        "affect_events": [greeted.event_id], "human_answer_people": 4}
    _, plan = await _why(learning, ask, donors, 1)
    assert plan is None  # the honest limit is kept by the caller
    _, none = await _why(learning, ask, {}, 2)
    assert none is None


async def test_delivered_turn_keeps_its_basis_and_old_rows_have_none(reactive_service):
    service, learning, demonstrate, ask = reactive_service
    await demonstrate("привет", "ну привет!")
    first, greeting = await ask("але")
    await service.record_sent(first, response_message_id=700, reply=greeting,
                              created_at=first.timestamp + timedelta(seconds=1))
    follow, _ = await ask("и к чему это?", n=1)
    assert "premises" in (await learning.prior_response_for(follow))["basis"]
    await learning._database.execute(
        "UPDATE language_meaning_intentions SET selection_json="
        "json_remove(selection_json,'$.basis')")
    later, _ = await ask("и к чему это?", n=2)
    assert (await learning.prior_response_for(later))["basis"] == {}
