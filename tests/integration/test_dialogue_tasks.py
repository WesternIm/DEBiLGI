from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.cognition.dialogue_tasks import (
    DialogueTaskRepository,
    contact_response_binding,
    evaluate_task_progress,
    infer_dialogue_task,
    task_candidate_metadata,
    unresolved_input_binding,
)
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.arbitration import (
    ArbitrationProfile,
    ResponseArbitrator,
    ResponseCandidate,
)
from debilgi.language.communication import ResponseComposer
from debilgi.language.generative_planner import plan_from_evidence, statement_thought
from debilgi.language.planning import ReplyContext
from debilgi.language.semantic_graph import MeaningGoal
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 9, 20, 9, tzinfo=UTC)


async def test_unknown_directive_slot_does_not_license_any_counterquestion(database):
    event = message("self-description", "привет, расскажи о себе")
    await save(database, event)
    situation = await DialogueTaskRepository(database).deliberate(
        event, prompt=event.payload["content"], consultation_messages=(), dialogue_state=None,
    )
    task = situation.active_task
    assert task.goal == "satisfy_directive"
    assert evaluate_task_progress(task, "А ты кто?").score == 0
    assert evaluate_task_progress(task, "Расскажи о себе.").score == 0


@pytest.fixture
async def database(tmp_path):
    database = Database(
        tmp_path / "dialogue-tasks.sqlite3",
        Path(__file__).resolve().parents[2] / "migrations",
    )
    await database.initialize()
    yield database
    await database.close()


def message(event_id: str, text: str, *, seconds: int = 0) -> Event:
    return Event(
        event_id,
        1,
        10,
        100,
        200,
        EventType.MESSAGE_REPLIED,
        NOW + timedelta(seconds=seconds),
        {"content": text, "author_is_bot": False, "bot_mentioned": True},
        "test",
    )


async def save(database, event):
    assert await EventRepository(database).append(event)


async def test_repeated_request_keeps_one_open_effect_and_failed_attempt(database):
    repository = DialogueTaskRepository(database)
    first = message("request-1", "скажи да")
    await save(database, first)
    situation = await repository.deliberate(
        first,
        prompt="скажи да",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert situation is not None and situation.active_task is not None
    task = situation.active_task
    assert task.goal == "perform_utterance"
    assert task.objective.required_content == "да"
    assert evaluate_task_progress(task, "Я сказал, что ты хищник.").score == 0
    assert evaluate_task_progress(task, "Да.").score == 1

    assert await repository.record_attempt(
        first,
        task_id=task.task_id,
        response_message_id=501,
        text="Я сказал, что ты хищник.",
        created_at=NOW + timedelta(seconds=1),
    )
    repeated = message("request-2", "А да не скажешь?", seconds=10)
    await save(database, repeated)
    repeated_situation = await repository.deliberate(
        repeated,
        prompt="А да не скажешь?",
        consultation_messages=(
            "user: скажи да",
            "agent: Я сказал, что ты хищник.",
        ),
        dialogue_state=None,
    )
    assert repeated_situation is not None and repeated_situation.active_task is not None
    assert repeated_situation.active_task.task_id == task.task_id
    assert repeated_situation.active_task.recurrence_count == 2
    attempt = await database.fetchone(
        "SELECT assessment FROM dialogue_task_attempts WHERE response_message_id = 501"
    )
    assert attempt["assessment"] == "failed"


async def test_locally_verifiable_effect_completes_only_after_sent_action(database):
    repository = DialogueTaskRepository(database)
    prompt = message("request", "скажи да")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="скажи да",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None and task.status == "open"
    assert await repository.record_attempt(
        prompt,
        task_id=task.task_id,
        response_message_id=777,
        text="Да.",
        created_at=NOW + timedelta(seconds=1),
    )
    completed = await repository.task(task.task_id)
    assert completed is not None and completed.status == "completed"


async def test_unrelated_corpus_surface_cannot_replace_an_active_task(database):
    repository = DialogueTaskRepository(database)
    prompt = message("request", "скажи да")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="скажи да",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None
    candidate = ResponseCandidate.create(
        event_id=prompt.event_id,
        source="learned_distribution",
        renderer="observed_word_transitions",
        text="Я сказал, что ты хищник.",
        speech_act="brief_statement",
        source_confidence=0.25,
        metadata=task_candidate_metadata(
            task,
            "Я сказал, что ты хищник.",
            {"authored_reply_template": False, "exploratory": True},
        ),
    )
    decision = ResponseArbitrator().decide_exploration(
        event_id=prompt.event_id,
        prompt="скажи да",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is None
    assert "dialogue_task_not_advanced" in decision.scores[0].rejection_reasons


async def test_next_step_uses_weighted_goal_then_builds_words_for_the_active_task(database):
    repository = DialogueTaskRepository(database)
    prompt = message("next-step", "ну и что теперь")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="ну и что теперь",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None
    action = statement_thought("Я отвечаю тебе.")
    assert action is not None
    result = plan_from_evidence(
        ReplyContext(
            prompt="ну и что теперь",
            source_event_id=prompt.event_id,
            dialogue_situation=situation,
            meaning_goals=(MeaningGoal(action, "learned:replying", 1.0),),
        )
    )
    proposal = result.proposals[0]
    surface = ResponseComposer(
        analyze_structured_utterance("ну и что теперь")
    ).compose(proposal.plan)[0]
    metadata = task_candidate_metadata(
        task,
        surface.text,
        {
            "authored_reply_template": False,
            "locally_composed": True,
            "meaning_origin": "derived",
            "semantic_origin": "derived",
            "premises": proposal.premises,
            "reasoning_evidence": tuple(
                premise.evidence_id for premise in proposal.premises
            ),
            "role_order": surface.role_order,
        },
    )

    assert surface.text == "Я отвечаю тебе теперь."
    assert metadata["dialogue_task_progress"] >= 0.5
    candidate = ResponseCandidate.create(
        event_id=prompt.event_id,
        source="semantic_reasoning",
        renderer="evidence_planner:continue_active_goal",
        text=surface.text,
        speech_act="brief_statement",
        source_confidence=proposal.confidence,
        key_points=result.objective.topics,
        metadata=metadata,
    )
    decision = ResponseArbitrator().decide(
        event_id=prompt.event_id,
        prompt="ну и что теперь",
        focus_concepts=task.objective.focus_terms,
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is not None
    assert decision.winner.text == surface.text


async def test_response_task_rejects_a_command_back_to_the_user(database):
    repository = DialogueTaskRepository(database)
    prompt = message("continuation", "и че")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="и че",
        consultation_messages=("agent: Я отвечаю тебе теперь.",),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None and task.objective.required_response_act == "respond"

    metadata = task_candidate_metadata(
        task,
        "Отвечай.",
        {
            "corpus_source_event_ids": ("old-turn",),
            "uses_persistent_word_statistics": True,
        },
    )

    assert metadata["dialogue_task_required"] is True
    assert metadata["dialogue_task_progress"] == 0
    assert metadata["dialogue_task_progress_reasons"] == (
        "command_reverses_response_obligation",
    )


async def test_dialogue_update_rejects_a_fluent_but_unrelated_database_fragment(database):
    repository = DialogueTaskRepository(database)
    prompt = message("stance", "Мне плевать на твои идеологии")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="Мне плевать на твои идеологии",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None and task.kind == "dialogue_update"

    metadata = task_candidate_metadata(
        task,
        "Первый том написан в тюрьме Ландсберг.",
        {
            "corpus_source_event_ids": ("article",),
            "uses_persistent_word_statistics": True,
            "semantic_relation_support": 1.0,
            "semantic_relation_conflicts": 0,
        },
    )

    assert metadata["dialogue_task_progress"] < 0.45
    assert metadata["context_relevance"] < 0.45


async def test_dialogue_update_accepts_a_current_proposition_stance_transform(database):
    repository = DialogueTaskRepository(database)
    prompt = message("stance-transform", "Мне плевать на твои идеологии")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="Мне плевать на твои идеологии",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None

    metadata = task_candidate_metadata(
        task,
        "На мои идеологии мне не плевать.",
        {
            "meaning_origin": "affect_weighted_stance",
            "semantic_origin": "current_turn_roles",
            "semantic_operation": "resist_attitude",
            "reasoning_evidence": (prompt.event_id,),
            "focused_request": "Мне плевать на твои идеологии",
        },
    )

    assert metadata["dialogue_task_progress"] >= 0.45
    assert "current_proposition_transformed_by_stance" in metadata[
        "dialogue_task_progress_reasons"
    ]


async def test_contact_accepts_only_an_agent_side_communicative_action(database):
    repository = DialogueTaskRepository(database)
    prompt = message("contact", "ку")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="ку",
        consultation_messages=(),
        dialogue_state=None,
    )
    task = situation.active_task
    assert task is not None and task.kind == "dialogue_update"

    unrelated = task_candidate_metadata(
        task,
        "Первый том написан в тюрьме.",
        {
            "corpus_source_event_ids": ("article",),
            "uses_persistent_word_statistics": True,
            "contact_request": True,
            "contact_response_binding": contact_response_binding(
                "Первый том написан в тюрьме."
            ),
        },
    )
    connected = task_candidate_metadata(
        task,
        "Я отвечаю тебе.",
        {
            "corpus_source_event_ids": ("dialogue",),
            "uses_persistent_word_statistics": True,
            "contact_request": True,
            "contact_response_binding": contact_response_binding("Я отвечаю тебе."),
        },
    )

    assert unrelated["dialogue_task_progress"] == 0
    assert connected["dialogue_task_progress"] >= 0.45


async def test_nonpolar_gap_does_not_accept_an_unrelated_yes_or_no(database):
    repository = DialogueTaskRepository(database)
    prompt = message("continuation", "ну?")
    await save(database, prompt)
    parent = await repository.deliberate(
        prompt,
        prompt="ну?",
        consultation_messages=("user: Дебилги",),
        dialogue_state=None,
    )
    task = parent.active_task
    assert task is not None
    situation = await repository.spawn_subtask(
        prompt,
        task,
        missing_slots=("evidence",),
    )

    metadata = task_candidate_metadata(
        situation.active_task,
        "Короче да или нет, я не считаю себя тупым.",
        {
            "corpus_source_event_ids": ("old-turn",),
            "uses_persistent_word_statistics": True,
        },
    )

    assert metadata["dialogue_task_progress"] == 0
    assert metadata["dialogue_task_progress_reasons"] == (
        "missing_evidence_not_advanced",
    )


async def test_correction_becomes_child_repair_instead_of_old_question(database):
    repository = DialogueTaskRepository(database)
    question = message("question", "Где ты видишь это слово?")
    await save(database, question)
    initial = await repository.deliberate(
        question,
        prompt="Где ты видишь это слово?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert initial is not None and initial.active_task is not None
    assert await repository.record_attempt(
        question,
        task_id=initial.active_task.task_id,
        response_message_id=888,
        text="Ты сказал лампа.",
        created_at=NOW + timedelta(seconds=1),
    )

    correction = message("correction", "Ты-то да, но не я", seconds=5)
    await save(database, correction)
    repaired = await repository.deliberate(
        correction,
        prompt="Ты-то да, но не я",
        consultation_messages=(
            "user: Где ты видишь это слово?",
            "agent: Ты сказал лампа.",
        ),
        dialogue_state=None,
    )
    task = repaired.active_task
    assert task is not None
    assert task.goal == "repair_shared_model"
    assert task.parent_task_id == initial.active_task.task_id
    assert dict(task.objective.revisions) == {
        "predicate": "сказать",
        "asserted_actor": "agent",
        "retracted_actor": "user",
        "object": "лампа",
    }

    planning = plan_from_evidence(
        ReplyContext(
            prompt="Ты-то да, но не я",
            turn_prompt="Ты-то да, но не я",
            source_event_id=correction.event_id,
            consultation_messages=(
                "user: Где ты видишь это слово?",
                "agent: Ты сказал лампа.",
            ),
            dialogue_situation=repaired,
        )
    )
    assert planning.proposals
    assert planning.proposals[0].operation == "task_repair_shared_model"
    assert planning.objective.predicate is None
    composer = ResponseComposer(analyze_structured_utterance("Ты-то да, но не я"))
    surface = composer.compose(planning.proposals[0].plan, limit=1)[0].text.casefold()
    assert "я сказал" in surface
    assert "ты не сказал" in surface


async def test_agent_can_spawn_missing_evidence_subtask(database):
    repository = DialogueTaskRepository(database)
    prompt = message("uncertain", "Хм, не знаю")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="Хм, не знаю",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert situation is not None and situation.active_task is not None
    child = await repository.spawn_subtask(
        prompt,
        situation.active_task,
        missing_slots=("evidence",),
    )
    assert child.active_task is not None
    assert child.active_task.parent_task_id == situation.active_task.task_id
    assert child.active_task.goal == "resolve_missing_evidence"
    assert child.active_task.objective.missing_slots == ("evidence",)
    assert (
        child.active_task.objective.required_response_act
        == situation.active_task.objective.required_response_act
    )
    generated_text = "Я знаю, как связать это с разговором."
    generated = task_candidate_metadata(
        child.active_task,
        generated_text,
        {
            "corpus_source_event_ids": ("observed-human-turn",),
            "uses_persistent_word_statistics": True,
        },
    )
    assert generated["dialogue_task_progress"] >= 0.45
    scripted = task_candidate_metadata(
        child.active_task,
        "Я не знаю ответ.",
        {"task_resolution": True},
    )
    assert scripted["dialogue_task_progress"] < 0.45
    assert await repository.record_attempt(
        prompt,
        task_id=child.active_task.task_id,
        response_message_id=999,
        text=generated_text,
        created_at=NOW + timedelta(seconds=1),
        predicted_progress=float(generated["dialogue_task_progress"]),
    )
    attempt = await database.fetchone(
        "SELECT predicted_progress FROM dialogue_task_attempts WHERE response_message_id = 999"
    )
    assert attempt is not None and attempt["predicted_progress"] == pytest.approx(
        generated["dialogue_task_progress"]
    )


async def test_missing_evidence_task_admits_supported_learned_language(database):
    repository = DialogueTaskRepository(database)
    prompt = message("preference-question", "Любишь батончики?")
    await save(database, prompt)
    parent = await repository.deliberate(
        prompt,
        prompt="Любишь батончики?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert parent is not None and parent.active_task is not None
    situation = await repository.spawn_subtask(
        prompt,
        parent.active_task,
        missing_slots=("unresolved_polar",),
    )
    task = situation.active_task
    assert task is not None
    text = "Батончики с шоколадом мне нравятся."
    candidate = ResponseCandidate.create(
        event_id=prompt.event_id,
        source="learned_distribution",
        renderer="observed_word_transitions",
        text=text,
        speech_act="brief_statement",
        source_confidence=0.25,
        metadata=task_candidate_metadata(
            task,
            text,
            {
                "authored_reply_template": False,
                "exploratory": True,
                "corpus_source_event_ids": ("human-corpus:1", "human-corpus:2"),
                "uses_persistent_word_statistics": True,
            },
        ),
    )
    decision = ResponseArbitrator().decide_exploration(
        event_id=prompt.event_id,
        prompt="Любишь батончики?",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is not None
    assert decision.winner.source == "learned_distribution"
    assert decision.winner.text == text


async def test_answer_task_rejects_a_learned_question_echo(database):
    repository = DialogueTaskRepository(database)
    prompt = message("quantity-question", "У тебя сколько бабушек?")
    await save(database, prompt)
    parent = await repository.deliberate(
        prompt,
        prompt="У тебя сколько бабушек?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert parent is not None and parent.active_task is not None
    assert parent.active_task.objective.question_word == "сколько"
    assert parent.active_task.objective.required_response_act == "answer"
    situation = await repository.spawn_subtask(
        prompt,
        parent.active_task,
        missing_slots=("unresolved_slot",),
    )
    task = situation.active_task
    assert task is not None and task.objective.required_response_act == "answer"

    echoed_question = "А что у меня сколько бабушек?"
    metadata = task_candidate_metadata(
        task,
        echoed_question,
        {
            "corpus_source_event_ids": ("human-corpus:1",),
            "uses_persistent_word_statistics": True,
        },
    )
    assert metadata["dialogue_task_progress"] == 0
    assert metadata["dialogue_task_progress_reasons"] == (
        "question_does_not_fulfill_response_obligation",
    )

    candidate = ResponseCandidate.create(
        event_id=prompt.event_id,
        source="learned_distribution",
        renderer="observed_word_transitions",
        text=echoed_question,
        speech_act="clarifying_question",
        source_confidence=0.25,
        metadata=metadata,
    )
    decision = ResponseArbitrator().decide_exploration(
        event_id=prompt.event_id,
        prompt="У тебя сколько бабушек?",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is None
    assert "dialogue_task_not_advanced" in decision.scores[0].rejection_reasons


async def test_contact_owns_current_turn_without_erasing_unfinished_answer(database):
    repository = DialogueTaskRepository(database)
    question = message("capability-question", "Ты машиной управлять можешь?")
    await save(database, question)
    parent = await repository.deliberate(
        question,
        prompt="Ты машиной управлять можешь?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert parent is not None and parent.active_task is not None
    child = await repository.spawn_subtask(
        question,
        parent.active_task,
        missing_slots=("unresolved_polar",),
    )
    assert child.active_task is not None

    contact = message("contact", "ауу", seconds=5)
    await save(database, contact)
    continued = await repository.deliberate(
        contact,
        prompt="ауу",
        consultation_messages=("user: Ты машиной управлять можешь?",),
        dialogue_state=None,
    )

    assert continued is not None and continued.active_task is not None
    assert continued.active_task.source_event_id == contact.event_id
    assert continued.active_task.objective.required_response_act == "respond"
    assert continued.active_task.goal == "respond_relevantly"
    contact_tasks = [
        task for task in continued.open_tasks if task.source_event_id == contact.event_id
    ]
    assert contact_tasks == [continued.active_task]
    assert any(task.task_id == child.active_task.task_id for task in continued.open_tasks)


@pytest.mark.parametrize("text", ["Ответь нормально", "Почему"])
def test_response_controls_and_bare_followups_do_not_become_literal_answers(text):
    task = infer_dialogue_task(
        text,
        source_event_id="current",
        latest_agent_turn="Предыдущий ответ.",
    )

    assert task.goal == "respond_relevantly"
    assert task.objective.required_content is None


@pytest.mark.parametrize("text", ["Всмысле", "Это разве ответ?"])
def test_answer_challenges_require_repair_instead_of_an_epistemic_script(text):
    task = infer_dialogue_task(
        text,
        source_event_id="current",
        latest_agent_turn="Предыдущий ответ.",
    )

    assert task.kind == "response_quality_repair"
    assert task.goal == "repair_shared_model"
    assert task.objective.required_response_act == "repair"


def test_bare_repair_never_inherits_the_agents_failed_topic():
    task = infer_dialogue_task(
        "Чего?",
        source_event_id="current",
        latest_user_turn="ку",
        latest_agent_turn="Автор резко отвергает марксизм.",
    )

    assert task.kind == "response_quality_repair"
    assert task.goal == "repair_shared_model"
    assert not {"автор", "марксизм"}.intersection(task.objective.focus_terms)


async def test_new_greeting_owns_current_turn_instead_of_resuming_old_question(database):
    repository = DialogueTaskRepository(database)
    question = message("old-question", "Где находится кварц?")
    await save(database, question)
    old = await repository.deliberate(
        question,
        prompt="Где находится кварц?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert old is not None and old.active_task is not None
    assert old.active_task.goal == "answer_question"

    greeting = message("new-greeting", "привет", seconds=5)
    await save(database, greeting)
    current = await repository.deliberate(
        greeting,
        prompt="привет",
        consultation_messages=("user: Где находится кварц?",),
        dialogue_state=None,
    )

    assert current is not None and current.active_task is not None
    assert current.active_task.source_event_id == greeting.event_id
    assert current.active_task.goal == "respond_relevantly"


def test_explicit_self_identity_question_does_not_inherit_the_agents_old_predicate():
    task = infer_dialogue_task(
        "так ты кто",
        source_event_id="identity-question",
        dialogue_state=SimpleNamespace(
            topic=SimpleNamespace(active_concepts=("отвечать",)),
            unresolved_references=(),
        ),
        latest_agent_turn="Я отвечаю.",
    )

    assert task.goal == "answer_question"
    assert task.objective.question_word == "кто"
    assert "отвечать" not in task.objective.focus_terms


def test_generic_speaking_action_cannot_replace_content_but_cognitive_limit_can():
    task = infer_dialogue_task("қотақ жепсің", source_event_id="unknown-input")
    speaking = evaluate_task_progress(
        task,
        "Я отвечаю.",
        {
            "uses_persistent_word_statistics": True,
            "unresolved_input_reaction": True,
            "unresolved_prompt_terms": ("қотақ", "жепсің"),
            "unresolved_input_binding": unresolved_input_binding("Я отвечаю."),
        },
    )
    cognitive = evaluate_task_progress(
        task,
        "Я не знаю.",
        {
            "uses_persistent_word_statistics": True,
            "unresolved_input_reaction": True,
            "unresolved_prompt_terms": ("қотақ", "жепсің"),
            "unresolved_input_binding": unresolved_input_binding("Я не знаю."),
        },
    )

    assert speaking.score == 0.0
    assert cognitive.score >= 0.58
    assert cognitive.reasons == ("learned_cognitive_state_marks_unresolved_input",)


async def test_stale_attention_call_does_not_resume_expired_question(database):
    repository = DialogueTaskRepository(database)
    question = message("stale-question", "Где находится кварц?")
    await save(database, question)
    old = await repository.deliberate(
        question,
        prompt="Где находится кварц?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert old is not None and old.active_task is not None

    attention = message("late-attention", "ауу", seconds=16 * 60)
    await save(database, attention)
    current = await repository.deliberate(
        attention,
        prompt="ауу",
        consultation_messages=(),
        dialogue_state=None,
    )

    assert current is not None and current.active_task is not None
    assert current.active_task.source_event_id == attention.event_id
    assert current.active_task.goal == "respond_relevantly"


def test_last_question_clause_defines_the_response_obligation():
    task = infer_dialogue_task(
        "Не знаю, а ты можешь?",
        source_event_id="mixed-turn",
    )

    assert task.goal == "answer_question"
    assert task.objective.predicate == "мочь"
    assert task.objective.required_response_act == "answer"


async def test_missing_answer_becomes_agent_generated_reasoning_subtask(database):
    repository = DialogueTaskRepository(database)
    prompt = message("question-without-evidence", "Где находится кварц?")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="Где находится кварц?",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert situation is not None and situation.active_task is not None
    assert situation.active_task.goal == "answer_question"
    context = ReplyContext(
        prompt="Где находится кварц?",
        turn_prompt="Где находится кварц?",
        source_event_id=prompt.event_id,
        dialogue_situation=situation,
    )
    unresolved = plan_from_evidence(context)
    assert not unresolved.proposals

    child = await repository.spawn_subtask(
        prompt,
        situation.active_task,
        missing_slots=unresolved.unresolved,
    )
    resolved = plan_from_evidence(replace(context, dialogue_situation=child))
    assert not resolved.proposals
    assert child.active_task is not None
    assert child.active_task.parent_task_id == situation.active_task.task_id


async def test_incomplete_directive_is_control_state_not_authored_reply(database):
    repository = DialogueTaskRepository(database)
    prompt = message("directive-without-object", "ну так узнай")
    await save(database, prompt)
    situation = await repository.deliberate(
        prompt,
        prompt="ну так узнай",
        consultation_messages=(),
        dialogue_state=None,
    )
    assert situation is not None and situation.active_task is not None
    assert situation.active_task.goal == "satisfy_directive"
    result = plan_from_evidence(
        ReplyContext(
            prompt="ну так узнай",
            turn_prompt="ну так узнай",
            source_event_id=prompt.event_id,
            dialogue_situation=situation,
        )
    )
    assert not result.proposals


def test_task_inference_is_structural_not_an_input_reply_table():
    command = infer_dialogue_task("произнеси кварц", source_event_id="one")
    another = infer_dialogue_task("повтори северный ветер", source_event_id="two")
    assert command.goal == another.goal == "perform_utterance"
    assert command.objective.required_content == "кварц"
    assert another.objective.required_content == "северный ветер"
    assert command.signature != another.signature

    repair = infer_dialogue_task(
        "Ты-то да, но не я",
        source_event_id="three",
        latest_agent_turn="Ты сказал лампа.",
    )
    assert repair.goal == "repair_shared_model"
    assert dict(repair.objective.revisions)["object"] == "лампа"
