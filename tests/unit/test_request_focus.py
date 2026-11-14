from dataclasses import replace

import pytest

from debilgi.language.generative_planner import resolve_evidence_context, statement_thought
from debilgi.language.planning import ReplyContext
from debilgi.language.request_focus import focus_request, supported_alternative


@pytest.mark.parametrize(
    "text,anchor",
    [
        ("Ладно, опустим это. Так мне выключать тебя?", "выключать"),
        ("Я потерял ключ. Где лежит книга?", "книга"),
        ("Где лежит ключ? А кто нашёл книгу?", "нашёл"),
    ],
)
def test_last_independent_question_is_the_request(text, anchor):
    result = focus_request(text, ())
    assert anchor in result.request_text
    assert result.request_text.endswith("?")
    assert result.turn_text == text
    assert result.request_text.count("?") == 1


@pytest.mark.parametrize(
    "control",
    [
        "Короче да или нет",
        "Да? Нет?",
        "Да или нет, выбери одно",
        "Нет, просто дай бинарный ответ, да или нет",
        "Дай ответ",
        "Но ты ещё не дал ответа про выключение тебя. Дай ответ",
        "А ты тему не переводи, ушлый",
    ],
)
def test_response_controls_keep_question_across_many_turns(control):
    question = "Мне тебя выключить?"
    turns = ("user: " + question,) + (
        "agent: Я дублировал ответ?",
        "user: Дай ответ",
    ) * 15
    result = focus_request(control, turns)
    assert result.request_text == question
    assert result.reason == "answer_control_keeps_request"
    assert result.turn_text == control


def test_manner_word_changes_response_policy_and_does_not_become_the_reply():
    result = focus_request(
        "Ответь нормально",
        ("user: Ты читаешь?", "agent: Нормально."),
    )

    assert result.request_text == "Ты читаешь?"
    assert result.reason == "answer_control_keeps_request"


def test_new_substantive_topic_stops_pending_request_search():
    result = focus_request(
        "Да или нет, выбери одно",
        (
            "user: Ты открыл дверь?",
            "user: Маша читает книгу.",
        ),
    )
    assert result.reason == "current_turn"
    assert result.request_text == result.turn_text


def test_command_with_real_object_is_not_a_discourse_control():
    context = ReplyContext(prompt="Дай ключ", consultation_messages=("user: Ты открыл дверь?",))
    assert resolve_evidence_context(context).prompt == "Дай ключ"


def test_competing_current_question_replaces_pending_request():
    result = focus_request("А где книга?", ("user: Ты открыл дверь?",))
    assert result.request_text == "А где книга?"


def test_infinitive_question_does_not_resume_an_older_conditional():
    original = ReplyContext(
        prompt="Мне тебя выключить?", consultation_messages=("user: А если я сейчас тебя выключу?",)
    )
    assert resolve_evidence_context(original).prompt == original.prompt


def test_focus_does_not_discard_condition_qualifying_a_question():
    text = "Если Маша найдёт ключ, кто откроет дверь?"
    assert focus_request(text, ()).request_text == text


def test_new_qualified_statement_does_not_get_replaced_with_old_question():
    context = ReplyContext(
        prompt="Маша якобы нашла ключ.", consultation_messages=("user: Кто открыл дверь?",)
    )
    assert resolve_evidence_context(context).prompt == context.prompt


@pytest.mark.parametrize(
    "fact,expected", [("Маша открыла дверь.", "да"), ("Маша не открыла дверь.", "нет")]
)
def test_binary_projection_requires_supported_claim_with_matching_time(fact, expected):
    thought = statement_thought(fact)
    assert supported_alternative("Маша открыла дверь?", ("да", "нет"), thought) == expected
    assert supported_alternative("Маша откроет дверь?", ("да", "нет"), thought) is None
    assert supported_alternative("Почему Маша открыла дверь?", ("да", "нет"), thought) is None
    assert supported_alternative("Маша не открыла дверь?", ("да", "нет"), thought) is None
    assert (
        supported_alternative(
            "Маша открыла дверь?", ("да", "нет"), replace(thought, clause_kind="question")
        )
        is None
    )


@pytest.mark.parametrize(
    "question_text", ["Петя открыл дверь?", "Маша открыла книгу?", "Маша сегодня открыла дверь?"]
)
def test_binary_projection_does_not_drop_participants_objects_or_time(question_text):
    thought = statement_thought("Маша открыла дверь.")
    assert supported_alternative(question_text, ("да", "нет"), thought) is None


def test_modal_projection_requires_the_whole_action_state_and_recipient():
    from debilgi.language.semantic_graph import parse_proposition

    thought = parse_proposition("Ты не хочешь быть выключенным.")
    assert supported_alternative("Ты хочешь быть выключенным?", ("да", "нет"), thought) == "нет"
    assert supported_alternative("Ты хочешь быть включенным?", ("да", "нет"), thought) is None
    assert supported_alternative("Ты не хочешь быть выключенным?", ("да", "нет"), thought) is None
    assert (
        supported_alternative("Почему ты хочешь быть выключенным?", ("да", "нет"), thought) is None
    )
    ability = parse_proposition("Ты не можешь отвечать мне.")
    assert supported_alternative("Ты можешь отвечать мне?", ("да", "нет"), ability) == "нет"
    assert supported_alternative("Ты можешь отвечать Маше?", ("да", "нет"), ability) is None
