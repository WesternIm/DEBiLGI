from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from debilgi.events.models import Event, EventType
from debilgi.language.dialogue_obligations import exchange_from_events, resolve_dialogue_exchange
from debilgi.language.dialogue_state import analyze_dialogue_state
from debilgi.language.productive_nlg import realize_thought


@pytest.mark.parametrize(
    "question,answer,role,terms",
    [
        (
            "Что ты подразумеваешь под «салфетка»?",
            "Это бумага такая для гигиены",
            "predicate_complement",
            {"салфетка", "бумага", "гигиена"},
        ),
        ("Что такое втулка?", "Металлическая деталь", "predicate_complement", {"втулка", "деталь"}),
        (
            "Что означает термос?",
            "Сосуд для напитков",
            "predicate_complement",
            {"термос", "сосуд", "напиток"},
        ),
        ("Кто открыл дверь?", "Маша", "subject", {"маша", "дверь"}),
        ("Кто нашёл ключ?", "Сосед", "subject", {"сосед", "ключ"}),
        ("Что ты открыл?", "Дверь", "direct_object", {"ты", "дверь"}),
        ("Что ты нашёл?", "Красную папку", "direct_object", {"ты", "папка"}),
        ("Где лежит ключ?", "На столе", "location", {"ключ", "стол"}),
        ("Где лежит книга?", "В комнате", "location", {"книга", "комната"}),
    ],
)
def test_answers_fill_semantic_slots_across_words_and_question_forms(question, answer, role, terms):
    state = analyze_dialogue_state(answer, (question,))
    exchange = state.exchange
    assert exchange is not None and exchange.status == "matched"
    assert exchange.obligation.expected_role == role
    thought = exchange.bound_proposition
    assert {d.argument.head.lemma for d in thought.dependencies} == terms
    surfaces = realize_thought(thought, limit=1)
    assert surfaces and surfaces[0].text.endswith("?")
    assert exchange.obligation.source_message == question
    assert exchange.answer_message == answer
    # Rebuilding from event text after restart yields the same state.
    assert resolve_dialogue_exchange(answer, (question,)) == exchange


@pytest.mark.parametrize(
    "question,answer",
    [
        ("Кто открыл дверь?", "На столе"),
        ("Где лежит ключ?", "Маша"),
        ("Что такое салфетка?", "Это не бумага"),
        ("Что такое салфетка?", "Бумага или ткань"),
        ("Что такое салфетка?", "Кстати, начался дождь"),
        ("Кто открыл дверь?", "Кто?"),
        ("Что такое салфетка?", "Бумага. Давай поговорим о кино"),
    ],
)
def test_incompatible_answers_keep_gap_unresolved(question, answer):
    result = resolve_dialogue_exchange(answer, (question,))
    assert result is not None and result.status == "unmatched"
    assert result.bound_proposition is None


def test_no_binding_across_an_intervening_turn_or_without_question():
    assert resolve_dialogue_exchange("Маша", ()) is None
    assert resolve_dialogue_exchange("Маша", ("Кто открыл дверь?", "Поменяем тему")) is None


def event_pair():
    now = datetime(2026, 9, 12, tzinfo=UTC)
    question = Event(
        event_id="q",
        guild_id=1,
        channel_id=10,
        actor_id=99,
        target_id=7,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=now,
        source="test",
        payload={"content": "Кто открыл дверь?", "author_is_bot": True, "message_id": 1},
    )
    answer = Event(
        event_id="a",
        guild_id=1,
        channel_id=10,
        actor_id=7,
        target_id=99,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=now + timedelta(seconds=1),
        source="test",
        payload={"content": "Маша", "reply_to_message_id": 1, "bot_user_id": 99},
    )
    return question, answer


def test_event_evidence_survives_reconstruction():
    q, a = event_pair()
    result = exchange_from_events((q,), a)
    assert result.status == "matched"
    assert (result.question_event_id, result.answer_event_id) == ("q", "a")


def test_name_answer_fills_the_question_slot_even_when_morphology_guesses_a_verb():
    result = resolve_dialogue_exchange("Дебилги", ("Как меня зовут?",))

    assert result is not None and result.status == "matched"
    assert result.obligation.expected_role == "name"
    thought = result.bound_proposition
    assert thought is not None and thought.clause_kind == "statement"
    assert thought.predicate.lemma == "называться"
    roles = {dependency.role: dependency.argument for dependency in thought.dependencies}
    assert roles["subject"].head.lemma == "я"
    assert roles["predicate_complement"].head.surface == "Дебилги"
    surfaces = realize_thought(thought, limit=1)
    assert surfaces and surfaces[0].text == "Я называюсь Дебилги."


@pytest.mark.parametrize("change", ["channel", "guild", "author", "old", "reply", "bot", "topic"])
def test_actual_question_is_scoped_to_recent_turn_and_participants(change):
    q, a = event_pair()
    if change == "channel":
        q = replace(q, channel_id=20)
    elif change == "guild":
        q = replace(q, guild_id=20)
    elif change == "author":
        a = replace(a, actor_id=8)
    elif change == "old":
        a = replace(a, timestamp=a.timestamp + timedelta(minutes=6))
    elif change == "reply":
        a = replace(a, payload={**a.payload, "reply_to_message_id": 9})
    elif change == "bot":
        q = replace(q, actor_id=123)
    elif change == "topic":
        q = replace(q, payload={**q.payload, "content": "Поменяем тему"})
    assert exchange_from_events((q,), a) is None


@pytest.mark.parametrize(
    "question,answer,predicate",
    [
        ("Почему ты сейчас меня выключишь?", "Для эксперимента", "выключить"),
        ("Зачем ты открыл дверь?", "Для проверки", "открыть"),
    ],
)
def test_purpose_answers_bind_action_participants_and_existing_modifiers(
    question, answer, predicate
):
    result = resolve_dialogue_exchange(answer, (question,))
    assert result.status == "matched"
    thought = result.bound_proposition
    assert thought.predicate.lemma == predicate
    roles = {dep.role: dep.argument for dep in thought.dependencies}
    assert roles["subject"].head.lemma == "ты"
    assert roles["cause"].preposition == "для"
    assert roles["cause"].grammatical_case == "genitive"
    if "сейчас" in question:
        assert roles["direct_object"].head.lemma == "я"
        assert roles["manner"].head.lemma == "сейчас"


@pytest.mark.parametrize("answer,negated", [("Да", False), ("Нет", True)])
def test_polar_answers_bind_polarity_without_inventing_new_action(answer, negated):
    result = resolve_dialogue_exchange(answer, ("Ты открыл дверь?",))
    assert result.status == "matched"
    assert result.bound_proposition.negated is negated
    assert result.bound_proposition.predicate.lemma == "открыть"


@pytest.mark.parametrize(
    "question,answer",
    [
        ("Ты открыл дверь?", "Не знаю"),
        ("Ты открыл дверь?", "Да, возможно"),
        ("Ты открыл дверь?", "Да или нет"),
        ("Почему ты открыл дверь?", "Для проверка"),
        ("Почему ты открыл дверь?", "На столе"),
    ],
)
def test_binding_does_not_erase_uncertainty_or_case_mismatch(question, answer):
    result = resolve_dialogue_exchange(answer, (question,))
    assert result is not None and result.status == "unmatched"


def test_negative_question_is_not_bound_as_ambiguous_yes_no():
    assert resolve_dialogue_exchange("Да", ("Ты не открыл дверь?",)) is None
