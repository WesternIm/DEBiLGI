from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from debilgi.events.models import Event, EventType
from debilgi.language.assertion_memory import (
    build_assertion_memory,
    event_followup_thought,
    recall_event_slot,
)
from debilgi.language.lexical_gaps import nonblocking_reference_terms
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.structured_utterance import analyze_structured_utterance


def turn(text, index=0, **changes):
    return Event(event_id=f"discord:message:{index}", guild_id=1, channel_id=10,
                 actor_id=100, target_id=123, event_type=EventType.MESSAGE_REPLIED,
                 timestamp=datetime(2026, 9, 13, tzinfo=UTC) + timedelta(seconds=index),
                 payload={"content": text, "bot_mentioned": True, "bot_user_id": 123,
                          **changes}, source="test")


@pytest.mark.parametrize("text", [
    "Я вчера потерял зюзюбрик", "Когда я потерял зюзюбрик?", "Где я потерял зюзюбрик?",
])
def test_unknown_object_is_a_reference_in_literal_event_tasks(text):
    assert nonblocking_reference_terms(analyze_structured_utterance(text), ("зюзюбрик",)) == (
        "зюзюбрик",
    )


@pytest.mark.parametrize("text", [
    "Можно ли есть зюзюбрик?", "Что такое зюзюбрик?", "Почему я потерял зюзюбрик?",
    "Зюзюбрик съедобный?", "Я фырбил зюзюбрик", "Если я потерял зюзюбрик, что делать?",
])
def test_properties_definitions_and_unknown_actions_still_need_meaning(text):
    assert nonblocking_reference_terms(analyze_structured_utterance(text), ("зюзюбрик",)) == ()


@pytest.mark.parametrize("report,question,expected", [
    ("Я вчера потерял зюзюбрик", "Когда я его потерял?", "вчера"),
    ("Я потерял зюзюбрик в парке", "Где я его потерял?", "в парке"),
    ("Я позавчера нашёл фырбик", "Когда я нашёл фырбик?", "позавчера"),
    ("Я положил фепку на стол", "Куда я положил фепку?", "на стол"),
])
def test_event_slot_recall_needs_no_definition(report, question, expected):
    current = turn(question, 1)
    memory = build_assertion_memory((turn(report),), current)
    thought = recall_event_slot(question, memory, current)
    assert thought is not None
    text = realize_thought(thought, limit=1)[0].text
    assert expected in text
    assert "source_kind:user_report" in thought.evidence
    assert "Ты" in text


@pytest.mark.parametrize("report, question", [
    ("Я положил фепку на стол", "Где я положил фепку?"),
    ("Я потерял фепку в парке", "Куда я потерял фепку?"),
])
def test_event_location_and_destination_are_not_interchanged(report, question):
    current = turn(question, 1)
    memory = build_assertion_memory((turn(report),), current)
    assert recall_event_slot(question, memory, current) is None


@pytest.mark.parametrize("reports", [
    ("Я не потерял зюзюбрик вчера",),
    ("Я потерял зюзюбрик",),
    ("Ты вчера потерял зюзюбрик",),
    ("Я вчера потерял зюзюбрик", "Я сегодня потерял фырбик"),
    ("Я вчера потерял зюзюбрик", "Я не потерял зюзюбрик"),
    ("Я вчера потерял зюзюбрик?",),
])
def test_missing_or_ambiguous_event_evidence_does_not_produce_an_answer(reports):
    current = turn("Когда я его потерял?", 10)
    memory = build_assertion_memory(tuple(turn(t, i) for i, t in enumerate(reports)), current)
    assert recall_event_slot(current.payload["content"], memory, current) is None


@pytest.mark.parametrize("changes", [{"actor_id": 222}, {"channel_id": 11}, {"guild_id": 2}])
def test_event_recall_does_not_cross_speaker_or_channel(changes):
    current = turn("Когда я его потерял?", 1)
    previous = replace(turn("Я вчера потерял зюзюбрик"), **changes)
    memory = build_assertion_memory((previous,), current)
    assert recall_event_slot(current.payload["content"], memory, current) is None


def test_followup_retains_unknown_object_without_asking_its_definition():
    thought = event_followup_thought(turn("Я вчера потерял зюзюбрик"))
    text = realize_thought(thought, limit=1)[0].text
    assert "зюзюбрик" in text and "вчера" in text and text.endswith("?")
    assert "подразумеваешь" not in text


@pytest.mark.parametrize("question,reply_to", [
    ("Когда я его потеряю?", None),
    ("Когда я его потерял?", 99), ("Когда я снова его потерял?", None),
])
def test_recall_preserves_tense_reference_and_question_qualifiers(question, reply_to):
    current = turn(question, 1, reply_to_message_id=reply_to)
    memory = build_assertion_memory((turn("Я вчера потерял зюзюбрик"),), current)
    assert recall_event_slot(question, memory, current) is None


def test_known_gender_mismatch_does_not_bind_a_pronoun():
    current = turn("Когда я её потерял?", 1)
    memory = build_assertion_memory((turn("Я вчера потерял ключ"),), current)
    assert recall_event_slot(current.payload["content"], memory, current) is None
