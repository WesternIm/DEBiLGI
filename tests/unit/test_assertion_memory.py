from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from debilgi.events.models import Event, EventType
from debilgi.language.assertion_memory import build_assertion_memory, plan_assertion_continuation
from debilgi.language.productive_nlg import realize_thought

NOW = datetime(2026, 9, 9, tzinfo=UTC)


def turn(text, index=0, *, bot=False):
    return Event(
        event_id=f"event:{index}",
        guild_id=1,
        channel_id=10,
        actor_id=123 if bot else 100,
        target_id=100 if bot else 123,
        event_type=EventType.MESSAGE_REPLIED,
        timestamp=NOW + timedelta(seconds=index),
        payload={"content": text, "author_is_bot": bot, "bot_user_id": 123},
        source="test",
    )


@pytest.mark.parametrize(
    "prior,lemma",
    [
        ("Я пока не открыл дверь.", "открыть"),
        ("Дождь пока не начался.", "начаться"),
        ("Я не считаю себя тупым пока.", "считать"),
    ],
)
def test_temporal_followup_generates_question_from_attributed_claim(prior, lemma):
    history = (turn(prior, bot=True),)
    current = turn("Ну всё, время настало", 2)
    memory = build_assertion_memory(history, current)
    plan = plan_assertion_continuation(current.payload["content"], memory, history, current)
    assert plan is not None
    assert plan.thought.predicate.lemma == lemma
    assert plan.thought.clause_kind == "question"
    assert not plan.thought.negated
    assert memory.assertions[0].thought.negated  # hypothesis cannot mutate memory
    assert plan.source_event_ids == ("event:0", "event:2")
    surfaces = realize_thought(plan.thought, limit=1)
    assert surfaces and surfaces[0].text.endswith("?")
    assert "время" not in surfaces[0].text.lower()


def test_speaker_binding_and_changes_preserve_history():
    history = (
        turn("Я открыл дверь.", 0),
        turn("Я не открыл дверь.", 1),
        turn("Ты открыл дверь.", 2, bot=True),
    )
    memory = build_assertion_memory(history, turn("А теперь?", 3))
    assert len(memory.assertions) == 3
    assert [link.kind for link in memory.links] == ["revised_position", "disagreement"]
    assert len(memory.active) == 2  # one latest claim per speaker, not universal truth
    assert all(
        claim.thought.dependencies[0].argument.head.entity_kind == "user:100"
        for claim in memory.assertions
    )


def test_replay_deduplicates_evidence_without_promoting_repetition_to_truth():
    first, second = turn("Петя открыл дверь.", 0), turn("Петя открыл дверь.", 1)
    current = turn("А теперь?", 2)
    memory = build_assertion_memory((second, first, first), current)
    assert len(memory.assertions) == 2
    assert memory.links[0].kind == "repeated_claim"
    assert len(memory.active) == 1
    assert memory == build_assertion_memory((first, second), current)


@pytest.mark.parametrize(
    "text",
    [
        "Ты открыл дверь?",
        "Если Петя открыл дверь, Коля вошёл.",
        "Петя любит красные яблоки.",
        "Он сказал «Я открыл дверь».",
        "Наверное, Петя открыл дверь.",
        "Он открыл дверь.",
    ],
)
def test_questions_uncertain_and_unsupported_claims_do_not_become_facts(text):
    assert not build_assertion_memory((turn(text),), turn("А теперь?", 2)).assertions


def test_different_actors_roles_and_tenses_are_not_contradictions():
    history = tuple(
        turn(text, i)
        for i, text in enumerate(
            (
                "Петя толкнул Колю.",
                "Коля не толкнул Петю.",
                "Петя открывает дверь.",
                "Петя не открыл дверь.",
            )
        )
    )
    assert not build_assertion_memory(history, turn("А теперь?", 5)).links


@pytest.mark.parametrize(
    "change",
    [
        {"guild_id": 2},
        {"channel_id": 11},
        {"actor_id": 999, "target_id": 123},
        {"timestamp": NOW - timedelta(minutes=6)},
        {"timestamp": NOW + timedelta(minutes=1)},
    ],
)
def test_scope_excludes_other_conversations_stale_and_future_events(change):
    old = replace(turn("Я пока не открыл дверь."), **change)
    assert not build_assertion_memory((old,), turn("А теперь?", 2)).assertions


@pytest.mark.parametrize("extra", ["Дождь пока не начался.", "Петя читает книгу."])
def test_ambiguous_or_changed_topic_does_not_force_temporal_link(extra):
    history = (turn("Я пока не открыл дверь.", 0, bot=True), turn(extra, 1))
    current = turn("А теперь?", 2)
    memory = build_assertion_memory(history, current)
    assert plan_assertion_continuation(current.payload["content"], memory, history, current) is None


def test_user_first_person_is_realized_as_second_person():
    history = (turn("Я пока не открываю дверь."),)
    current = turn("А теперь?", 2)
    plan = plan_assertion_continuation(
        "А теперь?", build_assertion_memory(history, current), history, current
    )
    assert plan is not None
    assert realize_thought(plan.thought, limit=1)[0].text == "Ты открываешь дверь?"


def test_other_bot_does_not_supply_agent_claims():
    old = replace(turn("Я пока не открыл дверь.", bot=True), actor_id=999)
    assert not build_assertion_memory((old,), turn("А теперь?", 2)).assertions


def test_number_of_objects_is_part_of_the_claim():
    history = (turn("Петя открыл дверь.", 0), turn("Петя не открыл двери.", 1))
    memory = build_assertion_memory(history, turn("А теперь?", 2))
    assert len(memory.assertions) == 2
    assert not memory.links


def test_discord_mention_does_not_hide_an_addressed_claim():
    old = turn("<@123> ты пока не открыл дверь.")
    memory = build_assertion_memory((old,), turn("А теперь?", 2))
    assert len(memory.assertions) == 1
    assert memory.assertions[0].thought.dependencies[0].argument.head.entity_kind == "agent"


def test_interleaved_disagreement_does_not_hide_a_speakers_revision():
    history = (
        turn("Я открыл дверь.", 0),
        turn("Ты не открыл дверь.", 1, bot=True),
        turn("Я не открыл дверь.", 2),
    )
    memory = build_assertion_memory(history, turn("А теперь?", 3))
    assert {(link.kind, link.earlier_event_id, link.later_event_id) for link in memory.links} == {
        ("disagreement", "event:0", "event:1"),
        ("agreement", "event:1", "event:2"),
        ("revised_position", "event:0", "event:2"),
    }
