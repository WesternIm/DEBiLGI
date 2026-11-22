"""Stage 4.22: stale donor hypotheses, the conditional limit, no unmarked self-assertion."""

import zlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.events.models import Event, EventType
from debilgi.language.communication import ReaderGoal, reader_goal
from debilgi.language.dialogue_learning import reverse_reaction_check
from debilgi.language.generative_planner import knowledge_limit_plan
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 19, 10, 9, tzinfo=UTC)
GUILD, CHANNEL, USER, BOT = 1, 10, 100, 777


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "limit.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def ev(text, n=1):
    return Event(f"e:{n}", GUILD, CHANNEL, USER, None, EventType.MESSAGE_CREATED, NOW,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text,
                  "author_is_bot": False, "bot_user_id": BOT, "bot_mentioned": True}, "discord")


def donor_hypothesis(text, goal, **extra):
    return {"text": text, "renderer": "x", "metadata": {
        "derivation": "human_answer_crossover", "sleep_opinion": True, "sleep_answer_slot": True,
        "sleep_query_goal": goal, **extra}}


def test_old_donor_answer_to_an_open_slot_about_the_agent_is_not_a_candidate():
    goal = {"addressed": True, "question_word": "что", "predicate": "делать"}
    event = ev("что делаешь?")
    for text in ("Наверное, я смотрю сериал.", "Наверное, я собираю букет."):
        assert ContinuousCognitionService.sleep_rejection(
            event, donor_hypothesis(text, goal)) == "donor_answer_about_self"
    # The same operation for another request (not an open slot about the agent) is not
    # rejected for that reason.
    polar = {"addressed": True, "question_word": None, "predicate": "делать"}
    assert ContinuousCognitionService.sleep_rejection(
        event, donor_hypothesis("Наверное, я смотрю сериал.", polar)) != "donor_answer_about_self"
    about_other = {"addressed": False, "question_word": "что", "predicate": "делать"}
    assert ContinuousCognitionService.sleep_rejection(
        event, donor_hypothesis("Наверное, ты смотришь сериал.", about_other)
    ) != "donor_answer_about_self"


async def _limit(db, text):
    event = ev(text)
    result = await PropositionMemory.for_meaning(db).retrieve_claims(event, text)
    context = SimpleNamespace(structure=analyze_structured_utterance(text), claim_result=result,
                              response_options=())
    return knowledge_limit_plan(context)


async def test_conditional_question_without_a_ground_limits_the_consequent(db):
    derived = await _limit(db, "а если хотел, то полетел бы?")
    assert derived is not None and derived.operation == "express_query_unknown"
    question = derived.plan.knowledge_limit.question
    assert question is not None and question.predicate.lemma == "полететь"
    assert dict(question.predicate.features)["mood"] == "conditional"
    subject = next(d.argument.head.lemma for d in question.dependencies if d.role == "subject")
    assert subject == "я"  # the question is about the agent, not an echo of the asker
    text = realize_thought(question, limit=1)[0].text
    assert text == "Я полетел бы?"


async def test_condition_with_a_reading_is_kept_in_the_plan(db):
    derived = await _limit(db, "если бы ты открыл дверь, ты бы полетел?")
    assert derived is not None
    plan = derived.plan
    assert plan.condition is not None and plan.condition.predicate.lemma == "открыть"
    assert plan.knowledge_limit.question.predicate.lemma == "полететь"


async def test_plain_question_keeps_the_old_limit_without_mood(db):
    derived = await _limit(db, "ты полетел?")
    assert derived is not None and derived.plan.condition is None
    assert dict(derived.plan.knowledge_limit.question.predicate.features)["mood"] != "conditional"


def _core(slots, kind="question"):
    return SimpleNamespace(component="core", hypothesis=(), slots=slots, clause_kind=kind,
                           meaning="slot_question")


def _plan(prompt):
    goal = reader_goal(analyze_structured_utterance(prompt))
    assert isinstance(goal, ReaderGoal)
    return SimpleNamespace(goal=goal, hypothesis=(), claim=SimpleNamespace(
        predicate=SimpleNamespace(lemma="slot_question")))


def _slots(*parts):
    return [{"part": p, "binding": b, "surface": s, "lemma": s} for p, b, s in parts]


def test_reaction_that_restates_the_asked_verb_as_a_statement_about_itself_is_rejected():
    prompt = "ты бы хотел полететь?"
    core = _core(_slots(("particle", "grammar", "да"), (None, "punctuation", ","),
                        ("verb", "prompt", "хотеть"), (None, "punctuation", ")")))
    check = reverse_reaction_check(_plan(prompt), (core,), prompt, "да, хотел) а ты?)")
    assert "unsupported_self_assertion" in check["issues"]


def test_reaction_with_a_question_word_is_not_a_self_assertion():
    prompt = "ты бы хотел полететь?"
    core = _core(_slots(("pronoun", "grammar", "кто"), ("verb", "prompt", "хотеть")))
    check = reverse_reaction_check(_plan(prompt), (core,), prompt, "кто их не хотел?)")
    assert "unsupported_self_assertion" not in check["issues"]


def test_instrumental_question_word_reaction_is_not_a_self_assertion():
    prompt = "ответишь?"
    core = _core(_slots(("pronoun", "grammar", "чем"), ("verb", "prompt", "ответить")))
    check = reverse_reaction_check(_plan(prompt), (core,), prompt, "вот чем ответишь?!")
    assert "unsupported_self_assertion" not in check["issues"]
