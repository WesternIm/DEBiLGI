"""Stage 4.20: «ты говорил, что P» is checked against the agent's own delivered turns."""

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.semantic_graph import parse_proposition, reported_proposition
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 19, 6, tzinfo=UTC)


@pytest.mark.parametrize("text,negated,tense", [
    ("ты говорил, что ты не рад", True, "past"),
    ("но ты говоришь, что ты рад", False, "present"),
    ("ты сказал мне, что ты рад", False, "past"),
])
def test_reported_proposition_reads_the_complement_of_an_addressee_speech_report(
        text, negated, tense):
    reported = reported_proposition(text)
    assert reported and reported.content.negated is negated and reported.tense == tense
    assert reported.content.predicate.lemma == parse_proposition(
        "ты рад", perspective="user").predicate.lemma


@pytest.mark.parametrize("text", [
    "ты знаешь, что ты рад", "скажи, что ты рад", "ты думаешь, что ты рад",
    "я сказал, что ты рад", "ты говорил что-то", "почему ты рад?"])
def test_reported_proposition_is_none_for_other_verbs_persons_and_plain_questions(text):
    assert reported_proposition(text) is None


def test_that_conjunction_of_a_report_is_not_a_question():
    from debilgi.language.dialogue_acts import analyze_turn_form

    assert analyze_turn_form("ты говорил, что ты не рад", follows_bot=True).kind != "question"
    assert analyze_turn_form("ты что рад", follows_bot=True).kind == "question"
    assert analyze_turn_form("ты говорил, что ты рад?", follows_bot=True).kind == "question"


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "reported.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def ev(n, *, actor=100, guild=1, channel=10, at=NOW, text="x"):
    return Event(f"e:{n}", guild, channel, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": n, "content": text, "author_is_bot": False}, "discord")


async def said(db, n, text, negated, *, ago, actor=100, guild=1, channel=10):
    """A delivered turn that took the position «я [не] рад»."""
    sent = NOW - ago
    await EventRepository(db).append(ev(n, actor=actor, guild=guild, channel=channel, at=sent))
    claim = asdict(parse_proposition("я не рад" if negated else "я рад", perspective="agent"))
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,created_at,status,"
        "response_message_id,sent_at,response_text) VALUES (?,?,?,?,?,?,'c','b','op','k',?,0,0,"
        "'{}','{}',?,'sent',?,?,?)",
        (f"i:{n}", f"i:{n}", f"e:{n}", guild, channel, actor, json.dumps({"claim": claim}),
         sent.isoformat(), 5000 + n, sent.isoformat(), text))


async def check(db, text, *, actor=100, guild=1):
    learning = DialogueStructureLearning(db, None)
    context = SimpleNamespace(structure=analyze_structured_utterance(text), prompt=text)
    return await learning.verify_reported_position(
        replace(ev(99, actor=actor, guild=guild), payload={"content": text}), context)


async def test_what_was_said_is_reported_with_the_receipt_of_that_turn(db):
    await said(db, 1, "Наверное, я не рад.", True, ago=timedelta(hours=16))
    await said(db, 2, "Наверное, я рад.", False, ago=timedelta(minutes=2))
    plan = await check(db, "ты говорил, что ты не рад")
    assert plan.operation == "verify_reported_position"
    assert plan.premises[0].evidence_id == "receipt:10:5001"
    assert plan.premises[0].kind == "sent_turn"
    assert plan.plan.known_slots[0][1] == "Наверное, я не рад."


async def test_the_opposite_position_is_quoted_as_what_was_really_said(db):
    await said(db, 1, "Наверное, я рад.", False, ago=timedelta(hours=1))
    plan = await check(db, "ты говорил, что ты не рад")
    assert plan.plan.known_slots[0][1] == "Наверное, я рад."
    # Present tense: only the latest position counts.
    await said(db, 2, "Наверное, я не рад.", True, ago=timedelta(minutes=1))
    plan = await check(db, "но ты говоришь, что ты рад")
    assert plan.plan.known_slots[0][1] == "Наверное, я не рад."


async def test_nothing_said_is_a_limit_not_a_denial(db):
    plan = await check(db, "ты говорил, что ты не рад")
    assert plan.operation == "verify_reported_position" and not plan.premises
    assert plan.plan.knowledge_limit is not None and not plan.plan.known_slots


async def test_turns_to_other_people_and_other_servers_are_not_found(db):
    await said(db, 1, "Наверное, я не рад.", True, ago=timedelta(hours=1), actor=200)
    await said(db, 2, "Наверное, я не рад.", True, ago=timedelta(hours=1), guild=2)
    plan = await check(db, "ты говорил, что ты не рад")
    assert not plan.premises and plan.plan.knowledge_limit is not None
