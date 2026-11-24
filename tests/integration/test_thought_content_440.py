"""Stage 4.40 (Э3): an inner failure becomes a question about what was not understood.

Flag ``DEBILGI_THOUGHT_CONTENT`` only. The words are those people of the server used; the budget
counts repairs, not the agent's other questions; the rate never exceeds people's own.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import BoredomLedger, BoredomRow
from debilgi.language.communication import ResponseComposer
from debilgi.language.dialogue_learning import DialogueStructureLearning, exchange_analysis
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, USER, OTHER = 1, 10, 100, 101


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "content440.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def event(n="q", actor=USER, text="x"):
    return Event(f"e:{n}", GUILD, CHANNEL, actor, None, EventType.MESSAGE_CREATED, NOW,
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
                  "bot_user_id": 7, "bot_mentioned": True}, "discord")


async def said(db, n, text, *, actor=OTHER, ago=timedelta(days=1)):
    at = NOW - ago
    await EventRepository(db).append(Event(
        f"e:{n}", GUILD, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
        {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
         "bot_user_id": 7, "bot_mentioned": True}, "discord"))
    await db.execute(
        "INSERT INTO language_discourse_turns (guild_id,channel_id,message_id,actor_id,source_id,"
        "utterance_type,question_status,content,topics_json,observed_at,created_at) "
        "VALUES (?,?,?,?,?,'assertion','not_question',?,'[]',?,?)",
        (GUILD, CHANNEL, zlib.crc32(n.encode()), actor, f"e:{n}", text, at.isoformat(),
         at.isoformat()))


async def sent(db, n, operation, *, ago=timedelta(minutes=2), question=False, text="x"):
    at = (NOW - ago).isoformat()
    await said(db, f"p{n}", "привет", actor=USER, ago=ago + timedelta(seconds=5))
    plan = {"claim": {"clause_kind": "question" if question else "statement"}}
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,status,response_message_id,"
        "sent_at,created_at,response_text) VALUES (?,'p',?,?,?,?,'k','b',?,?,?,0,0,'{}','{}',"
        "'sent',?,?,?,?)",
        (f"i:{n}", f"e:p{n}", GUILD, CHANNEL, USER, operation, n, json.dumps(plan),
         zlib.crc32(n.encode()), at, at, text))


def context(prompt, kind="question", addressed=True):
    return SimpleNamespace(
        prompt=prompt, structure=exchange_analysis(prompt)[0],
        turn_form=SimpleNamespace(kind=kind, addressed=addressed))


def ledger(*operations):
    rows = tuple(BoredomRow(f"i{i}", 60.0, f"k{i}", op, frozenset(), "", "", "")
                 for i, op in enumerate(operations))
    return BoredomLedger(rows, 0.76, 66528.0)


async def vocabulary(db):
    await said(db, "s1", "ты что спрашиваешь вообще")
    await said(db, "s2", "зачем ты это спрашиваешь")


async def repairs(db, prompt, *, kind="question", boredom=None, allowed=True, irritation=None):
    return await DialogueStructureLearning(db, None).repair_plans(
        event(), context(prompt, kind), "no_observed_social_form", allowed=allowed,
        irritation=irritation, boredom=boredom or ledger("express_query_unknown", "supposition",
                                                          "supposition", "supposition"))


async def test_the_budget_counts_repairs_and_not_the_agents_other_questions(db, flag_on,
                                                                           monkeypatch):
    partner = DialogueStructureLearning(db, None).interlocutor
    await sent(db, "c", "curiosity", question=True)
    assert await partner.clarification_allowed(event())  # its own curiosity is no repair
    await sent(db, "r", "repair", ago=timedelta(minutes=1), question=True)
    assert not await partner.clarification_allowed(event())
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT")
    assert not await partner.clarification_allowed(event())  # (before the flag: any question)


async def test_the_old_budget_is_spent_by_curiosity_without_the_flag(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    await sent(db, "c", "curiosity", question=True)
    assert not await DialogueStructureLearning(db, None).interlocutor.clarification_allowed(event())


async def test_a_fragment_is_asked_about_with_the_observed_verb(db, flag_on):
    await vocabulary(db)
    (plan,) = await repairs(db, "а потом что")
    assert plan.operation == "repair" and plan.plan.speech_act == "question"
    claim = plan.plan.claim
    assert claim.clause_kind == "question" and claim.predicate.lemma == "спрашивать"
    assert plan.premises[0].kind == "human_question_grammar"
    texts = [s.text for s in ResponseComposer(exchange_analysis("а потом что")[0])
             .compose(plan.plan, ())]
    assert texts == ["О чём ты спрашиваешь?"]


async def test_a_pressed_demand_is_asked_for_its_purpose(db, flag_on):
    await vocabulary(db)
    (plan,) = await repairs(db, "так почему же", kind="repair")
    assert plan.plan.claim.focus_role == "purpose"
    texts = [s.text for s in ResponseComposer(exchange_analysis("так почему же")[0])
             .compose(plan.plan, ())]
    assert texts == ["Зачем ты спрашиваешь?"]


async def test_nothing_is_authored_when_people_were_not_seen_to_ask(db, flag_on):
    assert await repairs(db, "а потом что") == ()
    await said(db, "z", "зачем это нужно")  # «зачем» alone is no operator
    assert await repairs(db, "а потом что") == ()


async def test_silence_stays_a_decision(db, flag_on):
    await vocabulary(db)
    hostile = SimpleNamespace(hostile=("register:x",), value=0.9, sensitivity=0.77)
    assert await repairs(db, "а потом что", irritation=hostile) == ()
    assert await repairs(db, "понятно", kind="fragment") == ()  # a closing of the topic
    assert await repairs(db, "а потом что", allowed=False) == ()  # the budget is spent
    spent = SimpleNamespace(hostile=(), value=0.9, sensitivity=0.77)
    assert await repairs(db, "а потом что", irritation=spent) == ()


async def test_a_statement_not_addressed_to_the_agent_is_not_repaired(db, flag_on):
    await vocabulary(db)
    learning = DialogueStructureLearning(db, None)
    plans = await learning.repair_plans(
        event(), context("а потом что", addressed=False), "no_observed_social_form",
        allowed=True, boredom=ledger("supposition"))
    assert plans == ()


async def test_repairs_never_outnumber_peoples_own_questions(db, flag_on):
    await vocabulary(db)
    # no exchange on the server has a question as its answer: the share is 0, a first repair
    # is allowed (nothing spent yet), a second one is not
    assert await repairs(db, "а потом что", boredom=ledger("supposition", "supposition"))
    assert not await repairs(db, "а потом что",
                             boredom=ledger("repair", "supposition", "supposition"))
    for i in range(10):  # people ask back in every other answer
        await said(db, f"a{i}", "ответ", ago=timedelta(days=2))
        await said(db, f"p{i}", "вопрос", ago=timedelta(days=2, minutes=1))
        await db.execute(
            "INSERT INTO language_observed_exchanges (answer_event_id,prompt_event_id,guild_id,"
            "channel_id,prompt_act,response_act,prompt_features_json,response_forms_json,"
            "topic_links_json,observed_at) VALUES (?,?,?,?,?,?,'[]','[]','[]',?)",
            (f"e:a{i}", f"e:p{i}", GUILD, CHANNEL, "statement",
             "information_question" if i % 2 else "statement", NOW.isoformat()))
    learning = DialogueStructureLearning(db, None)
    assert await learning.human_question_share(GUILD) == 0.5
    assert not await repairs(db, "а потом что", boredom=ledger("repair", "supposition"))
    assert await repairs(db, "а потом что", boredom=ledger(
        "repair", "supposition", "supposition", "supposition"))  # 2 of 5 = 40% stays under 50%


async def test_the_plan_is_deterministic(db, flag_on):
    await vocabulary(db)
    assert await repairs(db, "а потом что") == await repairs(db, "а потом что")


async def test_without_the_flag_there_is_no_repair(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    await vocabulary(db)
    assert await repairs(db, "а потом что") == ()


async def test_the_cause_of_a_world_limit_is_not_knowing_who_the_topic_is(db, flag_on):
    await said(db, "t", "Наполеон был прав?", actor=USER, ago=timedelta(minutes=3))
    learning = DialogueStructureLearning(db, None)
    prior = {"prompt_event_id": "e:t", "plan": {"knowledge_limit": {"query": {}}}}
    question, query, word = await learning._ignorance_identity(event(), prior)
    assert word == "кто" and question.focus_role == "predicate_complement"
    assert {(d.role, d.argument.head.lemma) for d in question.dependencies} == {
        ("subject", "наполеон"), ("predicate_complement", "кто")}
    assert query.guild_id == GUILD


async def test_the_cause_of_a_limit_about_the_agent_stays_with_the_old_reading(db, flag_on):
    await said(db, "t", "ты любишь спать?", actor=USER, ago=timedelta(minutes=3))
    prior = {"prompt_event_id": "e:t", "plan": {"knowledge_limit": {"query": {}}}}
    assert await DialogueStructureLearning(db, None)._ignorance_identity(event(), prior) is None
