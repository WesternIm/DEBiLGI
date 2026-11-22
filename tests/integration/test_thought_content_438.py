"""Stage 4.38 (Э2): curiosity — a question about the unknown topic next to the honest limit.

Flag ``DEBILGI_THOUGHT_CONTENT`` only. The plan is built from the journals and the clock
(U, I, B, R); the surface comes from the composer; a question is never a claim of anyone.
"""

import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import ClaimQuery
from debilgi.language.communication import (
    KnowledgeLimit,
    MessagePlan,
    ResponseComposer,
    reader_goal,
)
from debilgi.language.dialogue_learning import DialogueStructureLearning, exchange_analysis
from debilgi.language.generative_planner import DerivedPlan
from debilgi.language.semantic_graph import ClauseThought, ConceptLexeme
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, USER, OTHER = 1, 10, 100, 101


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "content438.sqlite3", Path("migrations"))
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


async def said(db, n, text, *, actor, ago):
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


def asked(text, *, self_ask=False):
    structure = exchange_analysis(text)[0]
    query = ClaimQuery(GUILD, CHANNEL, USER, NOW.isoformat(), "e:q", predicate="быть",
                       arguments=(), requested_role=None)
    limit = ClauseThought(ConceptLexeme("знать", "знать", "verb"), (), negated=True,
                          evidence=("e:q",))
    plan = MessagePlan(reader_goal(structure), limit, knowledge_limit=KnowledgeLimit(
        query, None, None), mixture=(("known", 1.0),))
    context = SimpleNamespace(
        structure=structure, claim_result=SimpleNamespace(status="Unknown", query=query),
        turn_form=SimpleNamespace(addressed=True), affect_features=(), relation=None)
    planning = SimpleNamespace(proposals=(DerivedPlan(plan, "express_query_unknown", ()),))
    return context, planning


async def plans(db, text, *, irritation=None, boredom=None):
    context, planning = asked(text)
    learning = DialogueStructureLearning(db, None)
    return await learning.curiosity_plans(
        event(), context, planning, irritation=irritation, boredom=boredom)


async def test_without_the_flag_there_is_no_inquiry(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    assert await plans(db, "Гитлер был прав?") == ()


async def test_an_unknown_person_is_asked_with_кто(db, flag_on):
    (plan,) = await plans(db, "Гитлер был прав?")
    assert plan.operation == "curiosity" and plan.answer_slot == "кто"
    claim = plan.plan.claim
    assert claim.clause_kind == "question" and claim.evidence == ("e:q",)
    assert {(d.role, d.argument.head.lemma) for d in claim.dependencies} == {
        ("subject", "гитлер"), ("predicate_complement", "кто")}
    assert plan.premises[0].text == "гитлер"
    assert plan.plan.knowledge_limit is None and not plan.plan.hypothesis
    surfaces = ResponseComposer(exchange_analysis("Гитлер был прав?")[0]).compose(plan.plan, ())
    assert [s.text for s in surfaces] == ["Кто Гитлер?"]


async def test_a_thing_is_asked_with_что_and_a_definition_request_keeps_the_limit(db, flag_on):
    (plan,) = await plans(db, "что такое шампунь?")
    assert plan.answer_slot == "что"
    # asked to define it, the agent does not know and asks back: the limit stays in the plan
    assert plan.plan.knowledge_limit is not None
    assert plan.plan.follow_up is not None and plan.plan.follow_up.clause_kind == "question"


async def test_the_agent_and_the_askers_own_matters_are_no_topic(db, flag_on):
    assert await plans(db, "я твой шатал") == ()
    assert await plans(db, "мой шампунь это средство для мытья волос") == ()
    assert await plans(db, "ты честный?") == ()


async def test_an_attack_is_not_asked_about(db, flag_on):
    hostile = SimpleNamespace(hostile=("register:долбоёб",), value=0.9)
    assert await plans(db, "Гитлер был прав?", irritation=hostile) == ()


async def test_strength_falls_with_people_who_already_speak_of_the_topic(db, flag_on):
    (alone,) = await plans(db, "Гитлер был прав?")
    await said(db, "a", "Гитлер был в Германии", actor=OTHER, ago=timedelta(days=1))
    await said(db, "b", "про Гитлера вчера читал", actor=OTHER + 1, ago=timedelta(days=2))
    (spoken,) = await plans(db, "Гитлер был прав?")
    assert spoken.plan.claim.confidence < alone.plan.claim.confidence / 2


async def test_a_topic_asked_and_answered_is_not_asked_again(db, flag_on):
    (plan,) = await plans(db, "Гитлер был прав?")
    await said(db, "p", "Гитлер был прав?", actor=USER, ago=timedelta(minutes=6))
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,status,response_message_id,"
        "sent_at,created_at) VALUES ('i1','p','e:p',?,?,?,'k','b','curiosity','p',?,0,0,'{}','{}',"
        "'sent',5,?,?)",
        (GUILD, CHANNEL, USER, __import__("json").dumps(
            {"claim": {"dependencies": [{"role": "subject", "argument": {"head": {
                "lemma": "гитлер"}}}]}, "goal": {"topics": ["гитлер"]}}),
         (NOW - timedelta(minutes=5)).isoformat(), (NOW - timedelta(minutes=5)).isoformat()))
    await said(db, "ans", "это диктатор, который развязал войну", actor=USER,
               ago=timedelta(minutes=4))
    assert await plans(db, "Гитлер был прав?") == ()
    assert plan  # (it was asked before the answer came)


async def test_the_plan_is_deterministic(db, flag_on):
    assert await plans(db, "Гитлер был прав?") == await plans(db, "Гитлер был прав?")
