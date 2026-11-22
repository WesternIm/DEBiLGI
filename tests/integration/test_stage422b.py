"""Stage 4.22b: retelling, dangling anaphora, own wake as an observed action, own-limit retelling.

Cases deliberately use verbs, pronouns and forms that the live dialogue never contained.
"""

import zlib
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import Known, Unknown
from debilgi.language.communication import ReaderGoal, reader_goal
from debilgi.language.dialogue_learning import (
    _reaction_parts,
    own_turn_plan,
    reverse_reaction_check,
)
from debilgi.language.generative_planner import knowledge_limit_plan
from debilgi.language.planning import ReplyContext
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 20, 4, 40, tzinfo=UTC)
GUILD, CHANNEL, USER, BOT = 1, 10, 100, 777


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "s422b.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def ev(text, n=1, *, guild=GUILD, at=None, mention=True):
    return Event(f"e:{guild}:{n}", guild, CHANNEL, USER, None, EventType.MESSAGE_CREATED,
                 at or NOW + timedelta(minutes=n),
                 {"message_id": zlib.crc32(f"{guild}{n}".encode()), "content": text,
                  "author_is_bot": False, "bot_user_id": BOT, "bot_mentioned": mention},
                 "discord")


async def _limit(db, text):
    event = ev(text)
    result = await PropositionMemory.for_meaning(db).retrieve_claims(event, text)
    context = SimpleNamespace(structure=analyze_structured_utterance(text), claim_result=result,
                              response_options=())
    return knowledge_limit_plan(context)


def _said(derived):
    return realize_thought(derived.plan.knowledge_limit.question, limit=1)[0].text


# --- retelling: person of possessives, subjectless clauses, «бы» --------------------------
@pytest.mark.parametrize(
    "heard, said",
    [
        ("ты видел мою кошку?", "Я видел твою кошку?"),
        ("ты слышал моих друзей?", "Я слышал твоих друзей?"),
        ("ты вернёшь мне мою книгу?", "Я верну тебе твою книгу?"),
        ("ты бы ответил на мой вопрос?", "Я ответил бы на твой вопрос?"),
        ("ты бы хотел полететь?", "Я хотел бы полететь?"),
    ],
)
async def test_limit_retells_the_clause_from_the_agents_side(db, heard, said):
    derived = await _limit(db, heard)
    assert derived is not None and derived.operation == "express_query_unknown"
    assert _said(derived) == said


@pytest.mark.parametrize("heard, verb", [("просыпался же?", "просыпаться"),
                                         ("уснул?", "уснуть"), ("купил машину?", "купить"),
                                         ("ответил?", "ответить")])
async def test_subjectless_past_question_names_the_agent_as_subject(db, heard, verb):
    derived = await _limit(db, heard)
    question = derived.plan.knowledge_limit.question
    assert question is not None and question.predicate.lemma == verb
    assert next(d.argument.head.lemma for d in question.dependencies if d.role == "subject") == "я"


@pytest.mark.parametrize("heard", ["понял?", "взял?"])
async def test_unrealizable_retelling_keeps_the_general_limit(db, heard):
    derived = await _limit(db, heard)
    assert derived is not None and derived.operation == "express_query_unknown"
    assert derived.plan.knowledge_limit.question is None  # not silence: the general limit stays


# --- anaphora ---------------------------------------------------------------------------
def _core(slots):
    return SimpleNamespace(component="core", hypothesis=(), slots=slots, clause_kind="question",
                           meaning="slot_question")


def _plan(prompt):
    goal = reader_goal(analyze_structured_utterance(prompt))
    assert isinstance(goal, ReaderGoal)
    return SimpleNamespace(goal=goal, hypothesis=(), claim=SimpleNamespace(
        predicate=SimpleNamespace(lemma="slot_question")))


def _slots(*parts):
    return [{"part": p, "binding": b, "surface": s, "lemma": s} for p, b, s in parts]


@pytest.mark.parametrize(
    "prompt, text, dangling",
    [
        ("ты бы хотел полететь?", "кто их не хотел?)", True),
        ("что сделал бы сегодня утром?", "кто их не сделал?", True),
        ("любишь кошек?", "кто их не любит?", False),
        ("любишь кошку?", "кто её не любит?", False),
        ("любишь кота?", "кто её не любит?", True),  # other pronoun and gender: still caught
        ("любишь кошек?", "кто им не рад?", False),
        ("как дела?", "кто им не рад?", True),
    ],
)
def test_reverse_check_rejects_any_anaphor_without_antecedent(prompt, text, dangling):
    core = _core(_slots(("pronoun", "grammar", "кто"), ("verb", "prompt", "любить")))
    check = reverse_reaction_check(_plan(prompt), (core,), prompt, text)
    assert ("unbound_anaphor" in check["issues"]) is dangling


@pytest.mark.parametrize(
    "prompt, answer, kept",
    [
        ("Любишь пироги??", "А кто их не любит.", True),
        ("Любишь кошку??", "А кто её не любит.", True),
        ("Любишь кота??", "А кто её не любит.", False),
        ("Видел ёжиков??", "А кто их не видел.", True),
        ("Видел ёжика??", "А кто их не видел.", False),
        ("Видел ёжика??", "А кто его не видел.", True),
    ],
)
def test_learning_drops_a_reaction_whose_pronoun_has_no_antecedent_in_its_prompt(
        prompt, answer, kept):
    assert bool(_reaction_parts(prompt, answer)) is kept


# --- own limit retold with the verb that was said ----------------------------------------
async def test_retelling_a_limit_keeps_the_said_verb_and_the_focused_request(db):
    derived = await _limit(db, "ты сделал?")
    plan = derived.plan
    prior = {"plan": asdict(plan), "decision": {}, "prompt_event_id": "p:1",
             "receipt_id": "receipt:1:2", "response_text": "Я не знаю, сделал ли я.",
             "operations": {"epistemic": ("помнить", "human:grammar")}}
    query = await PropositionMemory.for_meaning(db).retrieve_claims(ev("ты сделал?"), "ты сделал?")
    context = ReplyContext(prompt="ты ответишь?", prior_bot_turn=prior, claim_result=query)
    kept = own_turn_plan(context, None)
    assert kept.plan.claim.predicate.lemma == plan.claim.predicate.lemma  # «знать», not «помнить»
    # the delivered limit answered another request than the focused one → nothing to retell
    other = await PropositionMemory.for_meaning(db).retrieve_claims(
        ev("ты просыпался?", 2), "ты просыпался?")
    focused = replace(context, claim_result=other, focus_reason="answer_control_keeps_request")
    assert own_turn_plan(focused, None) is None
    # the same request in the other aspect is the same request
    same = await PropositionMemory.for_meaning(db).retrieve_claims(
        ev("ты делал?", 3), "ты делал?")
    assert own_turn_plan(replace(context, claim_result=same,
                                 focus_reason="answer_control_keeps_request"), None) is not None


# --- the real wake as an observed own action --------------------------------------------
async def _asleep(db):
    clock = [NOW]
    service = ContinuousCognitionService(db, clock=lambda: clock[0], presence=AsyncMock())
    await service.restore_sleep()
    clock[0] += timedelta(minutes=31)
    await service.rhythm_once()
    assert service.sleeping
    return service, clock


async def _ask(db, text, n, *, guild=GUILD):
    event = ev(text, n, guild=guild, at=NOW + timedelta(minutes=40 + n))
    await EventRepository(db).append(event)
    return await PropositionMemory.for_meaning(db).retrieve_claims(event, text)


async def _claims(db):
    return [dict(r) for r in await db.fetchall(
        "SELECT source_kind, predicate, guild_id FROM meaning_claims ORDER BY claim_id")]


@pytest.mark.parametrize("asked", ["просыпался?", "проснулся?", "ты просыпался?", "ты проснулся?"])
async def test_a_tag_that_wakes_the_agent_becomes_a_fact_about_itself(db, asked):
    service, _ = await _asleep(db)
    tag = ev("привет", 1, at=NOW + timedelta(minutes=35))
    await EventRepository(db).append(tag)
    before = await _claims(db)
    await service.wait_until_awake(tag)
    rows = [r for r in await _claims(db) if r not in before]
    assert rows == [{"source_kind": "agent_observed", "predicate": "проснуться", "guild_id": GUILD}]
    result = await _ask(db, asked, 2)
    assert isinstance(result, Known)
    assert all(c.source_kind == "agent_observed" for c in result.claims)


async def test_without_a_transition_there_is_no_record(db):
    service = ContinuousCognitionService(db, clock=lambda: NOW, presence=AsyncMock())
    await service.restore_sleep()
    tag = ev("привет", 1)
    await EventRepository(db).append(tag)
    await service.wait_until_awake(tag)  # the agent was awake: nothing to wake from
    assert await _claims(db) == []
    assert isinstance(await _ask(db, "просыпался?", 2), Unknown)


async def test_the_wake_does_not_cross_servers(db):
    service, _ = await _asleep(db)
    tag = ev("привет", 1, at=NOW + timedelta(minutes=35))
    await EventRepository(db).append(tag)
    await service.wait_until_awake(tag)
    assert isinstance(await _ask(db, "просыпался?", 2, guild=2), Unknown)


async def test_a_wake_without_the_waking_event_is_not_written(db):
    service, clock = await _asleep(db)
    clock[0] += timedelta(minutes=95)  # a cycle boundary with no tag: no waking event
    await db.execute("UPDATE continuous_cognition_runtime SET pending_activity=1")
    await service.rhythm_once()
    assert await _claims(db) == []


# --- a statement about the agent is checked against its own observed actions (Ш8) --------
async def _wake_then(db, text, n, *, guild=GUILD, mention=True):
    service, _ = await _asleep(db)
    tag = ev("привет", 1, at=NOW + timedelta(minutes=35))
    await EventRepository(db).append(tag)
    await service.wait_until_awake(tag)
    event = ev(text, n, guild=guild, at=NOW + timedelta(minutes=40 + n), mention=mention)
    await EventRepository(db).append(event)
    return await PropositionMemory.for_meaning(db).retrieve_claims(event, text)


@pytest.mark.parametrize("said", ["ну ты же проснулся", "ты проснулся", "ты уже просыпался"])
async def test_statement_about_the_agents_own_action_is_confirmed_by_its_record(db, said):
    result = await _wake_then(db, said, 2)
    assert isinstance(result, Known)
    assert {c.source_kind for c in result.claims} == {"agent_observed"}


@pytest.mark.parametrize("said, kwargs", [
    ("ты не проснулся", {}),  # negation: a journal is no proof of absence
    ("ты заснул", {}),  # another action: no record
    ("он проснулся", {}),  # not about the addressee
    ("ты проснулся", {"guild": 2}),  # another server
    ("ты проснулся", {"mention": False}),  # not addressed to the agent
])
async def test_other_statements_keep_their_ordinary_result(db, said, kwargs):
    assert isinstance(await _wake_then(db, said, 2, **kwargs), Unknown)


def test_computed_form_turns_plural_second_person_to_singular_by_features():
    from debilgi.language.dialogue_learning import _computed_form

    unit = {"source_id": "s:1", "kind": "verb", "pivot": "любить", "slots": [
        {"binding": "grammar", "part": "pronoun", "lemma": "вы", "surface": "вы",
         "features": {"number": "plural", "person": "second"}},
        {"binding": "grammar", "part": "pronoun", "lemma": "мы", "surface": "мы",
         "features": {"number": "plural", "person": "first"}},
        {"binding": "prompt", "part": "verb", "lemma": "любить", "surface": "любите",
         "features": {}, "source_role": "predicate"}]}
    core, _ = _computed_form(ev("любишь кошек?"), "любишь кошек?", unit, meaning="slot_question",
                             activation=0.33)
    slots = {s["lemma"]: s for s in core.slots}
    assert "ты" in slots and slots["ты"]["features"]["number"] == "singular"
    assert "мы" in slots  # only the addressee's plural is turned


@pytest.mark.parametrize("heard, said", [
    ("а что ты хочешь?", "Что я хочу?"),
    ("что ты любишь?", "Что я люблю?"),
    ("что ты ищешь?", "Что я ищу?"),
])
async def test_open_slot_about_the_agent_limits_its_own_predicate(db, heard, said):
    derived = await _limit(db, heard)
    assert derived is not None and derived.operation == "express_query_unknown"
    assert _said(derived) == said

