"""Stage 4.45 (Э8): S, the agent's own attitude to a topic, and a view as a move to the person.

Flag ``DEBILGI_THOUGHT_CONTENT`` only. A view is never a fact: it is kept as a stance of the agent
(`language_positions`) and never written to claims, evidence, support or hypotheses.
"""

import json
import zlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import (
    MeaningIntentRepository,
    Sympathy,
    move_type,
    reception_cost,
)
from debilgi.language.communication import MessagePlan, plan_issues, reader_goal
from debilgi.language.dialogue_learning import held_stance
from debilgi.language.position_reasoning import PositionRepository, StoredPosition, _position_key
from debilgi.language.semantic_graph import parse_proposition
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, ALICE, BOB = 1, 10, 100, 200
CLAIM = {"claim": {"clause_kind": "statement", "predicate": {"lemma": "изучить"},
                   "dependencies": [{"role": "direct_object",
                                     "argument": {"head": {"lemma": "конструкция"}}}]}}
LIMIT = {**CLAIM, "knowledge_limit": {"query": {}}}


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "stance445.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def event(n, text="x", *, user=ALICE, at=NOW):
    return Event(f"e:{n}", GUILD, CHANNEL, user, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
                  "bot_user_id": 7, "bot_mentioned": True}, "discord")


async def turn(db, n, *, user, outcome, plan=CLAIM, operation="retrieve_self_claim", days=1.0):
    """A delivered turn of the agent and the reaction of the person to it."""
    sent = NOW - timedelta(days=days)
    prompt, reaction = event(f"p{n}", user=user, at=sent), event(f"r{n}", user=user, at=sent)
    for item in (prompt, reaction):
        await EventRepository(db).append(item)
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id, parent_intent_id, prompt_event_id,"
        " guild_id, channel_id, user_id, context_key, behavior_key, operation, plan_key, plan_json,"
        " expected_reward, sample_count, selection_json, state_json, created_at, status,"
        " response_message_id, sent_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,0,0,'{}','{}',?,'sent',?,?)",
        (f"i:{n}", "", prompt.event_id, GUILD, CHANNEL, user, "c", f"b:{n}", operation, f"k:{n}",
         json.dumps(plan), sent.isoformat(), 1000 + n, sent.isoformat()))
    await db.execute(
        "INSERT INTO language_meaning_consequences (intention_id, event_id, outcome_type,"
        " observed_at) VALUES (?,?,?,?)", (f"i:{n}", reaction.event_id, outcome, sent.isoformat()))


async def other_turns(db, user, count, start=100, mixed=True):
    """What the person did otherwise (not about the topic): his average reaction to the agent."""
    other = {"claim": {"clause_kind": "statement", "predicate": {"lemma": "отправить"},
                       "dependencies": []}}
    for k in range(count):  # an even mix of going on and being rejected: the average is 0
        await turn(db, start + k, user=user, plan=other,
                   outcome="conversation_continued" if k % 2 == 0 or not mixed
                   else "explicit_rejection")


class Warmth:
    """cognition entity models of two people: one warm to the agent, one cold"""

    def __init__(self, table):
        self.table = table

    async def entity_model(self, guild, kind, entity):
        return SimpleNamespace(bot_action_response=self.table.get(entity, 0.5))


async def sympathy(db, topic="изучить", cognition=None, since=None):
    repository = MeaningIntentRepository(db, cognition=cognition)
    return await repository.sympathy(event("ask"), {topic}, since=since)


async def test_no_experience_is_no_attitude(db, flag_on):
    assert (await sympathy(db)) == Sympathy()
    # what the agent said only to admit not knowing is no experience of the topic
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation", plan=LIMIT)
    await turn(db, 2, user=ALICE, outcome="explicit_confirmation", plan=LIMIT)
    assert (await sympathy(db)).value == 0.0


async def test_flag_off_there_is_no_attitude(db, monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "0")
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation")
    assert (await sympathy(db)) == Sympathy()


async def test_reactions_count_against_the_persons_own_average(db, flag_on):
    await other_turns(db, ALICE, 4, mixed=False)  # he goes on after everything the agent says
    await turn(db, 1, user=ALICE, outcome="conversation_continued")
    # a topic spoken of while talk went on is not liked for that: no better than the average
    assert (await sympathy(db)).value == 0.0
    await turn(db, 2, user=ALICE, outcome="follow_up_question")  # pressed: worse than average
    assert (await sympathy(db)).value < 0.0


async def test_one_person_is_one_voice(db, flag_on):
    await other_turns(db, ALICE, 2)
    await turn(db, 1, user=ALICE, outcome="explicit_rejection")
    one = (await sympathy(db)).value
    for k in range(2, 12):
        await turn(db, k, user=ALICE, outcome="explicit_rejection")
    many = await sympathy(db)
    assert many.people == 1 and many.outcomes == 11
    assert one < 0.0 and many.value < 0.0 and abs(many.value) < 1.0  # saturates, never a verdict


async def test_the_warm_persons_reactions_weigh_more(db, flag_on):
    """E5: two people disagree; the one who is dear to the agent tips it, and basis shows P."""
    await other_turns(db, ALICE, 2)
    await other_turns(db, BOB, 2, start=200)
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation")   # good for him (a better day)
    await turn(db, 2, user=BOB, outcome="explicit_rejection")        # bad for the other
    even = await sympathy(db)
    warm = await sympathy(db, cognition=Warmth({ALICE: 0.9, BOB: 0.1}))
    cold = await sympathy(db, cognition=Warmth({ALICE: 0.1, BOB: 0.9}))
    assert abs(even.value) < 1e-9
    assert warm.value > 0.0 > cold.value
    assert sorted(warm.weights) == [pytest.approx(0.2), pytest.approx(1.8)]  # P is in the basis
    assert warm.people == 2 and not hasattr(warm, "authors")  # counts, never who


async def test_a_person_without_a_model_is_neutral(db, flag_on):
    await other_turns(db, ALICE, 2)
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation")
    plain = await sympathy(db)
    assert plain.weights == (1.0,) and plain.value > 0.0


async def test_only_this_server_is_read(db, flag_on):
    await other_turns(db, ALICE, 2)
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation")
    await db.execute("UPDATE language_meaning_intentions SET guild_id=2")
    assert (await sympathy(db)).value == 0.0


async def test_new_experience_alone_is_read_for_a_turn_of_a_view(db, flag_on):
    await other_turns(db, ALICE, 2)
    await turn(db, 1, user=ALICE, outcome="explicit_confirmation", days=3.0)
    await turn(db, 2, user=ALICE, outcome="explicit_rejection", days=0.5)
    after = (NOW - timedelta(days=1)).isoformat()
    assert (await sympathy(db, since=after)).value < 0.0 < (await sympathy(db)).value + 1.0
    assert (await sympathy(db, since=after)).outcomes == 1


def stored(stance="support", score=0.6):
    return StoredPosition("k", "изучить", "любить", stance, score, abs(score), (), 1)


def test_the_held_stance_stays_against_one_person_pressing():
    held = stored()
    one_person = Sympathy(value=-0.99, people=1)
    assert held_stance(held, Sympathy(value=0.3, people=1), one_person) == ("support", 0.6, False)


def test_the_held_stance_turns_on_new_experience_of_two_people_as_strong():
    held = stored()
    assert held_stance(held, Sympathy(), Sympathy(value=-0.7, people=2)) == (
        "oppose", -0.7, True)
    # weaker than the ground it stood on, or from a single person: it stays
    assert held_stance(held, Sympathy(), Sympathy(value=-0.5, people=2))[2] is False
    assert held_stance(held, Sympathy(), Sympathy(value=-0.9, people=1))[2] is False
    # the same sign is no reason to turn
    assert held_stance(held, Sympathy(), Sympathy(value=0.9, people=3))[2] is False


def test_with_no_stance_held_the_sign_of_s_decides():
    assert held_stance(None, Sympathy(value=0.2, people=1), Sympathy())[0] == "support"
    assert held_stance(None, Sympathy(value=-0.2, people=1), Sympathy())[0] == "oppose"


def test_a_view_is_a_move_of_its_own_kind_for_the_forecast():
    assert move_type("opinion") == "stance"
    assert reception_cost("stance", {"laughed": 0.6, "rejected": 0.2}) == pytest.approx(0.8)
    assert reception_cost("counter", {"laughed": 0.6, "rejected": 0.2}) == pytest.approx(0.2)


def view_plan(operator, kind="opinion"):
    thought = parse_proposition(analyze_structured_utterance("ты любишь изучать?"),
                                allow_question=True, perspective="user", action_object=True)
    thought = replace(thought, evidence=("m:1",))
    return MessagePlan(reader_goal(analyze_structured_utterance("ты любишь изучать?")), thought,
                       speech_act="opinion", hypothesis=(
                           ("kind", kind), ("operator", operator), ("operator_source", "m:1")))


def test_a_view_without_the_observed_frame_is_refused_a_marked_one_passes(flag_on):
    assert plan_issues(view_plan("думаю")) == ()
    assert plan_issues(view_plan("точно"))          # not a mark of a view: an unmarked assertion
    assert plan_issues(view_plan("наверное"))       # the hedge of a guess does not mark a view


async def test_a_delivered_view_is_a_stance_and_never_a_claim_or_hypothesis(db, flag_on):
    prompt = event("q1", "ты любишь изучать?")
    await EventRepository(db).append(prompt)
    hypothesis = {"kind": "opinion", "operator": "думаю", "operator_source": "m:1",
                  "topic": "изучать", "quality": "любить", "stance": "support", "score": "0.6",
                  "ground": json.dumps(["i:1"]), "revised": "", "basis": "{}"}
    plan = {**CLAIM, "hypothesis": list(hypothesis.items())}
    plan["claim"] = {**plan["claim"], "confidence": 0.6}
    plan["hypothesis"] = hypothesis
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id, parent_intent_id, prompt_event_id,"
        " guild_id, channel_id, user_id, context_key, behavior_key, operation, plan_key, plan_json,"
        " expected_reward, sample_count, selection_json, state_json, created_at, status)"
        " VALUES ('i:v','',?,?,?,?,'c','b','opinion','k',?,0,0,'{}','{}',?,'prepared')",
        (prompt.event_id, GUILD, CHANNEL, ALICE, json.dumps(plan), NOW.isoformat()))
    before = {t: (await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in (
        "meaning_claims", "meaning_evidence", "meaning_support", "meaning_positions",
        "continuous_cognition_hypotheses")}
    repository = MeaningIntentRepository(db)
    assert await repository.record_sent(prompt, intention_id="i:v", response_message_id=5,
                                        created_at=NOW, text="Думаю, я люблю изучать.")
    after = {t: (await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in before}
    assert after == before
    held = await PositionRepository(db).recall(GUILD, _position_key("изучать", "любить"))
    assert held is not None and held.stance == "support"
    # the only other trace: the record of the expression (the reactions to it are read by it)
    assert (await db.fetchone("SELECT COUNT(*) FROM language_expressed_positions"))[0] == 1


@pytest.mark.parametrize(("text", "expected"), [
    ("ты любишь изучать?", ("любить", "изучать")),
    ("тебе нравится музыка?", ("нравиться", "музыка")),
    ("что ты думаешь о Гитлере?", ("нравиться", "гитлер")),
    ("а Гитлер был прав?", ("право", "гитлер")),
    ("ты умный?", None),          # about the agent itself: not a view on a topic
    ("ты любишь меня?", None),    # a person, not a topic
    ("кто ты?", None),
])
def test_which_questions_ask_for_a_view_and_on_what(flag_on, text, expected):
    from debilgi.language.dialogue_learning import _opinion_reading

    asked = parse_proposition(analyze_structured_utterance(text), allow_question=True,
                              perspective="user", evaluation=True, action_object=True)
    reading = _opinion_reading(asked) if asked is not None else None
    assert (reading[:2] if reading else None) == expected
