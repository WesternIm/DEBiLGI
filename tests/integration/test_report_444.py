"""Stage 4.44 (Э7): the reply of a person to the agent's question is received, kept as that person's
position (``meaning_positions``, scope «report»), and never becomes a claim or a fact. Flag
``DEBILGI_THOUGHT_CONTENT`` only; the position is read back only for the person who said it.
"""

import json
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import (
    REPORT_KEY,
    REPORT_SCOPE,
    MeaningIntentRepository,
    move_type,
    reaction_class,
)
from debilgi.language.dialogue_learning import (
    DialogueStructureLearning,
    answered_topics,
    reported_words,
)
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, USER, OTHER = 1, 10, 100, 200
QUESTION = {"claim": {"clause_kind": "question", "dependencies": [
    {"role": "subject", "argument": {"head": {"lemma": "наполеон"}}}]}}
ANSWER = "Наполеон Бонапарт — французский полководец и император, живший давно"


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "report444.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def message(n, text, *, user=USER):
    return Event(f"e:{n}", GUILD, CHANNEL, user, None, EventType.MESSAGE_CREATED, NOW,
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
                  "bot_user_id": 7, "bot_mentioned": True}, "discord")


def asked(operation="curiosity"):
    return {"intention_id": "i:1", "operation": operation, "plan_json": json.dumps(QUESTION)}


async def received(db, repository, n, text, *, user=USER, operation="curiosity"):
    event = message(n, text, user=user)
    await EventRepository(db).append(event)
    return event, await repository._observe_answer(
        event, asked(operation), QUESTION, "conversation_continued")


async def positions(db):
    return [dict(r) for r in await db.fetchall("SELECT * FROM meaning_positions")]


def test_a_reply_answers_only_what_the_question_was_about():
    assert answered_topics(ANSWER, QUESTION) == ("наполеон",)
    assert answered_topics("а он что?", QUESTION) == ()          # a question back
    assert answered_topics("понятно", QUESTION) == ()            # a closing
    assert answered_topics("ахахах", QUESTION) == ()             # laughter
    assert answered_topics("сегодня жарко", QUESTION) == ()      # another topic: evaded


def test_the_first_words_of_the_reply_are_what_is_kept():
    said = "Наполеон Бонапарт — французский полководец и император"
    assert reported_words("<@7> " + ANSWER) == said


def test_the_new_receptions_are_read_from_the_outcome():
    assert reaction_class("Наполеон — император", "answered", False) == "answered"
    assert reaction_class("сегодня жарко", "evaded", False) == "evaded"
    assert reaction_class("пошёл ты", "evaded", True) == "hostile"
    assert move_type("acknowledge_report") == "reaction"
    assert move_type("curiosity") == "counter"


async def test_an_answer_is_his_position_not_a_claim(db, flag_on):
    repository = MeaningIntentRepository(db)
    event, outcome = await received(db, repository, "1", ANSWER)
    assert outcome == "answered"
    [row] = await positions(db)
    assert (row["meaning_key"], row["scope"], row["actor_id"], row["source_event_id"]) == (
        REPORT_KEY + "наполеон", REPORT_SCOPE, USER, event.event_id)
    # no claim of anyone, no evidence, no support is made by receiving it
    for table in ("meaning_claims", "meaning_evidence", "meaning_support"):
        assert (await db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0


async def test_an_evasion_is_not_kept(db, flag_on):
    repository = MeaningIntentRepository(db)
    _, outcome = await received(db, repository, "1", "сегодня жарко")
    assert outcome == "evaded" and await positions(db) == []


async def test_only_the_person_himself_gets_his_position_back(db, flag_on):
    repository = MeaningIntentRepository(db)
    event, _ = await received(db, repository, "1", ANSWER)
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (event.event_id, GUILD, CHANNEL, 1, USER, "assertion", "not_question", ANSWER,
         NOW.isoformat()))
    learning = DialogueStructureLearning(db, None)
    later = message("later", "а Наполеон был прав?")
    later = Event(later.event_id, GUILD, CHANNEL, USER, None, EventType.MESSAGE_CREATED,
                  datetime(2026, 11, 24, tzinfo=UTC), later.payload, "discord")
    found = await learning._reported_by(later, ("наполеон",))
    assert found and found["topic"] == "наполеон" and found["text"] == ANSWER
    stranger = Event("e:s", GUILD, CHANNEL, OTHER, None, EventType.MESSAGE_CREATED,
                     later.utc_timestamp, later.payload, "discord")
    assert await learning._reported_by(stranger, ("наполеон",)) is None


async def test_without_the_flag_nothing_is_received(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    repository = MeaningIntentRepository(db)
    _, outcome = await received(db, repository, "1", ANSWER)
    assert outcome == "conversation_continued" and await positions(db) == []
    learning = DialogueStructureLearning(db, None)
    assert await learning._reported_by(message("x", "?"), ("наполеон",)) is None


def test_the_floor_does_not_turn_an_inner_failure_into_silence():
    """4.44, debt 1: below the floor nothing is said; a failed first pass waives it (the second
    pass of the failure for any eligible candidate, a turn with no attack only for the honest
    limit); without the waiver the earlier reading stays."""
    from debilgi.language.arbitration import (
        ArbitrationProfile,
        ResponseArbitrator,
        ResponseCandidate,
    )

    def candidate(text, status):
        return ResponseCandidate.create(
            event_id="event:one", source="semantic_reasoning", renderer="test", text=text,
            speech_act="statement", source_confidence=0.6,
            metadata={"communicative_status": status})

    def decide(*items, waive_floor=None):
        return ResponseArbitrator(minimum_score=100.0).decide(
            event_id="event:one", prompt="а как", focus_concepts=(), candidates=tuple(items),
            recent_replies=(), profile=ArbitrationProfile.baseline(1), waive_floor=waive_floor)

    limit, question = candidate("Не знаю.", "express_limit"), candidate("О чём ты?", "ask")
    assert decide(limit, question).winner is None
    assert decide(limit, question, waive_floor="repair").winner is not None
    assert decide(question, waive_floor="limit").winner is None
    assert decide(limit, question, waive_floor="limit").winner.text == "Не знаю."
