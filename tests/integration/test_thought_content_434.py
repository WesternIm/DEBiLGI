"""Stage 4.34 (DEBILGI_THOUGHT_CONTENT): irritation R, separate from boredom B.

R is a function of the journal of what a person did to the agent and of the clock; an attack is
recognised by dictionary register + structure, never by a phrase list; R never moves facts.
"""

import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.affect import hostility_cues, irritation_reading
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import (
    BoredomLedger,
    BoredomRow,
    goal_material,
)
from debilgi.language.dialogue_learning import normalized_response
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 21, 17, 50, tzinfo=UTC)
GUILD, CHANNEL, USER, OTHER = 1, 10, 100, 101


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "content434.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def ev(n, text="x", *, actor=USER, guild=GUILD, at=NOW):
    return Event(f"e:{n}", guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text,
                  "author_is_bot": False, "bot_user_id": 7, "bot_mentioned": True}, "discord")


async def said(db, n, text, *, actor=USER, ago=timedelta(seconds=30), status="withheld",
               guild=GUILD, reply=None):
    at = NOW - ago
    await EventRepository(db).append(ev(f"t{n}", text, actor=actor, guild=guild, at=at))
    await db.execute(
        "INSERT INTO language_response_traces (trace_id,source_event_id,guild_id,channel_id,"
        "user_id,prompt,status,final_text,response_message_id,started_at) VALUES "
        "(?,?,?,?,?,?,?,?,?,?)",
        (f"tr:{n}", f"e:t{n}", guild, CHANNEL, actor, "<@7>" + text, status, reply,
         None, at.isoformat()))


# ------------------------------------------------------------------ recognising an attack
@pytest.mark.parametrize("text", ["але пидарас", "ты гандон", "пошел нахуй", "ты пойдешь нахуй?"])
def test_an_attack_is_read_from_register_and_target(text):
    assert hostility_cues(text)


@pytest.mark.parametrize("text", [
    "привет", "что делаешь", "ты молодец", "ты красава", "я твой друг", "он гандон",
    "ты не рад?", "ты тусил вчера", "играю в игру на компе", "ну так что",
])
def test_ordinary_turns_are_not_attacks(text):
    assert hostility_cues(text) == ()


def test_the_cues_name_their_evidence_not_a_phrase():
    assert all(c.startswith(("register:", "negative_valence")) for c in hostility_cues("ты гандон"))


# ------------------------------------------------------------------------- R from the journal
async def test_without_the_flag_there_is_no_irritation(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    assert await irritation_reading(db, ev(1), "пошел нахуй") is None


async def test_irritation_grows_with_repeated_attacks(db, flag_on):
    values = []
    for n in range(4):
        values.append((await irritation_reading(db, ev("now", "пошел нахуй"), "пошел нахуй")).value)
        await said(db, n, "пошел нахуй", ago=timedelta(seconds=30 * (4 - n)))
    assert values[0] < values[1] < values[2] < values[3] < 1.0
    assert values[0] > 0.0


async def test_irritation_fades_with_the_clock_and_is_not_stored(db, flag_on):
    await said(db, 1, "пошел нахуй", ago=timedelta(seconds=20))
    await said(db, 2, "ты гандон", ago=timedelta(seconds=10))
    fresh = (await irritation_reading(db, ev("now", "что делаешь"), "что делаешь")).value
    later = (await irritation_reading(
        db, ev("now2", "что делаешь", at=NOW + timedelta(hours=2)), "что делаешь")).value
    assert fresh > 0.5 and later < 0.01
    assert (await db.fetchone("SELECT COUNT(*) FROM affect_emotions"))[0] == 0


async def test_irritation_is_per_person_and_per_server(db, flag_on):
    await said(db, 1, "пошел нахуй", actor=OTHER)
    await said(db, 2, "пошел нахуй", guild=2)
    assert (await irritation_reading(db, ev("now", "что делаешь"), "что делаешь")).value == 0.0


async def test_a_normal_question_after_an_attack_is_not_hostile_itself(db, flag_on):
    await said(db, 1, "пошел нахуй")
    reading = await irritation_reading(db, ev("now", "что делаешь"), "что делаешь")
    assert reading.hostile == () and reading.value > 0.0


async def test_repeating_a_question_after_an_answer_is_stronger_than_after_silence(db, flag_on):
    await said(db, 1, "медведь или муравей", status="sent", reply="медведь")
    asked = "медведь или муравей"
    after_answer = (await irritation_reading(db, ev("now", asked), asked)).value
    await db.execute("UPDATE language_response_traces SET status='withheld', final_text=NULL")
    after_silence = (await irritation_reading(db, ev("now", asked), asked)).value
    assert after_answer > after_silence > 0.0


async def test_different_texts_are_not_a_repeat(db, flag_on):
    await said(db, 1, "у тебя есть сознание?", status="sent", reply="возможно")
    now = ev("now", "а ты предположи")
    assert (await irritation_reading(db, now, "а ты предположи")).value == 0.0


async def test_irritation_is_deterministic(db, flag_on):
    await said(db, 1, "пошел нахуй")
    a = await irritation_reading(db, ev("now", "пошел нахуй"), "пошел нахуй")
    assert a == await irritation_reading(db, ev("now", "пошел нахуй"), "пошел нахуй")
    assert a.impulses[0]["event_id"] == "e:t1"


async def test_an_attack_is_not_a_fact_about_the_agent(db, flag_on):
    before = [(await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in (
        "meaning_claims", "meaning_evidence", "meaning_support",
        "continuous_cognition_hypotheses")]
    await said(db, 1, "ты гандон")
    await irritation_reading(db, ev("now", "ты гандон"), "ты гандон")
    after = [(await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in (
        "meaning_claims", "meaning_evidence", "meaning_support",
        "continuous_cognition_hypotheses")]
    assert before == after


# --------------------------------------------------------- B: the same skeleton, other material
def _row(material, key="k"):
    return BoredomRow("i:1", 10.0, key, "compose_reaction:slot_question", frozenset(),
                      normalized_response("какую книгу читаешь?"), "r:1", "x", None, True,
                      frozenset(material))


def test_a_question_about_other_material_is_not_a_repeat():
    ledger = BoredomLedger((_row({"книга"}),), 0.76, 66528.0)
    same = ledger.reading(behavior_key="k", material=frozenset({"книга"}))
    other = ledger.reading(behavior_key="k", material=frozenset({"журнал"}))
    assert same.value > 0.0 and other.value == 0.0


def test_without_material_the_old_identity_holds():
    ledger = BoredomLedger((_row(set()),), 0.76, 66528.0)
    assert ledger.reading(behavior_key="k").value > 0.0


def test_goal_material_reads_topics_and_bound_roles():
    goal = {"topics": ["журнал"], "known_roles": [["subject", "ты"], ["direct_object", "журнал"]]}
    assert "журнал" in goal_material(goal)


async def test_an_ellipsis_resolved_to_the_old_question_is_not_a_repeat(db, flag_on):
    """The journal holds what the person wrote; the agent's resolved prompt is not a repeat."""
    await said(db, 1, "у тебя есть сознание?", status="sent", reply="возможно")
    now = ev("now", "а ты предположи")
    assert (await irritation_reading(db, now, "у тебя есть сознание?")).value == 0.0
