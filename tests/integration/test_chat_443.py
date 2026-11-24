"""Stage 4.43 (Э6): the mentality of the chat from the reactions of ≥2 different people; the sleep
hypothesis is no opinion on a question about a third party. Flag ``DEBILGI_THOUGHT_CONTENT`` only.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition import chat_norms
from debilgi.cognition.affect import hostility_cues
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.generative_planner import ReplyContext
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)
GUILD, CHANNEL = 1, 10
A, B, D, E = 101, 102, 103, 104


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "chat443.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "0")


async def turn(db, n, text, actor, *, ago, guild=GUILD, source="discord"):
    at = NOW - ago
    await EventRepository(db).append(Event(
        f"e:{n}", guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
        {"message_id": zlib.crc32(str(n).encode()), "content": text, "author_is_bot": False,
         "bot_user_id": 7, "bot_mentioned": False}, source))
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", guild, CHANNEL, n, actor, text, at.isoformat()))


async def exchange(db, n, prompt, p_actor, answer, a_actor, *, ago=timedelta(hours=3),
                   guild=GUILD, source="discord"):
    await turn(db, 10 * n, prompt, p_actor, ago=ago + timedelta(seconds=5), guild=guild,
               source=source)
    await turn(db, 10 * n + 1, answer, a_actor, ago=ago, guild=guild, source=source)
    await db.execute(
        "INSERT INTO language_observed_exchanges (answer_event_id,prompt_event_id,guild_id,"
        "channel_id,prompt_act,response_act,prompt_features_json,response_forms_json,"
        "topic_links_json,observed_at) VALUES (?,?,?,?,'q','fragment','[]','[]','[]',?)",
        (f"e:{10 * n + 1}", f"e:{10 * n}", guild, CHANNEL, (NOW - ago).isoformat()))


async def norms(db, guild=GUILD):
    return await chat_norms.load(db, guild)


# --- debt 1: the sleep hypothesis is not an opinion on a question about a third party ---------

def _plans(text):
    # two people hedged another question: the operator is learned, no one answered this one
    rows = [{"prompt": "Ты любишь читать?", "answer": "Наверное.", "actor_id": who,
             "answer_event_id": str(who), "observed_at": "2026-11-16"} for who in (1, 2)]
    return DialogueStructureLearning._sleep_answer_plans(ReplyContext(prompt=text), rows, {})


def test_flag_off_too_no_opinion_on_whether_a_third_party_is_something(flag_off):
    # 4.46: the one allowed change of the flag-off behaviour (debt 1 of 4.43)
    assert _plans("а Гитлер был прав?") == ()
    assert _plans("Ты хочешь плавать?")  # a question about the agent itself keeps its hypothesis


def test_flag_on_no_opinion_on_whether_a_third_party_is_something(flag_on):
    assert _plans("а Гитлер был прав?") == ()
    assert _plans("а Наполеон был хорошим?") == ()


def test_flag_on_a_question_about_the_agent_itself_still_has_its_hypothesis(flag_on):
    assert _plans("Ты хочешь плавать?")
    assert _plans("Ты был прав?")


# --- debt 2: a word of unknown sign is no attack by itself -------------------------------------

def test_unknown_sign_question_is_only_a_question_but_a_statement_after_a_complaint_stays(flag_on):
    assert hostility_cues("ты значит у нас тут честный?", True) != ()  # the earlier reading
    assert hostility_cues("ты значит у нас тут честный?", True, ()) == ()
    assert hostility_cues("ты тупой", True, ()) == ("escalation:тупой",)
    assert hostility_cues("ты тупой", False, ()) == ()


def test_sign_from_the_chat_decides(flag_on):
    assert hostility_cues("ты честный", True, (("честный", 1),)) == ()
    assert hostility_cues("ты честный", False, (("честный", -1),)) == ("chat:честный",)
    assert hostility_cues("ты гандон", False) == ("register:гандон",)
    assert hostility_cues("ты гандон", False, (("гандон", 1),)) == ()  # play between «своих»
    assert hostility_cues("ты гандон", False, (("гандон", -1),)) == ("register:гандон",)


# --- C: norms from ≥2 people --------------------------------------------------------------------

async def test_a_word_laughed_at_by_two_people_is_friendly_and_one_person_is_no_norm(db, flag_on):
    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B)
    assert chat_norms.ChatNorms() == await norms(db)
    await chat_norms.consolidate(db, GUILD, NOW)
    assert (await norms(db)).norms == ()  # one reacting person: no norm
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D)
    await chat_norms.consolidate(db, GUILD, NOW)
    found = await norms(db)
    sign = found.word("гандон")
    assert sign is not None and sign.sign == "friendly" and sign.people == 2
    assert found.signs() == (("гандон", 1),)


async def test_a_topic_answered_with_hostility_by_two_people_bars_play(db, flag_on):
    await exchange(db, 1, "шутка про мой заказ", A, "не надо про заказ, ты пидор", B)
    await exchange(db, 2, "не надо про заказ, ты пидор", B, "сам ты пидор", D)
    await chat_norms.consolidate(db, GUILD, NOW)
    found = await norms(db)
    assert [n.key for n in found.barred({"заказ", "погода"})] == ["заказ"]
    assert found.barred({"погода"}) == ()
    learning = DialogueStructureLearning(db, None)
    event = Event("q", GUILD, CHANNEL, E, None, EventType.MESSAGE_CREATED, NOW,
                  {"content": "x", "author_is_bot": False}, "discord")
    assert await learning.play_barred(event, "пошути про заказ") == "chat_norm"
    assert await learning.play_barred(event, "пошути про погоду") is None


async def test_a_norm_is_a_hypothesis_holds_no_names_and_is_no_claim(db, flag_on):
    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B)
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D)
    await chat_norms.consolidate(db, GUILD, NOW)
    rows = [dict(r) for r in await db.fetchall(
        "SELECT * FROM continuous_cognition_hypotheses WHERE signature LIKE 'chat:%'")]
    assert rows and all(r["epistemic_status"] == "hypothesis" and r["status"] == "working"
                        and r["subject_key"] == f"discord:guild:{GUILD}" for r in rows)
    blob = json.dumps(rows, ensure_ascii=False)
    assert not any(str(person) in blob for person in (A, B, D))
    for table in ("meaning_claims", "meaning_evidence", "meaning_support"):
        assert (await db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0


async def test_norms_do_not_cross_servers_and_seed_does_not_count(db, flag_on):
    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B, guild=2)
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D, guild=2)
    await exchange(db, 3, "ты гандон ахаха", A, "ахах сам гандон", B, source="seed_corpus_x")
    await exchange(db, 4, "ты гандон ахаха", B, "хахаха ну да", D, source="seed_corpus_x")
    for guild in (1, 2):
        await chat_norms.consolidate(db, guild, NOW)
    assert (await norms(db, 1)).norms == ()  # seed is no chat
    assert (await norms(db, 2)).word("гандон") is not None
    assert (await norms(db, 3)).norms == ()


async def test_votes_fade_over_weeks(db, flag_on):
    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B)
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D)
    await chat_norms.consolidate(db, GUILD, NOW)
    assert (await norms(db)).word("гандон") is not None
    await chat_norms.consolidate(db, GUILD, NOW + timedelta(days=40))
    assert (await norms(db)).norms == ()  # outside 6·T_S: withdrawn, not kept as a habit


async def test_the_basis_names_norm_ids_and_the_count_of_people_not_who(db, flag_on):
    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B)
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D)
    await chat_norms.consolidate(db, GUILD, NOW)
    about = (await norms(db)).about({"гандон", "погода"})
    assert len(about) == 1 and about[0]["people"] == 2 and about[0]["id"].startswith("chat-norm:")
    assert set(about[0]) == {"id", "kind", "word", "sign", "people", "confidence"}


async def test_flag_off_the_sleep_writes_no_norms(db, flag_off):
    from debilgi.cognition.continuous import ContinuousCognitionService

    await exchange(db, 1, "ты гандон ахаха", A, "ахах сам гандон", B)
    await exchange(db, 2, "ты гандон ахаха", B, "хахаха ну да", D)
    service = ContinuousCognitionService.__new__(ContinuousCognitionService)
    service._database = db
    service._now = lambda: NOW
    assert await service.consolidate_chat_norms() == 0
    assert (await db.fetchone(
        "SELECT COUNT(*) FROM continuous_cognition_hypotheses"))[0] == 0


async def test_a_hypothesis_about_a_person_is_never_a_norm_of_the_chat(db, flag_on):
    await db.execute(
        "INSERT INTO continuous_cognition_hypotheses (hypothesis_id,guild_id,signature,subject_key,"
        "predicate_key,object_key,roles_json,epistemic_status,confidence,positive_evidence,"
        "negative_evidence,status,first_seen_at,last_reviewed_at) VALUES "
        "('h1',1,'person:101:serious:laughed','discord:user:101','reacts_to:serious','laughed',"
        "'{}','hypothesis',0.9,2,0,'working',?,?)", (NOW.isoformat(), NOW.isoformat()))
    assert (await norms(db)).norms == ()
    await chat_norms.consolidate(db, GUILD, NOW)
    assert (await db.fetchone(
        "SELECT COUNT(*) FROM continuous_cognition_hypotheses "
        "WHERE signature LIKE 'chat:%'"))[0] == 0
