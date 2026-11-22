"""Stage 4.32 (DEBILGI_THOUGHT_CONTENT): own doings and the open request.

Flag off = the previous behaviour. Flag on: «что делаешь?» is answered with a marked guess from the
agent's own record without a request to guess, a fragment reaction («серьёзно??») does not take the
focus of a later explicit request, and a guess never becomes a fact.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.cognition.dialogue_tasks import DialogueTaskRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import Known, Unknown, activity_thought
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.request_focus import focus_request, is_fragment_reaction
from debilgi.language.semantic_graph import MeaningRecord
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 21, 11, 20, tzinfo=UTC)
BOT = 777
GUILD, CHANNEL, USER = 1, 10, 100
TABLES = ("meaning_claims", "meaning_evidence", "meaning_support",
          "continuous_cognition_hypotheses")


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "content432.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def ev(n, text="x", *, actor=USER, guild=GUILD, at=NOW, bot=False):
    return Event(f"e:{n}", guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text, "author_is_bot": bot,
                  "bot_user_id": BOT, "bot_mentioned": True}, "discord")


async def human_turn(db, n, text, *, actor, guild=GUILD):
    at = NOW - timedelta(hours=3)
    await EventRepository(db).append(ev(n, text, actor=actor, guild=guild, at=at))
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", guild, CHANNEL, n, actor, text, at.isoformat()))


async def teach_hedge(db, guild=GUILD):
    await human_turn(db, 1, "наверное, он спит", actor=201, guild=guild)
    await human_turn(db, 2, "наверное, да", actor=202, guild=guild)


async def act(db, n, predicate, target, *, ago, status="completed"):
    event = ev(3000 + n)
    await EventRepository(db).append(event)
    await PropositionMemory.for_meaning(db).record_agent_activity(
        event, source_key=f"receipt:{n}", operation="send", status=status,
        thought=activity_thought(predicate, target), occurred_at=NOW - ago, receipt_id=n)


def limit_planning():
    return SimpleNamespace(proposals=(SimpleNamespace(
        plan=SimpleNamespace(knowledge_limit=object())),))


async def ask(db, text, *, owner_text=None, guild=GUILD):
    event = Event("e:900", guild, CHANNEL, USER, None, EventType.MESSAGE_CREATED, NOW,
                  {"message_id": 900, "content": owner_text or text, "author_is_bot": False,
                   "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    claim_result = await PropositionMemory.for_meaning(db).retrieve_claims(
        Event(event.event_id, guild, CHANNEL, USER, None, EventType.MESSAGE_CREATED, NOW,
              {**event.payload, "content": text}, "discord"), text)
    context = SimpleNamespace(structure=analyze_structured_utterance(text), prompt=text,
                              claim_result=claim_result)
    return await DialogueStructureLearning(db, None).supposition_plans(
        event, context, limit_planning()), claim_result


async def counts(db):
    return {t: (await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in TABLES}


# ---------------------------------------------------------------- (a) own doings
async def test_flag_off_a_bare_question_about_the_doing_keeps_the_limit(db):
    await teach_hedge(db)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    plans, result = await ask(db, "что ты делаешь?")
    assert plans == () and isinstance(result, Unknown)


async def test_flag_on_the_doing_is_a_marked_guess_without_a_request_to_guess(db, flag_on):
    await teach_hedge(db)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    before = await counts(db)
    plans, result = await ask(db, "что ты делаешь?")
    assert isinstance(result, Unknown)  # the past doing is no present Known claim
    assert plans and plans[0].operation == "supposition"
    plan = plans[0].plan
    assert dict(plan.hypothesis)["kind"] == "own_evidence_opinion"
    assert dict(plan.hypothesis)["operator"] == "наверное"
    assert plan.claim.predicate.lemma == "отправить"
    assert plans[0].premises[0].kind == "agent_observed"
    # a guess is no fact: nothing is written, and the same question is still Unknown
    assert await counts(db) == before
    _, again = await ask(db, "что ты делаешь?")
    assert isinstance(again, Unknown)
    # no randomness: the same record gives the same plan
    repeated, _ = await ask(db, "что ты делаешь?")
    assert [p.plan.claim for p in repeated] == [p.plan.claim for p in plans]


async def test_flag_on_without_ground_hedge_or_in_the_past_there_is_no_guess(db, flag_on):
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    assert (await ask(db, "что ты делаешь?"))[0] == ()  # no observed hedge
    await teach_hedge(db)
    assert (await ask(db, "что ты делал вчера?"))[0] == ()  # not the present
    assert (await ask(db, "что ты любишь делать?"))[0] == ()  # not the doing
    await db.execute("UPDATE meaning_claims SET source_time='2026-11-21T09:00:00+00:00'")
    assert (await ask(db, "что ты делаешь?"))[0] == ()  # nothing in the last ten minutes


async def test_flag_on_other_server_hedge_is_not_learned(db, flag_on):
    await teach_hedge(db, guild=2)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    assert (await ask(db, "что ты делаешь?"))[0] == ()


@pytest.mark.parametrize("flag", ["0", "1"])
async def test_an_operation_overlapping_the_question_is_a_known_claim(db, monkeypatch, flag):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", flag)
    await act(db, 1, "обрабатывать", "сообщение", ago=timedelta(seconds=20), status="active")
    plans, result = await ask(db, "что ты делаешь?")
    assert isinstance(result, Known) and plans == ()


# ---------------------------------------------------------------- (b) the focus of a request
@pytest.mark.parametrize("text", ["серьёзно??", "правда?", "серьёзно?", "ну да?"])
def test_a_reaction_without_predicate_or_question_word_is_a_fragment(text):
    assert is_fragment_reaction(text)


@pytest.mark.parametrize("text", [
    "что делаешь?", "почему?", "где?", "ты бегаешь?", "кто быстрее? медведь или муравей?",
    "медведь или муравей?"])
def test_a_predicate_a_question_word_or_a_choice_is_a_request(text):
    assert not is_fragment_reaction(text)


TURNS = ("user: ку бро", "agent: привет!!", "user: что делаешь?", "agent: я не знаю, что я делаю.",
         "user: серьёзно??")


def test_flag_off_the_fragment_takes_the_focus(monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    focus = focus_request("а ты предположи", TURNS)
    assert focus.request_text == "серьёзно??" and focus.reason == "answer_control_keeps_request"


def test_flag_on_the_command_keeps_the_open_request_before_the_fragment(flag_on):
    focus = focus_request("а ты предположи", TURNS)
    assert focus.request_text == "что делаешь?" and focus.reason == "answer_control_keeps_request"
    # a real earlier question still stops at the latest request
    later = (*TURNS, "agent: x", "user: ты бегаешь?")
    assert focus_request("а ты предположи", later).request_text == "ты бегаешь?"


async def test_the_task_of_a_fragment_does_not_capture_the_command(db, monkeypatch):
    repository = DialogueTaskRepository(db)
    asked = []
    for n, (text, minute) in enumerate((("что ты делаешь?", 3), ("серьёзно??", 2))):
        event = ev(100 + n, text, at=NOW - timedelta(minutes=minute))
        await EventRepository(db).append(event)
        await repository.deliberate(event, prompt=text, consultation_messages=tuple(asked),
                                    dialogue_state=None)
        asked.append(f"user: {text}")
    async def request(flag, n):
        monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", flag)
        follow = ev(n, "а ты предположи", at=NOW)
        await EventRepository(db).append(follow)
        situation = await repository.deliberate(
            follow, prompt="а ты предположи", consultation_messages=tuple(asked),
            dialogue_state=None)
        return situation.active_task.objective.request_text

    assert await request("0", 200) == "серьёзно??"
    assert await request("1", 201) == "что ты делаешь?"


# ---------------------------------------------------------------- (c) a question is not a report
async def legacy_fact(db, n, text):
    """The legacy record an older reader stored for a question without «?»."""
    from dataclasses import asdict

    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )
    thought = ClauseThought(
        ConceptLexeme("делать", "делаешь", "verb"),
        (SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
         SemanticDependency("direct_object", ConceptArgument(ConceptLexeme("что", "что", "noun")))),
        confidence=1.0)
    record = MeaningRecord(f"m:{n}", f"e:{n}", text, "fact", (), (thought,))
    await db.execute(
        "INSERT INTO language_meaning_records VALUES (?,?,?,?,?,?,?,?)",
        (record.meaning_id, f"e:{n}", GUILD, CHANNEL, USER, "fact",
         json.dumps(asdict(record), ensure_ascii=False), NOW.isoformat()))


@pytest.mark.parametrize(("flag", "reports"), [("0", 1), ("1", 0)])
async def test_a_legacy_fact_that_is_a_question_is_not_projected_as_a_report(
        db, monkeypatch, flag, reports):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", flag)
    event = ev(50, "че делаешь")
    await EventRepository(db).append(event)
    await legacy_fact(db, 50, "че делаешь")
    await PropositionMemory.for_meaning(db).observe(event)
    row = await db.fetchone(
        "SELECT COUNT(*) FROM meaning_claims WHERE source_event_id='e:50' "
        "AND source_kind IN ('other_report','self_report')")
    assert row[0] == reports


async def test_a_question_without_a_mark_gives_no_claim_from_the_current_reader(db, flag_on):
    event = ev(51, f"<@{BOT}> че делаешь")
    await EventRepository(db).append(event)
    await PropositionMemory.for_meaning(db).observe(event)
    row = await db.fetchone("SELECT COUNT(*) FROM meaning_claims WHERE source_event_id='e:51'")
    assert row[0] == 0
