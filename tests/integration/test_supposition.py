"""Stage 4.21: a marked supposition from the agent's own record instead of a bare limit."""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.cognition.dialogue_tasks import DialogueTaskRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.canonical_claims import activity_thought
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.semantic_graph import conditional_proposition
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 19, 10, 9, tzinfo=UTC)
BOT = 777
GUILD, CHANNEL, USER = 1, 10, 100


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "supposition.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def ev(n, text="x", *, actor=USER, guild=GUILD, channel=CHANNEL, at=NOW, bot=False):
    return Event(f"e:{n}", guild, channel, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text, "author_is_bot": bot,
                  "bot_user_id": BOT, "bot_mentioned": True}, "discord")


async def human_turn(db, n, text, *, actor, ago=timedelta(hours=3), guild=GUILD):
    at = NOW - ago
    await EventRepository(db).append(ev(n, text, actor=actor, guild=guild, at=at))
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", guild, CHANNEL, n, actor, text, at.isoformat()))


async def teach_hedge(db, guild=GUILD):
    await human_turn(db, 1, "наверное, он спит", actor=201, guild=guild)
    await human_turn(db, 2, "наверное, да", actor=202, guild=guild)


async def act(db, n, predicate, target, *, ago):
    event = ev(3000 + n)
    await EventRepository(db).append(event)
    await PropositionMemory.for_meaning(db).record_agent_activity(
        event, source_key=f"receipt:{n}", operation="send",
        thought=activity_thought(predicate, target), occurred_at=NOW - ago, receipt_id=n)


async def gap(db, n, prompt, *, actor=USER, guild=GUILD):
    await EventRepository(db).append(ev(f"g{n}", prompt, actor=actor, guild=guild))
    details = {"actor_id": actor, "channel_id": CHANNEL, "prompt": prompt, "kinds": ["unknown"],
               "meaning_scope": '["meaning","k","что",true]'}
    await db.execute(
        "INSERT INTO continuous_cognition_gaps (gap_id,guild_id,signature,question_kind,"
        "answer_role,occurrence_count,status,source_event_id,first_seen_at,last_seen_at,"
        "details_json) VALUES (?,?,?,'unknown','dialogue_response',1,'open',?,?,?,?)",
        (f"g:{n}", guild, f"sleep:{n}", f"e:g{n}", NOW.isoformat(), NOW.isoformat(),
         json.dumps(details, ensure_ascii=False)))


def limit_planning():
    return SimpleNamespace(proposals=(SimpleNamespace(
        plan=SimpleNamespace(knowledge_limit=object())),))


async def ask(db, text, *, owner_text=None, actor=USER, guild=GUILD):
    """The plans for ``text`` (the focused request) when the turn itself says ``owner_text``."""
    event = ev(900, owner_text or text, actor=actor, guild=guild, at=NOW)
    event = Event(event.event_id, guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, NOW,
                  {"message_id": 900, "content": owner_text or text, "author_is_bot": False,
                   "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    structure = analyze_structured_utterance(text)
    claim_result = await PropositionMemory.for_meaning(db).retrieve_claims(
        Event(event.event_id, guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, NOW,
              {**event.payload, "content": text}, "discord"), text)
    context = SimpleNamespace(structure=structure, prompt=text, claim_result=claim_result)
    return await DialogueStructureLearning(db, None).supposition_plans(
        event, context, limit_planning())


async def claims_count(db):
    return (await db.fetchone("SELECT COUNT(*) FROM meaning_claims"))[0]


async def test_asked_to_suppose_the_agent_offers_a_marked_guess_from_its_own_action(db):
    await teach_hedge(db)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    before = await claims_count(db)
    plans = await ask(db, "что ты делаешь?", owner_text="а если предположить")
    assert plans and plans[0].operation == "supposition"
    plan = plans[0].plan
    assert dict(plan.hypothesis)["operator"] == "наверное"
    assert dict(plan.hypothesis)["operator_source"] in plan.claim.evidence
    assert plans[0].premises[0].kind == "agent_observed"
    assert plan.claim.predicate.lemma == "отправить"
    assert await claims_count(db) == before  # a supposition is no fact


async def test_without_a_request_for_a_guess_the_limit_stays(db):
    await teach_hedge(db)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    assert await ask(db, "что ты делаешь?") == ()


async def test_no_own_action_in_ten_minutes_means_no_guess(db):
    await teach_hedge(db)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=30))
    assert await ask(db, "что ты делаешь?", owner_text="а если предположить") == ()


async def test_no_observed_hedge_means_no_unmarked_guess(db):
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    assert await ask(db, "что ты делаешь?", owner_text="а если предположить") == ()


async def test_hedge_of_another_server_is_not_learned(db):
    await teach_hedge(db, guild=2)
    await act(db, 1, "отправить", "сообщение", ago=timedelta(minutes=3))
    assert await ask(db, "что ты делаешь?", owner_text="а если предположить") == ()


async def test_asked_what_it_thinks_the_agent_reports_its_open_question_of_this_person(db):
    await gap(db, 1, "а ты умеешь летать ?")
    await gap(db, 2, "а ты умеешь плавать ?", actor=USER + 1)  # another person's question
    await gap(db, 3, "а ты умеешь прыгать ?", guild=2)  # another server's
    plans = await ask(db, "что ты думаешь?")
    assert [p.plan.knowledge_limit.question.predicate.lemma for p in plans] == ["уметь"]
    assert plans[0].plan.claim.predicate.lemma == "думать"
    assert plans[0].premises[0].kind == "open_question"
    flying = plans[0].plan.knowledge_limit.question.dependencies
    assert "летать" in {d.argument.head.lemma for d in flying}


async def test_nothing_open_for_this_person_means_no_report(db):
    await gap(db, 2, "а ты умеешь плавать ?", actor=USER + 1)
    assert await ask(db, "что ты думаешь?") == ()


async def test_a_conditional_frame_licenses_a_guess_but_needs_a_ground(db):
    await teach_hedge(db)
    assert await ask(db, "а если хотел, то полетел бы?") == ()


def test_conditional_reader_gives_the_consequent_and_the_mood():
    read = conditional_proposition("а если хотел, то полетел бы?")
    assert read.consequent.predicate.lemma == "полететь" and read.irrealis is True
    plain = conditional_proposition("а если ты хочешь, то полетишь?")
    assert plain.consequent.predicate.lemma == "полететь" and plain.irrealis is False
    both = conditional_proposition("но если пойдёт дождь, я возьму зонт")
    assert both.condition.predicate.lemma == "пойти"
    assert both.consequent.predicate.lemma == "взять"
    assert conditional_proposition("если что, пиши") is None
    assert conditional_proposition("а если предположить") is None


@pytest.mark.parametrize("text", [
    "а если хотел, то полетел бы?", "а если ты хочешь, то полетишь?",
    "но если пойдёт дождь, я возьму зонт"])
def test_leading_conjunction_does_not_hide_the_condition_marker(text):
    structure = analyze_structured_utterance(text)
    assert len(structure.clauses) == 2
    assert [link.relation for link in structure.clause_links] == ["advcl:condition"]


async def test_fresh_question_outranks_stale_open_tasks(db):
    repository = DialogueTaskRepository(db)
    for n in range(13):
        stale = ev(100 + n, f"почему ты рад{n}?", at=NOW - timedelta(days=3, minutes=n))
        await EventRepository(db).append(stale)
        await repository.deliberate(stale, prompt=f"почему ты рад{n}?",
                                    consultation_messages=(), dialogue_state=None)
    await db.execute("UPDATE dialogue_tasks SET priority=1.0")
    fresh = ev(200, "что ты делаешь?", at=NOW - timedelta(minutes=1))
    await EventRepository(db).append(fresh)
    await repository.deliberate(fresh, prompt="что ты делаешь?", consultation_messages=(),
                                dialogue_state=None)
    follow = ev(201, "а если предположить", at=NOW)
    await EventRepository(db).append(follow)
    situation = await repository.deliberate(
        follow, prompt="а если предположить", consultation_messages=(), dialogue_state=None)
    assert situation.active_task.objective.request_text == "что ты делаешь?"
