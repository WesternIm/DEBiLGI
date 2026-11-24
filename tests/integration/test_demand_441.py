"""Stage 4.41 (Э4): a demand pressed again is not closed; the reaction teaches the choice.

Flag ``DEBILGI_THOUGHT_CONTENT`` only. The experience is the frequency of an outcome that left the
demand open (for this operation, in this context, for this person), never a hand-set reward.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository, unclosed_share
from debilgi.language.generative_planner import DialogueObjective
from debilgi.language.semantic_graph import parse_proposition
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, USER = 1, 10, 100


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "demand441.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def message(n, text, *, at=NOW, bot=False):
    return Event(f"e:{n}", GUILD, CHANNEL, USER, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": bot,
                  "bot_user_id": 7, "bot_mentioned": True}, "discord")


async def turn(db, n, prompt, operation, outcomes, *, ago):
    """The agent's delivered turn to ``prompt`` and what the person did next."""
    at = NOW - ago
    await EventRepository(db).append(message(f"p{n}", prompt, at=at))
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,status,response_message_id,"
        "sent_at,created_at,response_text) VALUES (?,'p',?,?,?,?,?,?,?,?,?,0,0,'{}','{}','sent',"
        "?,?,?,'ответ')",
        (f"i:{n}", f"e:p{n}", GUILD, CHANNEL, USER, json.dumps(["polar", None]), f"b:{operation}",
         operation, n, json.dumps({"claim": {"clause_kind": "statement"}}),
         zlib.crc32(n.encode()), at.isoformat(), at.isoformat()))
    for k, outcome in enumerate(outcomes):
        await EventRepository(db).append(message(f"c{n}{k}", "x", at=at + timedelta(seconds=5)))
        await db.execute(
            "INSERT INTO language_meaning_consequences (intention_id,event_id,outcome_type,"
            "observed_at) VALUES (?,?,?,?)",
            (f"i:{n}", f"e:c{n}{k}", outcome, (at + timedelta(seconds=5)).isoformat()))


def test_the_share_is_a_frequency_that_never_reaches_one():
    rows = [{"outcome_type": "follow_up_question", "intention_id": "a"}]
    assert unclosed_share(rows) == (0.5, ("a",))
    rows += [{"outcome_type": "follow_up_question", "intention_id": "b"}] * 3
    share, ids = unclosed_share(rows)
    assert 0.5 < share < 1.0 and ids == ("a", "b")
    assert unclosed_share([{"outcome_type": "conversation_continued",
                            "intention_id": "a"}]) == (0.0, ())
    assert unclosed_share([]) == (0.0, ())


async def test_what_the_person_pressed_again_is_read_per_operation(db, flag_on):
    await turn(db, "1", "ты любишь спать?", "compose_reaction:acknowledge",
               ["follow_up_question"], ago=timedelta(hours=30))
    await turn(db, "2", "ты любишь рыбалку?", "express_query_unknown",
               ["conversation_continued"], ago=timedelta(hours=29))
    repository = MeaningIntentRepository(db)
    objective = DialogueObjective("polar", None, None, (), ())
    shares = await repository.unclosed_by_operation(message("now", "ты любишь спать?"), objective)
    # (4.42: a pressing 30 hours old weighs 2^(-age/T_S), not 1)
    from debilgi.experience.meaning_intents import decay_weight, social_half_life

    w = decay_weight(30 * 3600.0, social_half_life())
    assert shares == {"compose_reaction:acknowledge": pytest.approx(w / (w + 1.0), rel=1e-3)}


async def test_without_the_flag_nothing_is_read(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    await turn(db, "1", "ты любишь спать?", "compose_reaction:acknowledge",
               ["follow_up_question"], ago=timedelta(hours=30))
    objective = DialogueObjective("polar", None, None, (), ())
    assert await MeaningIntentRepository(db).unclosed_by_operation(
        message("now", "ты любишь спать?"), objective) == {}


@pytest.mark.parametrize(("content", "pressed"), [
    ("так да или нет", False),  # (only the question mark made this one a follow-up before)
    ("<@7> ты любишь спать", True),  # the same demand repeated
    ("почему", True),  # a repair of the last turn, with no question mark
    ("какое", True),
    ("поеду на рыбалку завтра", False),
])
async def test_a_repeat_or_a_repair_leaves_the_demand_open(db, flag_on, content, pressed):
    await EventRepository(db).append(message("p", "<@7> ты любишь спать?"))
    prior = {"prompt_event_id": "e:p", "response_text": "Я не знаю ответа."}
    reading = await MeaningIntentRepository(db)._presses_demand(message("r", content), prior)
    if content == "так да или нет":
        return  # decided by the older «?»/«или» rule, not here
    assert reading is pressed


async def test_a_pressing_is_not_read_without_the_flag(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    await EventRepository(db).append(message("p", "ты любишь спать?"))
    prior = {"prompt_event_id": "e:p", "response_text": "Я не знаю ответа."}
    assert not await MeaningIntentRepository(db)._presses_demand(
        message("r", "ты любишь спать"), prior)


def test_the_object_of_a_retold_question_is_kept_only_when_asked_to(flag_on, monkeypatch):
    def read(**options):
        parse_proposition.cache_clear()
        thought = parse_proposition("ты любишь спать?", allow_question=True, **options)
        return thought and [(d.role, d.argument.head.lemma) for d in thought.dependencies]

    assert read() is None
    assert read(action_object=True) == [("subject", "я"), ("action_complement", "спать")]
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "0")
    parse_proposition.cache_clear()  # (the reading is memoized by its arguments, not by the flag)
    assert read(action_object=True) is None


def reply(n, text, seconds):
    return Event(f"r:{n}", GUILD, CHANNEL, USER, 200, EventType.MESSAGE_REPLIED,
                 NOW + timedelta(seconds=seconds),
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
                  "bot_mentioned": True}, "test")


async def pressed_demand(db):
    from debilgi.cognition.dialogue_tasks import DialogueTaskRepository

    repository = DialogueTaskRepository(db)
    first = reply("1", "скажи да", 0)
    assert await EventRepository(db).append(first)
    task = (await repository.deliberate(first, prompt="скажи да", consultation_messages=(),
                                        dialogue_state=None)).active_task
    assert await repository.record_attempt(first, task_id=task.task_id, response_message_id=501,
                                           text="Я сказал, что ты хищник.",
                                           created_at=NOW + timedelta(seconds=1))
    again = reply("2", "А да не скажешь?", 10)
    assert await EventRepository(db).append(again)
    situation = await repository.deliberate(
        again, prompt="А да не скажешь?",
        consultation_messages=("user: скажи да", "agent: Я сказал, что ты хищник."),
        dialogue_state=None)
    return task, situation


async def test_the_pressed_demand_is_held_by_the_turn_it_started_from(db, flag_on):
    task, situation = await pressed_demand(db)
    events = await db.fetchall(
        "SELECT task_id FROM dialogue_task_events WHERE relation='pressed'")
    assert [row["task_id"] for row in events] == [task.task_id]
    assert 0.9 < situation.pressed <= 1.0  # one fresh pressing, decayed by R's half-life
    assert dict(situation.features)["task_pressed"] == situation.pressed


async def test_without_the_flag_no_pressing_is_recorded(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    _, situation = await pressed_demand(db)
    assert not await db.fetchall(
        "SELECT 1 FROM dialogue_task_events WHERE relation='pressed'")
    assert situation.pressed == 0.0 and "task_pressed" not in dict(situation.features)
