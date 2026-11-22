"""Stage 4.24: play about an embedded verb, the elided question, «почему?» after an own limit.

Words that were not in the live dialogue: врач, пилот, художник, котёнок; verbs «стать», «быть».
"""

import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.communication import ResponseComposer, plan_issues
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.generative_planner import PlanningResult
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.word_relations import WordRelation
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 20, 10, 9, tzinfo=UTC)
BOT = 777
GUILD, CHANNEL, USER = 1, 10, 100


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "nested.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


async def human_turn(db, n, text, *, actor, ago=timedelta(hours=3)):
    at = NOW - ago
    event = Event(f"e:{n}", GUILD, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
                  {"message_id": zlib.crc32(str(n).encode()), "content": text,
                   "author_is_bot": False, "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    await EventRepository(db).append(event)
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", GUILD, CHANNEL, n, actor, text, at.isoformat()))


async def corpus(db, text, authors=(401, 402), base=100):
    for i, author in enumerate(authors):
        await human_turn(db, base + i, text, actor=author)


def relation(governor, dependent, name="obj", case="instrumental", count=3):
    return WordRelation(governor, dependent, name, "noun", case, count)


async def ask(db, text, relations, *, earlier=()):
    event = Event("e:900", GUILD, CHANNEL, USER, None, EventType.MESSAGE_CREATED, NOW,
                  {"message_id": 900, "content": text, "author_is_bot": False,
                   "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    context = SimpleNamespace(
        structure=analyze_structured_utterance(text), prompt=text, claim_result=None,
        word_relations=tuple(relations), personality=None,
        consultation_messages=tuple(earlier))
    plans = await DialogueStructureLearning(db, None).fantasy_plans(
        event, context, PlanningResult(None, (), (), ()))
    return plans, context


def said(plans, context):
    assert plans
    return [s.text for s in ResponseComposer(context.structure).compose(plans[0].plan)]


async def test_material_of_the_play_belongs_to_the_embedded_verb_and_takes_its_case(db):
    await corpus(db, "я хочу быть врачом")
    relations = (relation("быть", "врач", case="nominative"),)  # the corpus said «врач» bare
    plans, context = await ask(db, "а хотел бы быть?", relations)
    claim = plans[0].plan.claim
    attached = [d for d in claim.dependencies if d.head]
    assert [(d.role, d.argument.head.lemma, d.head) for d in attached] == [
        ("direct_object", "врач", "action_complement")]
    assert attached[0].argument.grammatical_case is None  # nominative of the corpus is dropped
    assert not plan_issues(plans[0].plan)
    assert said(plans, context)[0].startswith("Я хотел бы быть врачом")


async def test_another_embedded_verb_and_noun(db):
    await corpus(db, "мы хотим стать художниками", base=200)
    relations = (relation("стать", "художник"),)
    plans, context = await ask(db, "а хотел бы стать?", relations)
    assert said(plans, context)[0].startswith("Я хотел бы стать художником")


async def test_a_short_verb_has_no_prefixed_family(db):
    # «забыть пароль» is not «быть …»: the stem of «быть» is too short for a family.
    await corpus(db, "я забыл пароль", base=300)
    plans, context = await ask(db, "а хотел бы быть?", (relation("забыть", "пароль",
                                                                case="accusative"),))
    lemmas = {d.argument.head.lemma for d in plans[0].plan.claim.dependencies} if plans else set()
    assert "пароль" not in lemmas


async def test_the_elided_question_takes_the_complement_of_the_same_verb_from_the_earlier_turn(db):
    plans, context = await ask(
        db, "а хотел бы быть?", (),
        earlier=("user: ты хочешь быть как пилот?", "agent: Я не знаю, хочу ли я."))
    claim = plans[0].plan.claim
    assert [(d.role, d.argument.head.lemma, d.head) for d in claim.dependencies if d.head] == [
        ("comparison", "пилот", "action_complement")]
    assert said(plans, context)[0].startswith("Я хотел бы быть как пилот")


async def test_what_the_agent_borrows_from_the_question_must_be_fit_for_play(db):
    # A name (or a word of anger) of the earlier question is the person's, not the agent's play.
    plans, context = await ask(
        db, "а хотел бы быть?", (), earlier=("user: ты хочешь быть как Гитлер?",))
    claim = plans[0].plan.claim
    assert not [d for d in claim.dependencies if d.head]
    assert "гитлер" not in said(plans, context)[0].casefold()


async def test_the_agents_own_words_are_never_the_earlier_turn(db):
    plans, context = await ask(
        db, "а хотел бы быть?", (), earlier=("agent: ты хочешь быть как пилот?",))
    assert not [d for d in plans[0].plan.claim.dependencies if d.head]


async def test_play_goes_on_with_the_same_embedded_verb(db):
    await corpus(db, "я хочу быть врачом")
    await corpus(db, "я хочу быть пилотом", base=150)
    relations = (relation("быть", "врач", count=4), relation("быть", "пилот", count=4))
    first, context = await ask(db, "а хотел бы быть?", relations)
    event = Event("e:901", GUILD, CHANNEL, USER, None, EventType.MESSAGE_CREATED, NOW,
                  {"message_id": 901, "content": "ещё", "author_is_bot": False,
                   "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    claim = first[0].plan.claim
    more = await DialogueStructureLearning(db, None).fantasy_plans(
        event, context, PlanningResult(None, (), (), ()), continuation=claim,
        said={d.argument.head.lemma for d in claim.dependencies})
    if more:  # another thing to be, still «быть <…>», never «хотел бы <…>»
        texts = [s.text for s in ResponseComposer(context.structure).compose(more[0].plan)]
        assert "быть" in texts[0] and "хотел бы" in texts[0]
