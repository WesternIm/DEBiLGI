"""Stage 4.33 (DEBILGI_THOUGHT_CONTENT): boredom B and own doings older than ten minutes.

B is a function of the journal of the agent's delivered turns and the clock (no draw, no stored
number); the older deeds are a marked guess with an honest time word the people of the server use.
"""

import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import (
    TIRED_REPEATS,
    BoredomLedger,
    BoredomRow,
    MeaningIntentRepository,
    decay_weight,
    limit_scope,
)
from debilgi.language.canonical_claims import Unknown, activity_thought
from debilgi.language.dialogue_learning import (
    DialogueStructureLearning,
    delivered_report_plan,
    normalized_response,
)
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 21, 11, 20, tzinfo=UTC)
BOT = 777
GUILD, CHANNEL, USER = 1, 10, 100
TABLES = ("meaning_claims", "meaning_evidence", "meaning_support",
          "continuous_cognition_hypotheses")


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "content433.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def ev(n, text="x", *, actor=USER, guild=GUILD, at=NOW):
    return Event(f"e:{n}", guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text,
                  "author_is_bot": False, "bot_user_id": BOT, "bot_mentioned": True}, "discord")


async def human_turn(db, n, text, *, actor, guild=GUILD):
    at = NOW - timedelta(hours=3)
    await EventRepository(db).append(ev(n, text, actor=actor, guild=guild, at=at))
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", guild, CHANNEL, n, actor, text, at.isoformat()))


async def teach(db, *phrases, guild=GUILD, base=10):
    for i, phrase in enumerate(phrases):
        await human_turn(db, base + i, phrase, actor=200 + i, guild=guild)


HEDGE = ("наверное, он спит", "наверное, да")
TIME = ("я вчера спал", "а я вчера гулял", "недавно был дождь", "недавно так же было",
        "сегодня тепло", "а сегодня холодно")
OFTEN = ("я часто сплю", "обычно я часто гуляю")


async def act(db, n, predicate, target, *, ago):
    event = ev(3000 + n)
    await EventRepository(db).append(event)
    await PropositionMemory.for_meaning(db).record_agent_activity(
        event, source_key=f"receipt:{n}", operation="send", status="completed",
        thought=activity_thought(predicate, target), occurred_at=NOW - ago, receipt_id=n)


async def ask(db, text):
    event = ev(900, text)
    claim_result = await PropositionMemory.for_meaning(db).retrieve_claims(event, text)
    context = SimpleNamespace(structure=analyze_structured_utterance(text), prompt=text,
                              claim_result=claim_result)
    planning = SimpleNamespace(proposals=(SimpleNamespace(
        plan=SimpleNamespace(knowledge_limit=object())),))
    plans = await DialogueStructureLearning(db, None).supposition_plans(event, context, planning)
    return plans, claim_result


def words(plans):
    return [(p.plan.claim.predicate.lemma,
             [d.argument.head.lemma for d in p.plan.claim.dependencies if d.role == "manner"])
            for p in plans]


async def counts(db):
    return {t: (await db.fetchone(f"SELECT COUNT(*) FROM {t}"))[0] for t in TABLES}


# ---------------------------------------------------------------- B: a function of journal + clock
def row(n, age, key="k", op="express_query_unknown", terms=(), text="", scope=None):
    return BoredomRow(f"i:{n}", age, key, op, frozenset(terms), normalized_response(text),
                      f"receipt:1:{n}", text, scope)


def ledger(*rows):
    return BoredomLedger(tuple(rows), sensitivity=0.76, half_life=0.77 * 86400.0)


def test_boredom_grows_with_repeats_and_falls_with_the_clock():
    fresh = [row(i, 30.0 * (i + 1)) for i in range(3)]
    values = [ledger(*fresh[:n]).reading(behavior_key="k").value for n in range(4)]
    assert values[0] == 0.0 and values[0] < values[1] < values[2] < values[3] < 1.0
    # the same three repeats a day later: weaker, not forgotten (a state, not a blacklist)
    later = [row(i, 30.0 * (i + 1) + 86400.0) for i in range(3)]
    assert ledger(*later).reading(behavior_key="k").value < values[3]
    assert ledger(*later).reading(behavior_key="k").value > 0.0
    assert decay_weight(0, 100.0) == 1.0 and decay_weight(100.0, 100.0) == 0.5


def test_boredom_is_deterministic_and_carries_its_sources():
    one = ledger(row(1, 10.0), row(2, 20.0))
    assert one.reading(behavior_key="k") == ledger(row(1, 10.0), row(2, 20.0)).reading(
        behavior_key="k")
    assert one.reading(behavior_key="k").sources == ("i:1", "i:2")
    assert one.reading(behavior_key="other").value == 0.0


def test_the_same_material_under_another_meaning_is_a_smaller_repeat():
    old = ledger(row(1, 10.0, key="a", op="supposition", terms=("отправить", "сообщение")))
    same = old.reading(behavior_key="a", operation="supposition",
                       terms=("отправить", "сообщение"))
    other = old.reading(behavior_key="b", operation="supposition",
                        terms=("обработать", "сообщение"))
    unrelated = old.reading(behavior_key="c", operation="supposition", terms=("проснуться",))
    assert same.value > other.value > unrelated.value == 0.0


def test_a_not_knowing_of_the_same_question_is_one_thing_in_any_wording():
    scope = limit_scope({"predicate": "делать", "arguments": [["subject", "agent"]],
                         "snapshot_at": "t1", "exclude_event_id": "e1"})
    same_scope = limit_scope({"predicate": "делать", "arguments": [["subject", "agent"]],
                              "snapshot_at": "t2", "exclude_event_id": "e2"})
    assert scope == same_scope  # clocks and event ids are not the question
    spent = ledger(row(1, 10.0, key="x", scope=scope), row(2, 20.0, key="y", scope=scope))
    again = spent.reading(operation="express_query_unknown", scope=scope)
    assert again.repeats >= TIRED_REPEATS


def test_the_surface_repeat_is_read_from_the_wording():
    old = ledger(row(1, 10.0, key="a", text="я не знаю, что я делаю"))
    assert old.reading(text="Я не знаю, что я делаю.").surface > 0.0
    assert old.reading(text="Не знаю, что я делаю.").surface == 0.0


async def test_the_journal_is_read_only_with_the_flag_and_only_for_this_person(db, monkeypatch):
    repository = MeaningIntentRepository(db)
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    assert await repository.boredom_ledger(ev(1)) is None
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")
    empty = await repository.boredom_ledger(ev(1))
    assert empty is not None and empty.rows == ()
    assert 0.0 < empty.half_life < 86400.0 and empty.sensitivity == pytest.approx(0.76)


# ---------------------------------------------------------------- older deeds with a time
async def test_flag_off_there_are_no_older_deeds(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=26))
    assert (await ask(db, "что ты делаешь?"))[0] == ()


async def test_an_old_deed_is_a_marked_guess_with_a_time_word_people_use(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=26))
    before = await counts(db)
    plans, result = await ask(db, "что ты делаешь?")
    assert isinstance(result, Unknown)
    assert words(plans) == [("обработать", ["вчера"])]
    plan = plans[0].plan
    assert dict(plan.hypothesis)["kind"] == "own_evidence_opinion"
    assert dict(plan.hypothesis)["operator"] == "наверное"
    assert plans[0].premises[0].kind == "agent_observed"
    # a guess is no fact, and the same question is still Unknown
    assert await counts(db) == before
    assert isinstance(( await ask(db, "что ты делаешь?"))[1], Unknown)
    assert words((await ask(db, "что ты делаешь?"))[0]) == words(plans)  # no draw


async def test_the_day_word_follows_the_calendar_and_falls_back_to_recency(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=2))
    await act(db, 2, "изучить", "конструкцию", ago=timedelta(hours=26))
    await act(db, 3, "проснуться", None, ago=timedelta(hours=50))
    plans, _ = await ask(db, "что ты делаешь?")
    assert dict(words(plans)) == {"обработать": ["сегодня"], "изучить": ["вчера"],
                                  "проснуться": ["недавно"]}  # «позавчера» nobody here says


async def test_without_a_time_word_people_use_there_is_no_old_deed(db, flag_on):
    await teach(db, *HEDGE)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=26))
    assert (await ask(db, "что ты делаешь?"))[0] == ()
    await teach(db, "вчера так было", "а вчера иначе", guild=2, base=50)  # another server's words
    assert (await ask(db, "что ты делаешь?"))[0] == ()


async def test_a_deed_older_than_three_half_lives_is_not_recent(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(days=4))
    assert (await ask(db, "что ты делаешь?"))[0] == ()


async def test_equal_deeds_are_one_material_ranked_by_their_decayed_mass(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    for n in range(4):
        await act(db, n, "обработать", "сообщение", ago=timedelta(hours=26, minutes=n))
    await act(db, 9, "изучить", "конструкцию", ago=timedelta(hours=26))
    plans, _ = await ask(db, "что ты делаешь?")
    assert [w[0] for w in words(plans)] == ["обработать", "изучить"]


async def test_no_deed_material_for_questions_that_are_not_about_doing(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=26))
    for text in ("у тебя есть сознание?", "ты рад?", "какой у тебя цвет?", "ты любишь делать?"):
        plans, _ = await ask(db, text)
        assert plans == (), text


async def test_a_named_day_is_not_answered_from_other_days(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=26))
    assert (await ask(db, "что ты делал сегодня?"))[0] == ()


async def test_a_frequency_question_gets_the_habit_in_the_imperfective_people_use(db, flag_on):
    await teach(db, *HEDGE, *TIME, *OFTEN)
    for n in range(3):
        await act(db, n, "отправить", "сообщение", ago=timedelta(hours=5, minutes=n))
    asked = "а чем чаще всего занимаешься?"
    habit = [w for w in words((await ask(db, asked))[0]) if w[0] != "отправить"]
    assert habit == []  # no verb of this server pairs with the doing: no invented word
    await db.execute(
        "INSERT INTO language_generation_word_relations (guild_id,channel_id,governor,dependent,"
        "relation,dependent_pos,grammatical_case,count) VALUES (?,?,?,?,?,?,?,?)",
        (GUILD, CHANNEL, "отправлять", "сообщение", "obj", "noun", "accusative", 3))
    plans, _ = await ask(db, asked)
    assert words(plans)[0] == ("отправлять", ["часто"])
    assert dict(plans[0].plan.hypothesis)["kind"] == "own_evidence_opinion"


# ------------------------------------------------------- the report of the agent's own words
def test_a_report_does_not_nest_and_does_not_carry_the_sentence_mark(flag_on):
    goal = SimpleNamespace()
    inner = delivered_report_plan(goal, {"receipt_id": "receipt:1:1", "text": "Возможно, я рад."})
    first = inner.plan.claim.dependencies[1].argument.head.lemma
    assert first == "«Возможно, я рад»"
    nested = delivered_report_plan(goal, {
        "receipt_id": "receipt:1:2", "text": "Я сказал «Возможно, я рад.»."})
    assert nested.plan.claim.dependencies[1].argument.head.lemma == first


def test_a_report_keeps_its_old_form_without_the_flag(monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    plan = delivered_report_plan(SimpleNamespace(), {
        "receipt_id": "receipt:1:1", "text": "Возможно, я рад."})
    assert plan.plan.claim.dependencies[1].argument.head.lemma == "«Возможно, я рад.»"


def test_a_spent_wording_is_marked_for_the_arbiter(flag_on):
    from debilgi.discord.replies import SafeMentionReplyService as Service

    scope = limit_scope({"predicate": "делать"})
    spent = ledger(row(1, 10.0, scope=scope), row(2, 20.0, scope=scope))
    proposal = SimpleNamespace(
        operation="express_query_unknown", premises=(),
        plan=SimpleNamespace(
            knowledge_limit=SimpleNamespace(query={"predicate": "делать"}),
            claim=SimpleNamespace(predicate=SimpleNamespace(lemma="делать"), dependencies=())))
    reading = Service._boredom_of(spent, proposal, {"behavior_key": "z"}, "Я не знаю.")
    assert reading["spent"] is True
    fresh = Service._boredom_of(ledger(row(1, 10.0, scope=scope)), proposal,
                                {"behavior_key": "z"}, "Я не знаю.")
    assert fresh["spent"] is False


def test_a_report_already_made_is_spent_in_any_word_order(flag_on):
    from debilgi.discord.replies import SafeMentionReplyService as Service

    proposal = SimpleNamespace(
        operation="verify_reported_position", premises=(SimpleNamespace(kind="sent_turn"),),
        plan=SimpleNamespace(
            knowledge_limit=None,
            claim=SimpleNamespace(predicate=SimpleNamespace(lemma="сказать"), dependencies=())))
    made = ledger(row(1, 10.0, key="report", op="verify_reported_position",
                      text="Я сказал «не знаю»"))
    again = Service._boredom_of(made, proposal, {"behavior_key": "report"}, "Сказал я «не знаю».")
    assert again["spent"] is True  # the words differ, the report is the same
    other = Service._boredom_of(made, proposal, {"behavior_key": "another"}, "Я сказал «да».")
    assert other["spent"] is False


def test_only_a_clear_aspect_pair_gives_the_habit_verb():
    from debilgi.language.dialogue_learning import _same_root_pair

    assert _same_root_pair("отправлять", "отправить") and _same_root_pair("изучать", "изучить")
    assert _same_root_pair("делать", "сделать")
    # the permissive morphological reader pairs these; a habit with such a verb would be false
    assert not _same_root_pair("обращать", "обработать")
    assert not _same_root_pair("работать", "обработать")


async def test_a_false_friend_verb_of_the_server_is_not_the_habit(db, flag_on):
    await teach(db, *HEDGE, *TIME, *OFTEN)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(hours=5))
    await db.execute(
        "INSERT INTO language_generation_word_relations (guild_id,channel_id,governor,dependent,"
        "relation,dependent_pos,grammatical_case,count) VALUES (?,?,?,?,?,?,?,?)",
        (GUILD, CHANNEL, "обращать", "внимание", "obj", "noun", "accusative", 3))
    plans, _ = await ask(db, "а чем чаще всего занимаешься?")
    assert plans == ()


async def test_a_named_day_gets_no_guess_even_when_a_guess_is_asked(db, flag_on):
    await teach(db, *HEDGE, *TIME)
    await act(db, 1, "обработать", "сообщение", ago=timedelta(minutes=3))
    await act(db, 2, "изучить", "конструкцию", ago=timedelta(hours=50))
    plans, _ = await ask(db, "а что ты делал вчера?")
    assert plans == ()
    plans, _ = await ask(db, "что ты делаешь?")  # the present question still gets the material
    assert plans != ()


def test_what_was_said_to_another_person_of_the_channel_counts_half():
    mine = ledger(row(1, 10.0))
    other = ledger(BoredomRow("i:9", 10.0, "k", "express_query_unknown", frozenset(), "", "r:9",
                              "", None, False))
    assert 0.0 < other.reading(behavior_key="k").repeats < mine.reading(behavior_key="k").repeats
    assert other.reading(behavior_key="k").repeats == pytest.approx(
        0.5 * mine.reading(behavior_key="k").repeats)


def test_a_spent_wording_is_a_ranking_key_not_a_rejection():
    from debilgi.language.arbitration import ScoredCandidate

    assert ScoredCandidate.__dataclass_fields__["spent"].default is False
    assert ScoredCandidate.__dataclass_fields__["eligible"].default is not False  # B never sets it


async def test_the_journal_is_the_channels_with_the_addressee_marked(db, flag_on):
    for n, user in enumerate((USER, 101)):
        await EventRepository(db).append(ev(f"p{n}", actor=user))
        await db.execute(
            "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,"
            "prompt_event_id,guild_id,"
            "channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
            "expected_reward,sample_count,selection_json,state_json,created_at,sent_at,"
            "response_message_id,response_text) VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"i:{n}", f"i:{n}", f"e:p{n}", GUILD, CHANNEL, user, "[]", "k",
             "express_query_unknown", "pk",
             "{}", 0.0, 0, "{}", "{}", NOW.isoformat(),
             (NOW - timedelta(minutes=n + 1)).isoformat(), 5000 + n, "Я не знаю."))
    got = await MeaningIntentRepository(db).boredom_ledger(ev(1))
    assert sorted((r.intention_id, r.person) for r in got.rows) == [("i:0", True), ("i:1", False)]
