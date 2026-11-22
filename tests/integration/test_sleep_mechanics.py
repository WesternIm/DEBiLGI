"""Stage 4.15: meaning-keyed holes, revisits across cycles and restarts mid-sleep."""

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from debilgi.cognition.continuous import ContinuousCognitionService
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.response_trace import ResponseTraceRepository
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 17, 10, tzinfo=UTC)


def event(n=1, text="а ты предположи", **kwargs):
    return Event(f"sleep:{n}", 1, 2, 3, 9, EventType.MESSAGE_CREATED,
                 NOW + timedelta(seconds=n),
                 {"message_id": n, "bot_user_id": 9, "bot_mentioned": True,
                  "content": text, "author_is_bot": False, **kwargs}, "discord")


def candidate(text="Я не знаю."):
    return {"renderer": "sleep_replay:query_unobserved_relation", "text": text,
            "speech_act": "brief_statement", "source_confidence": 0.5,
            "metadata": {"communicative_status": "express_limit",
                         "locally_composed": True, "authored_reply_template": False}}


async def hole(db, item, *, answer=None, kind="withheld"):
    await EventRepository(db).append(item)
    trace = ResponseTraceRepository(db)
    await trace.begin(item, prompt=str(item.payload["content"]))
    await trace.finish(item.event_id, status=kind, final_text=answer)


async def sleeping(db, replay=None):
    clock = [NOW]
    service = ContinuousCognitionService(db, clock=lambda: clock[0], presence=AsyncMock(),
                                         replay=replay)
    await service.restore_sleep()
    clock[0] += timedelta(minutes=30)
    await service.rhythm_once()
    assert service.sleeping
    return service, clock, None


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "sleep-mechanics.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


async def gaps(db):
    return [dict(r) for r in await db.fetchall(
        "SELECT * FROM continuous_cognition_gaps WHERE signature LIKE 'sleep:%' "
        "ORDER BY first_seen_at")]


async def runtime(db):
    return dict(await db.fetchone("SELECT * FROM continuous_cognition_runtime"))


async def test_same_meaning_with_other_wording_is_one_more_occurrence(db):
    service, _, _ = await sleeping(db)
    for n, text in ((1, "значит так и будем молчать?"), (2, "так и будешь молчать?"),
                    (3, "ты рад?"), (4, "но ты рад?"), (5, "ты чему так рад?")):
        item = event(n, text)
        await hole(db, item)
        assert await service.capture_sleep_gap(item.event_id) == 1
    rows = await gaps(db)
    by_count = sorted((r["occurrence_count"], json.loads(r["details_json"])["prompt"])
                      for r in rows)
    # Silence (future, inclusive «мы» or «ты») and the polar «рад» merge; the
    # reason question about the same state asks for another slot and stays apart.
    assert [c for c, _ in by_count] == [1, 2, 2]
    silence = next(r for r in rows if "молчать" in r["details_json"])
    details = json.loads(silence["details_json"])
    assert details["occurrences"] == ["sleep:1", "sleep:2"]
    assert silence["source_event_id"] == "sleep:2"
    assert json.loads(silence["meaning_scope"])[0] == "meaning"


async def test_other_person_or_channel_keeps_its_own_hole(db):
    service, _, _ = await sleeping(db)
    first = event(1, "так и будешь молчать?")
    other = replace(event(2, "так и будешь молчать?"), actor_id=77)
    for item in (first, other):
        await hole(db, item)
        await service.capture_sleep_gap(item.event_id)
    assert [r["occurrence_count"] for r in await gaps(db)] == [1, 1]


async def test_legacy_surface_holes_are_rekeyed_and_merged_on_restore(db):
    rows = []
    for n, text in ((1, "значит так и будем молчать?"), (2, "так и будешь молчать?")):
        item = event(n, text)
        await hole(db, item)
        prompt = ContinuousCognitionService._sleep_prompt(item)
        signature = "sleep:" + hashlib.sha256(
            f"{item.channel_id}:{item.actor_id}:{prompt}".encode()).hexdigest()[:32]
        gap_id = f"sleep-gap:1:{signature[6:]}"
        stamp = (NOW + timedelta(minutes=n)).isoformat()
        await db.execute(
            "INSERT INTO continuous_cognition_gaps (gap_id,guild_id,signature,question_kind,"
            "answer_role,source_event_id,first_seen_at,last_seen_at,details_json,"
            "replayed_count,visited_occurrences) VALUES (?,1,?,'withhold','dialogue_response',"
            "?,?,?,?,1,1)",
            (gap_id, signature, item.event_id, stamp, stamp,
             json.dumps({"kinds": ["withhold"], "actor_id": item.actor_id,
                         "channel_id": item.channel_id, "prompt": prompt, "old_text": ""})))
        rows.append(gap_id)
    await db.execute(
        "INSERT INTO continuous_cognition_hypotheses (hypothesis_id,guild_id,signature,"
        "predicate_key,roles_json,epistemic_status,confidence,first_seen_at,last_reviewed_at,"
        "sleep_gap_id) VALUES ('h',1,'h','sleep_response','{}','hypothesis',.5,?,?,?)",
        (NOW.isoformat(), NOW.isoformat(), rows[1]))
    service = ContinuousCognitionService(db, clock=lambda: NOW + timedelta(hours=1))
    await service.restore_sleep()
    merged = await gaps(db)
    assert len(merged) == 1
    gap = merged[0]
    assert gap["occurrence_count"] == 2 and gap["replayed_count"] == 2
    assert gap["first_seen_at"] == (NOW + timedelta(minutes=1)).isoformat()
    assert gap["last_seen_at"] == (NOW + timedelta(minutes=2)).isoformat()
    assert gap["source_event_id"] == "sleep:2"
    assert json.loads(gap["details_json"])["occurrences"] == ["sleep:1", "sleep:2"]
    assert (await db.fetchone("SELECT sleep_gap_id FROM continuous_cognition_hypotheses"))[0] \
        == gap["gap_id"]
    # A later repeat lands on the same hole, and a second restore changes nothing.
    again = event(3, "ну так и будешь молчать?")
    await hole(db, again)
    await service.capture_sleep_gap(again.event_id)
    assert [r["occurrence_count"] for r in await gaps(db)] == [3]
    assert await ContinuousCognitionService(db).merge_sleep_gaps() == 0


async def test_sleep_priority_counts_occurrences_before_freshness(db):
    replay = AsyncMock(return_value=())
    service, _, _ = await sleeping(db, replay)
    for n, text in ((1, "так и будешь молчать?"), (2, "ты будешь молчать?"),
                    (3, "а ты кто?")):
        item = event(n, text)
        await hole(db, item)
        await service.capture_sleep_gap(item.event_id)
    await service.sleep_once()
    assert replay.call_args.args[0].event_id == "sleep:2"


class VaryingReplay:
    """Each visit examines a wider window of plan keys over the same pieces."""

    def __init__(self, windows, accept=None):
        self.windows = windows
        self.accept = accept or {}
        self.calls = []

    async def __call__(self, item, *, variation=None):
        visit = variation["visit"]
        self.calls.append((item.event_id, visit, tuple(variation["tried"])))
        window = self.windows[min(visit, len(self.windows) - 1)]
        variation["examined"].extend(k for k in window if k not in variation["tried"])
        return tuple(self.accept.get(visit, ()))


async def test_open_holes_are_revisited_each_cycle_until_exhausted(db):
    replay = VaryingReplay([("a",), ("a", "b"), ("a", "b"), ("a", "b", "c")])
    service, clock, _ = await sleeping(db, replay)
    item = event(1, "ты чему так рад?")
    await hole(db, item, answer="Я не знаю, почему я рад.", kind="sent")
    await service.capture_sleep_gap(item.event_id)
    visits = []
    for _cycle in range(3):
        for _ in range(4):
            await service.sleep_once()
        gap = (await gaps(db))[0]
        visits.append((gap["replayed_count"], gap["exhausted_at"] is not None))
        clock[0] += timedelta(minutes=90)
        await service.rhythm_once()
        assert service.sleeping
    # Cycle 1 replays, cycle 2 finds a new plan «b», cycle 3 finds nothing new.
    assert visits == [(1, False), (2, False), (3, True)]
    assert json.loads((await gaps(db))[0]["details_json"])["tried"] == ["a", "b"]
    state = await runtime(db)
    # Each executed unit (replays, revisits, synonyms, ...) is counted.
    assert state["sleep_units"] >= 3 and state["total_work_processed"] >= state["sleep_units"]
    # An exhausted hole is idle until new human material arrives.
    for _ in range(4):
        await service.sleep_once()
    assert (await gaps(db))[0]["replayed_count"] == 3
    fresh = replace(event(50, "кстати, я вчера гулял", bot_mentioned=False),
                    timestamp=clock[0] + timedelta(seconds=1))
    await EventRepository(db).append(fresh)
    clock[0] += timedelta(minutes=90)
    await service.rhythm_once()
    await service.sleep_once()
    gap = (await gaps(db))[0]
    assert gap["replayed_count"] == 4 and gap["exhausted_at"] is None
    assert replay.calls[-1][1] == 3 and replay.calls[-1][2] == ("a", "b")


async def test_found_hypothesis_stops_revisits_and_its_rejection_reopens_them(db):
    replay = VaryingReplay([("a",), ("a", "b")], accept={0: (candidate(),)})
    service, clock, _ = await sleeping(db, replay)
    item = event(1)
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    assert await service.sleep_once() == 1
    clock[0] += timedelta(minutes=90)
    await service.rhythm_once()
    for _ in range(4):
        await service.sleep_once()
    assert len(replay.calls) == 1
    await db.execute("UPDATE continuous_cognition_hypotheses SET status='rejected'")
    clock[0] += timedelta(minutes=90)
    await service.rhythm_once()
    await service.sleep_once()
    assert len(replay.calls) == 2 and replay.calls[-1][1] == 1


async def test_merged_hole_revisits_rotate_over_its_occurrences(db):
    replay = VaryingReplay([("a",), ("b",), ("c",)])
    service, clock, _ = await sleeping(db, replay)
    for n, text in ((1, "значит так и будем молчать?"), (2, "так и будешь молчать?")):
        item = event(n, text)
        await hole(db, item)
        await service.capture_sleep_gap(item.event_id)
    for _ in range(3):
        await service.sleep_once()
        clock[0] += timedelta(minutes=90)
        await service.rhythm_once()
    assert [c[0] for c in replay.calls] == ["sleep:2", "sleep:1", "sleep:2"]


async def test_process_restart_mid_sleep_resumes_with_real_units(db):
    replay = VaryingReplay([("a",), ("a", "b"), ("a", "b", "c")])
    service, clock, _ = await sleeping(db, replay)
    item = event(1, "так и будешь молчать?")
    await hole(db, item)
    await service.capture_sleep_gap(item.event_id)
    service._maintenance = AsyncMock(return_value=True)
    for _ in range(5):
        await service.sleep_once()
    # The old process spent this cycle: replay, two experiments, maintenance.
    before = await runtime(db)
    assert before["experiment_units"] == 2 and before["maintenance_at"] is not None
    units = before["sleep_units"]
    await service.sleep_once()
    assert (await runtime(db))["sleep_units"] == units
    # Reboot 20 minutes later, still inside the stored cycle.
    clock[0] += timedelta(minutes=20)
    await db.close()
    await db.initialize()
    restarted = ContinuousCognitionService(db, clock=lambda: clock[0], replay=replay)
    await restarted.start()
    try:
        for _ in range(200):
            if (await runtime(db))["sleep_units"] > units:
                break
            await asyncio.sleep(0.01)
    finally:
        await restarted.stop()
    after = await runtime(db)
    assert restarted.sleeping and after["sleep_restarts"] == 1
    assert after["sleep_units"] > units
    assert after["sleep_cycles"] == before["sleep_cycles"] + 1
    assert after["cycle_started_at"] == clock[0].isoformat()
    assert (await gaps(db))[0]["replayed_count"] == 2


async def test_restart_after_an_expired_cycle_follows_the_boundary_rule(db):
    service, clock, _ = await sleeping(db)
    await service.note_activity(event(bot_mentioned=False))  # pending message, no tag
    clock[0] += timedelta(hours=3)
    restarted = ContinuousCognitionService(db, clock=lambda: clock[0])
    assert await restarted.resume_sleep_after_restart() == "awake"
    assert not restarted.sleeping
    state = await runtime(db)
    assert state["last_activity_at"] == clock[0].isoformat()
    # Silence is counted from that real activity: 30 minutes later it sleeps.
    clock[0] += timedelta(minutes=29)
    assert await restarted.rhythm_once() == "awake"
    clock[0] += timedelta(minutes=1)
    assert await restarted.rhythm_once() == "sleeping"


async def test_real_replay_revisit_reports_and_skips_tried_plans(db):
    from debilgi.config import Settings
    from debilgi.discord.bot import create_bot
    from debilgi.events.normalizer import normalize_message_created

    settings = Settings(discord_token=None, database_path=db.path,
                        migrations_path=Path("migrations"), test_replies_enabled=True)
    bot = create_bot(settings, db, EventRepository(db))
    try:
        await bot.pipeline.start()
        previous = None
        for n, text in enumerate(("Ты хочешь читать?", "Хочу!", "Наверное.",
                                  "А ты хочешь гулять?", "Не хочу.")):
            item = normalize_message_created(
                message_id=400 + n, guild_id=1, channel_id=2, author_id=30 + n % 2,
                created_at=NOW - timedelta(minutes=10 - n), content=text, author_is_bot=False,
                reply_to_message_id=previous, reply_to_author_id=30 + (n - 1) % 2 if n else None,
                bot_user_id=9)
            await bot.pipeline.publish(item)
            await bot.pipeline.join()
            previous = 400 + n
        target = event(80, "Ты хочешь плавать?")
        await hole(db, target, answer="Я не знаю, хочу ли я плавать.", kind="sent")
        first = {"visit": 0, "tried": ()}
        original = await bot.reply_service.replay_sleep_episode(target)
        varied = await bot.reply_service.replay_sleep_episode(target, variation=first)
        # Visit 0 is the original replay; it now reports what it examined.
        assert [v["text"] for v in varied] == [v["text"] for v in original]
        assert first["examined"]
        second = {"visit": 1, "tried": tuple(first["examined"])}
        revisit = await bot.reply_service.replay_sleep_episode(target, variation=second)
        assert not set(second["examined"]) & set(first["examined"])
        assert all(v["metadata"]["intention_plan_key"] not in first["examined"]
                   for v in revisit)
        assert not await db.fetchone("SELECT 1 FROM meaning_claims WHERE source_kind='sleep'")
    finally:
        await bot.close()


def test_analytic_future_is_read_only_for_the_hole_key():
    from debilgi.language.semantic_graph import parse_proposition

    # Daytime readers keep their historical behaviour.
    assert parse_proposition("так и будешь молчать?", allow_question=True) is None
    thought = parse_proposition("так и будешь молчать?", allow_question=True,
                                future_auxiliary=True)
    assert thought.predicate.lemma == "молчать"
    assert dict(thought.predicate.features)["tense"] == "future"
    assert parse_proposition("ты будешь хотеть есть?", allow_question=True,
                             future_auxiliary=True) is None


def _guess(text, negated):
    return {"text": text, "metadata": {
        "sleep_opinion": True, "sleep_canonical": {},
        "message_plan": {"claim": _CLAIM(negated)}}}


def _CLAIM(negated):
    from dataclasses import asdict

    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    return asdict(ClauseThought(
        ConceptLexeme("быть", "быть", "verb", (("tense", "present"),)),
        (SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
         SemanticDependency("predicate_complement",
                            ConceptArgument(ConceptLexeme("рад", "рад", "adjective")))),
        negated=negated, confidence=0.5))


class _Ctx:
    def __init__(self, features=()):
        self.affect_features = features


def _stance(text, negated, recent, features=(), kind="question"):
    from types import SimpleNamespace

    from debilgi.cognition.continuous import ContinuousCognitionService

    event = SimpleNamespace(turn_form=SimpleNamespace(kind=kind))
    return ContinuousCognitionService._own_evidence_stance(
        event, _guess(text, negated), _Ctx(features), recent)


def test_guess_polarity_follows_the_agents_own_affect_without_asserting():
    glad = (("emotion_valence", 0.6), ("mood_valence", 0.2))
    positive = _stance("Наверное, я рад.", False, ((), ()), glad)
    negative = _stance("Наверное, я не рад.", True, ((), ()), glad)
    assert positive["stance_consistency"] > 0.5 > negative["stance_consistency"]
    assert positive["stance_evidence"] == ["own_affect"] and "hard_rejection" not in positive


def test_guess_without_agent_evidence_sets_no_stance():
    assert _stance("Наверное, я рад.", False, ((), ())) == {}


def test_guess_cannot_contradict_the_agents_own_turn_on_a_repeated_question():
    said = ((_CLAIM(False),), ("Наверное, я рад.",))
    contradiction = _stance("Наверное, я не рад.", True, said)
    assert contradiction["hard_rejection"] == "own_turn_contradiction"
    assert contradiction["stance_consistency"] == 0.0
    # The position survives one repeated question; said twice already, it would only loop.
    repeat = _stance("Наверное, я рад.", False, said)
    assert "hard_rejection" not in repeat and repeat["restated_position"] is True
    looping = _stance("Наверное, я рад.", False, (said[0], ("Наверное, я рад.",) * 2))
    assert looping["hard_rejection"] == "repeats_own_turn"
    assert "hard_rejection" not in _stance("Наверное, я не рад.", True, said, kind="statement")


def test_guess_polarity_is_tied_to_the_events_that_moved_the_agents_own_affect():
    from types import SimpleNamespace

    from debilgi.cognition.continuous import ContinuousCognitionService

    glad = (("emotion_valence", 0.6), ("mood_valence", 0.2))
    events = (("e:late", 0.275), ("e:peak", 0.28), ("e:low", -0.4), ("e:early", 0.28))

    def stance(text, negated):
        event = SimpleNamespace(turn_form=SimpleNamespace(kind="question"))
        return ContinuousCognitionService._own_evidence_stance(
            event, _guess(text, negated), _Ctx(glad), ((), ()), events)

    positive = stance("Наверное, я рад.", False)
    # Same sign only, strongest stimulus first, the earlier of equals first.
    assert positive["affect_events"] == ["e:early", "e:peak", "e:late"]
    assert positive["affect_basis"] == "familiarity"
    assert "affect_events" not in stance("Наверное, я не рад.", True)
    assert "affect_events" not in _stance("Наверное, я рад.", False, ((), ()))


async def _position_hypothesis(db, negated):
    from dataclasses import asdict

    from debilgi.language.semantic_graph import parse_proposition

    claim = asdict(parse_proposition("я не рад" if negated else "я рад", perspective="agent"))
    data = {"actor_id": 3, "channel_id": 2, "receipts": [700], "candidate": {
        "text": "Наверное, я не рад." if negated else "Наверное, я рад.",
        "metadata": {"sleep_opinion": True, "sleep_canonical": {},
                     "message_plan": {"claim": claim}}}}
    await db.execute(
        "INSERT INTO continuous_cognition_hypotheses (hypothesis_id,guild_id,signature,"
        "predicate_key,roles_json,epistemic_status,confidence,first_seen_at,last_reviewed_at) "
        "VALUES ('h1',1,'sleep:h1','sleep_response',?,'hypothesis',0.5,?,?)",
        (json.dumps(data, ensure_ascii=False), NOW.isoformat(), NOW.isoformat()))


async def _feedback(db, n, text):
    service = ContinuousCognitionService(db, clock=lambda: NOW)
    item = event(n, text, reply_to_message_id=700)
    await EventRepository(db).append(item)
    changed = await service.observe_sleep_feedback(item)
    relations = [r[0] for r in await db.fetchall(
        "SELECT relation FROM continuous_cognition_evidence WHERE source_event_id=?",
        (item.event_id,))]
    return changed, relations


async def test_question_or_retelling_that_presupposes_the_position_is_not_a_verdict(db):
    await _position_hypothesis(db, negated=True)
    for n, text in ((1, "почему ты не рад?"), (2, "но ты не рад?"),
                    (3, "ты говорил, что ты не рад")):
        assert await _feedback(db, n, text) == (0, [])
    row = await db.fetchone("SELECT status,confidence FROM continuous_cognition_hypotheses")
    assert tuple(row) == ("working", 0.5)
    # A person's own statement of the same meaning is a vote, by its sign.
    assert await _feedback(db, 4, "ты не рад") == (1, ["supports"])


async def test_opposite_statement_contradicts_and_plain_agreement_supports(db):
    await _position_hypothesis(db, negated=False)
    assert await _feedback(db, 1, "ты не рад") == (1, ["contradicts"])
    status = await db.fetchone("SELECT status FROM continuous_cognition_hypotheses")
    assert status[0] == "rejected"
    await db.execute("DELETE FROM continuous_cognition_evidence")
    await db.execute("UPDATE continuous_cognition_hypotheses SET status='working'")
    assert await _feedback(db, 2, "да") == (1, ["supports"])
    # Other meaning: the general attribution is unchanged.
    await db.execute("DELETE FROM continuous_cognition_evidence")
    assert await _feedback(db, 3, "не, ты чё") == (1, ["contradicts"])


def _past(negated, text, n=1, ago_hours=16):
    claim = _CLAIM(negated)
    return {"claim": claim, "text": text, "intention_id": f"i:{n}", "receipt_id": f"receipt:2:{n}",
            "sent_at": (NOW - timedelta(hours=ago_hours)).isoformat()}


def _with_history(text, negated, positions, grounds=(), kind="question"):
    from types import SimpleNamespace

    event = SimpleNamespace(turn_form=SimpleNamespace(kind=kind))
    return ContinuousCognitionService._own_evidence_stance(
        event, _guess(text, negated), _Ctx(), ((), (), ), (),
        {"positions": positions, "grounds": list(grounds)})


def test_a_position_from_yesterday_is_not_flipped_without_a_new_ground():
    old = [_past(True, "Наверное, я не рад.")]
    flipped = _with_history("Наверное, я рад.", False, old)
    assert flipped["hard_rejection"] == "own_turn_contradiction"
    kept = _with_history("Наверное, я не рад.", True, old)
    assert "hard_rejection" not in kept and kept["restated_position"] is True


def test_a_changed_position_names_what_it_revises_and_the_persons_statement():
    old = [_past(True, "Наверное, я не рад.")]
    revised = _with_history("Наверное, я рад.", False, old, grounds=["e:77"])
    assert "hard_rejection" not in revised
    assert revised["revises"] == {"intention_id": "i:1", "receipt": "receipt:2:1",
                                  "sent_at": old[0]["sent_at"], "negated": True}
    assert revised["revised_because"] == ["e:77"]
    from debilgi.experience.meaning_intents import delivered_basis

    basis = delivered_basis((), {**_guess("x", False)["metadata"], **revised})
    assert basis["revises"]["receipt"] == "receipt:2:1" and basis["revised_because"] == ["e:77"]


def test_a_position_said_twice_is_not_repeated_a_third_time_word_for_word():
    twice = [_past(False, "Наверное, я рад.", 1), _past(False, "Наверное, я рад.", 2, 2)]
    assert _with_history("Наверное, я рад.", False, twice[:1])["restated_position"] is True
    assert _with_history("Наверное, я рад.", False, twice)["hard_rejection"] == "repeats_own_turn"


async def test_position_history_reads_this_person_only_and_grounds_need_a_statement(db):
    from dataclasses import asdict

    from debilgi.language.semantic_graph import parse_proposition

    async def turn(n, actor, text, at):
        await EventRepository(db).append(Event(
            f"h:{n}", 1, 2, actor, 9, EventType.MESSAGE_CREATED, at,
            {"message_id": n, "content": text, "author_is_bot": False}, "discord"))
        await db.execute(
            "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,"
            "actor_id,utterance_type,question_status,content,observed_at) VALUES "
            "(?,?,?,?,?,?,?,?,?)",
            (f"h:{n}", 1, 2, n, actor, "assertion", "not_question", text, at.isoformat()))

    claim = asdict(parse_proposition("я не рад", perspective="agent"))
    sent = NOW - timedelta(hours=16)
    await EventRepository(db).append(Event(
        "h:0", 1, 2, 3, 9, EventType.MESSAGE_CREATED, sent,
        {"message_id": 0, "content": "?", "author_is_bot": False}, "discord"))
    for actor, n in ((3, 1), (4, 2)):
        await db.execute(
            "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,"
            "prompt_event_id,guild_id,channel_id,user_id,context_key,behavior_key,operation,"
            "plan_key,plan_json,expected_reward,sample_count,selection_json,state_json,"
            "created_at,status,response_message_id,sent_at,response_text) VALUES "
            "(?,?,'h:0',1,2,?,'c','b','op','k',?,0,0,'{}','{}',?,'sent',?,?,?)",
            (f"i:{n}", f"i:{n}", actor, json.dumps({"claim": claim}), sent.isoformat(), 900 + n,
             sent.isoformat(), "Наверное, я не рад."))
    await turn(1, 3, "почему ты рад?", NOW - timedelta(hours=1))
    await turn(2, 3, "ты говорил, что ты рад", NOW - timedelta(minutes=50))
    await turn(3, 4, "ты рад", NOW - timedelta(minutes=40))      # another person
    service = ContinuousCognitionService(db, clock=lambda: NOW)
    probe = event(50, "так ты рад?")
    glad = {"text": "Наверное, я рад.", "metadata": {
        "sleep_opinion": True, "sleep_canonical": {}, "message_plan": {"claim": asdict(
            parse_proposition("я рад", perspective="agent"))}}}
    got = await service._own_position_history(probe, glad)
    assert [p["intention_id"] for p in got["positions"]] == ["i:1"] and got["grounds"] == []
    await turn(4, 3, "ты рад", NOW - timedelta(minutes=30))
    got = await service._own_position_history(probe, glad)
    assert got["grounds"] == ["h:4"]
