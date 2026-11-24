"""Stage 4.42 (Э5): hypotheses about a person, tested by their reactions; who the person is for the
agent. Flag ``DEBILGI_THOUGHT_CONTENT`` only. A hypothesis is a forecast of a reaction to a kind of
move: it lives in ``continuous_cognition_hypotheses`` and is never a claim.
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.cognition.affect import hostility_cues, stimulus
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import (
    MeaningIntentRepository,
    decay_weight,
    move_type,
    reaction_class,
    reception_cost,
    reweigh_person,
    social_half_life,
    unclosed_share,
)
from debilgi.social.models import PrimitiveVector
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
GUILD, CHANNEL, USER, OTHER = 1, 10, 100, 200


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "person442.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


def message(n, text, *, user=USER, guild=GUILD, at=NOW):
    return Event(f"e:{n}", guild, CHANNEL, user, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(n.encode()), "content": text, "author_is_bot": False,
                  "bot_user_id": 7, "bot_mentioned": True}, "discord")


def prior(n, operation, kind=None):
    return {"intention_id": f"i:{n}", "operation": operation,
            "plan_json": json.dumps({"claim": {"clause_kind": "statement"},
                                     **({"hypothesis": {"kind": kind}} if kind else {})})}


async def reaction(db, repository, n, text, operation, outcome="conversation_continued", **kw):
    event = message(n, text, **kw)
    await EventRepository(db).append(event)
    return await repository.observe_person(event, prior(n, operation), outcome)


async def hypotheses(db):
    return [dict(r) for r in await db.fetchall(
        "SELECT * FROM continuous_cognition_hypotheses WHERE signature LIKE 'person:%'"
        " ORDER BY predicate_key, object_key")]


def test_a_move_is_read_from_the_agents_own_operation():
    assert move_type("retrieve_proposition") == "serious"
    assert move_type("express_query_unknown") == "serious"  # the honest limit takes the question
    assert move_type("compose_reaction:acknowledge") == "reaction"
    assert move_type("compose_reaction:mirror_amusement") == "play"
    assert move_type("compose_reaction:slot_question") == "counter"
    assert move_type("curiosity") == move_type("repair") == "counter"
    assert move_type("supposition", "fantasy") == "play"


def test_a_reaction_is_classed_by_outcome_cues_and_laughter():
    assert reaction_class("ты любишь спать?", "follow_up_question", False) == "pressed"
    assert reaction_class("ты гандон", "conversation_continued", True) == "hostile"
    assert reaction_class("не то", "explicit_rejection", False) == "rejected"
    assert reaction_class("ахахах", "conversation_continued", False) == "laughed"
    assert reaction_class("понятно", "conversation_continued", False) == "continued"


def test_laughter_is_unwelcome_only_at_a_move_taken_seriously():
    assert reception_cost("serious", {"laughed": 0.6, "continued": 0.3}) == 0.6
    assert reception_cost("play", {"laughed": 0.6}) == 0.0
    assert reception_cost("play", {"hostile": 0.4}) == 0.4


def test_what_was_pressed_long_ago_weighs_less_by_the_horizon_of_the_person():
    rows = [{"outcome_type": "follow_up_question", "intention_id": "a",
             "observed_at": (NOW - timedelta(days=7)).isoformat()}]
    old, _ = unclosed_share(rows, NOW)
    rows[0]["observed_at"] = NOW.isoformat()
    fresh, _ = unclosed_share(rows, NOW)
    assert fresh == 0.5 and 0.0 < old < fresh
    w = decay_weight(7 * 86400.0, social_half_life())
    assert old == pytest.approx(w / (w + 1.0))


async def test_a_reaction_makes_a_hypothesis_and_a_second_one_tests_it(db, flag_on):
    repository = MeaningIntentRepository(db)
    assert await reaction(db, repository, "1", "ахахах", "retrieve_proposition") == "laughed"
    [h] = await hypotheses(db)
    assert (h["subject_key"], h["predicate_key"], h["object_key"]) == (
        f"discord:user:{USER}", "reacts_to:serious", "laughed")
    assert h["epistemic_status"] == "hypothesis" and h["positive_evidence"] == 1
    forecast = await repository.person_forecast(message("now", "?"))
    assert forecast["serious"]["checks"] == 1  # one episode: it does not act
    await reaction(db, repository, "2", "кек", "express_query_unknown")
    forecast = await repository.person_forecast(message("now", "?"))
    assert forecast["serious"]["checks"] == 2 and forecast["serious"]["top"] == "laughed"
    # another reaction to the same kind contradicts it and starts its own hypothesis
    await reaction(db, repository, "3", "понятно", "retrieve_proposition")
    rows = {h["object_key"]: h for h in await hypotheses(db)}
    assert rows["laughed"]["negative_evidence"] == 1 and rows["continued"]["positive_evidence"] == 1
    assert rows["laughed"]["confidence"] < 2 / 3  # (a forecast is a trust, here 2 against 1)


async def test_a_hypothesis_is_never_a_claim_and_never_leaves_the_person(db, flag_on):
    repository = MeaningIntentRepository(db)
    for n in "12":
        await reaction(db, repository, n, "ахахах", "retrieve_proposition")
    for table in ("meaning_claims", "meaning_evidence", "meaning_support"):
        assert (await db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0] == 0
    # what the agent learned of one person is not read for another, nor for another server
    assert await repository.person_forecast(message("x", "?", user=OTHER)) == {}
    assert await repository.person_forecast(message("y", "?", guild=2)) == {}
    from debilgi.cognition.continuous import ContinuousCognitionService

    recalled = await ContinuousCognitionService(db).relevant_facts(
        GUILD, ["discord:user:100", "reacts_to:serious", "laughed"])
    assert recalled == ()  # the working-conclusions recall never serves them


async def test_without_the_flag_nothing_is_learned_about_the_person(db, monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    repository = MeaningIntentRepository(db)
    assert await reaction(db, repository, "1", "ахахах", "retrieve_proposition") is None
    assert await hypotheses(db) == []
    assert await repository.person_forecast(message("now", "?")) == {}


async def test_the_night_recounts_trust_and_a_stale_forecast_fades(db, flag_on):
    repository = MeaningIntentRepository(db)
    for n in "12":
        await reaction(db, repository, n, "ахахах", "retrieve_proposition")
    [fresh] = await hypotheses(db)
    week = NOW + timedelta(days=7)
    await reweigh_person(db, [fresh["hypothesis_id"]], week)
    [stale] = await hypotheses(db)
    assert stale["confidence"] < fresh["confidence"]
    assert (stale["positive_evidence"], stale["status"], stale["epistemic_status"]) == (
        2, "working", "hypothesis")  # the evidence counts are raw; the trust is what fades


def test_an_attack_is_not_a_pleasant_stimulus_for_a_familiar_person():
    familiar = PrimitiveVector(agreement=0.3, resistance=0.1, approval=0.4, curiosity=0.2,
                               aversion=0.05, trust=0.9, arousal=0.3)
    assert stimulus(familiar).valence > 0  # (the sign the reaction vector alone gave an insult)
    assert stimulus(familiar, hostile=True).valence < 0


def test_the_word_after_a_rejection_is_read_like_one_after_hostility():
    assert hostility_cues("ты тупой") == ()
    assert hostility_cues("ты тупой", True) == ("escalation:тупой",)


def test_the_agent_speaks_of_itself_in_its_own_gender(monkeypatch):
    from debilgi.language.productive_nlg import ConceptLexeme, _inferred_head_features

    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    assert "gender" not in _inferred_head_features(ConceptLexeme("я", "я", "pronoun"))
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")
    assert _inferred_head_features(ConceptLexeme("я", "я", "pronoun"))["gender"] == "masculine"
    # the second person is not the agent: nothing is lent to it
    assert "gender" not in _inferred_head_features(ConceptLexeme("ты", "ты", "pronoun"))


async def test_the_warmth_of_a_person_moves_with_their_reactions(db, flag_on):
    from debilgi.cognition.core_repository import CognitiveRepository
    from debilgi.cognition.models import EntityType

    cognition = CognitiveRepository(db)
    await db.execute(
        "INSERT INTO cognition_entity_models VALUES (?, 'user', ?, 0.5, 0.5, 0.5, 0.5, 5.0, 0.5,"
        " 0.5, 0, 0.5, 3, ?)", (GUILD, USER, NOW.isoformat()))
    repository = MeaningIntentRepository(db, cognition=cognition)
    await reaction(db, repository, "1", "ты гандон", "retrieve_proposition")
    await reaction(db, repository, "2", "понятно", "retrieve_proposition")
    model = await cognition.entity_model(GUILD, EntityType.USER, USER)
    # hostile (0) then continued (1) at the rate the other fields of the model use
    assert model.bot_action_response == pytest.approx((0.5 * 0.85) * 0.85 + 0.15)
    stored = (await db.fetchone("SELECT bot_action_response FROM cognition_entity_models"))[0]
    assert stored == pytest.approx(model.bot_action_response)
    other = await cognition.entity_model(GUILD, EntityType.USER, OTHER)
    assert other is None  # nobody else's model was touched
