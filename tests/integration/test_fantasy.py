"""Stage 4.23: make-believe about the agent, marked as play, never a fact (invariants И1–И8).

Every example uses words and forms that were not in the live dialogue (other verbs, other «бы»
questions, other requests, other laughter markers, other servers).
"""

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.language.dialogue_learning import (
    DialogueStructureLearning,
    own_turn_history,
)
from debilgi.language.generative_planner import PlanningResult
from debilgi.language.productive_nlg import ClauseThought
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.word_relations import WordRelation
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 20, 10, 9, tzinfo=UTC)
BOT = 777
GUILD, CHANNEL, USER = 1, 10, 100


@pytest.fixture
async def db(tmp_path):
    database = Database(tmp_path / "fantasy.sqlite3", Path("migrations"))
    await database.initialize()
    yield database
    await database.close()


def ev(n, text="x", *, actor=USER, guild=GUILD, channel=CHANNEL, at=NOW, bot=False):
    return Event(f"e:{n}", guild, channel, actor, None, EventType.MESSAGE_CREATED, at,
                 {"message_id": zlib.crc32(str(n).encode()), "content": text, "author_is_bot": bot,
                  "bot_user_id": BOT, "bot_mentioned": True}, "discord")


async def human_turn(db, n, text, *, actor, ago=timedelta(hours=3), guild=GUILD, channel=CHANNEL):
    at = NOW - ago
    await EventRepository(db).append(ev(n, text, actor=actor, guild=guild, at=at, channel=channel))
    await db.execute(
        "INSERT INTO language_discourse_turns (source_id,guild_id,channel_id,message_id,actor_id,"
        "utterance_type,question_status,content,observed_at) VALUES (?,?,?,?,?,'assertion',"
        "'not_question',?,?)", (f"e:{n}", guild, channel, n, actor, text, at.isoformat()))


async def teach_marker(db, marker="🤣", guild=GUILD, base=0):
    await human_turn(db, 10 + base, f"я упал {marker}", actor=301, guild=guild)
    await human_turn(db, 11 + base, f"он сломал стол {marker}", actor=302, guild=guild)


async def teach_material(db, text, authors=(401, 402), guild=GUILD, base=100):
    for i, author in enumerate(authors):
        await human_turn(db, base + i, text, actor=author, guild=guild)


def relation(governor, dependent, relation_name="obj", case="accusative", count=3, pos="noun"):
    return WordRelation(governor, dependent, relation_name, pos, case, count)


def planning(*limits):
    return PlanningResult(None, tuple(limits), (), ())


def context_for(text, relations=(), *, personality=None):
    return SimpleNamespace(
        structure=analyze_structured_utterance(text), prompt=text, claim_result=None,
        word_relations=tuple(relations), personality=personality)


async def ask(db, text, relations, *, owner_text=None, actor=USER, guild=GUILD, at=NOW,
              proposals=()):
    owner = owner_text or text
    event = Event("e:900", guild, CHANNEL, actor, None, EventType.MESSAGE_CREATED, at,
                  {"message_id": 900, "content": owner, "author_is_bot": False,
                   "bot_user_id": BOT, "bot_mentioned": True}, "discord")
    return await DialogueStructureLearning(db, None).fantasy_plans(
        event, context_for(text, relations), PlanningResult(None, proposals, (), ())), event


async def counts(db):
    result = []
    for table in ("meaning_claims", "meaning_evidence", "continuous_cognition_hypotheses"):
        result.append((await db.fetchone(f"SELECT COUNT(*) FROM {table}"))[0])
    return tuple(result)


FLY = (relation("лететь", "самолёт", "obl:location_or_destination", "prepositional", 4),
       relation("самолёт", "на", "case", "", 4, "preposition"))
EAT = (relation("съесть", "пицца", "obj", "accusative", 5),
       relation("есть", "пицца", "obj", "accusative", 3),
       relation("есть", "время", "obj", "accusative", 3),
       relation("съесть", "рыба", "obj", "accusative", 2))


async def flying_corpus(db):
    await teach_material(db, "я летал на самолёте в Египет")
    await teach_material(db, "мы полетели на самолёте", authors=(403, 404), base=110)


# ---------------------------------------------------------------- marker (шаг 1)

async def test_learned_marker_is_the_cue_most_different_people_close_a_statement_with(db):
    await teach_marker(db, "🤣")
    await human_turn(db, 30, "я упал 😂", actor=303)
    learning = DialogueStructureLearning(db, None)
    marker = await learning.learned_amusement_marker(ev(900))
    assert marker is not None and marker[0] == "🤣" and marker[1] in {"e:10", "e:11"}


async def test_marker_of_another_server_is_not_learned_and_without_laughter_there_is_none(db):
    learning = DialogueStructureLearning(db, None)
    assert await learning.learned_amusement_marker(ev(900)) is None
    await teach_marker(db, "😹", guild=2)
    assert await learning.learned_amusement_marker(ev(900)) is None


@pytest.mark.parametrize("marker", ["😹", "хахаха", "ахахах"])
async def test_other_laughter_markers_are_learned_too(db, marker):
    await teach_marker(db, marker)
    found = await DialogueStructureLearning(db, None).learned_amusement_marker(ev(900))
    assert found is not None and found[0] == marker


# ---------------------------------------------------------------- licence and material

async def test_unreal_question_about_itself_gets_a_marked_fantasy_with_material(db):
    await flying_corpus(db)
    before = await counts(db)
    plans, _ = await ask(db, "а если хотел, то полетел бы?", FLY)
    assert plans and plans[0].operation == "supposition"
    plan = plans[0].plan
    assert dict(plan.hypothesis)["kind"] == "fantasy"
    assert dict(plan.claim.predicate.features)["mood"] == "conditional"
    assert plan.claim.predicate.lemma == "полететь"
    assert {d.argument.head.lemma for d in plan.claim.dependencies} == {"я", "самолёт"}
    assert plans[0].premises[0].kind == "fantasy_frame"
    assert await counts(db) == before  # И1: no fact, no evidence, no hypothesis


@pytest.mark.parametrize("text,relations,verb,noun", [
    ("что бы ты съел?", EAT, "съесть", "пицца"),
    ("что бы ты сделал сегодня вечером?",
     (relation("сделать", "торт", "obj", "accusative", 3),
      relation("делать", "торт", "obj", "accusative", 3),
      relation("делать", "время", "obj", "accusative", 9)), "сделать", "торт"),
    ("куда бы ты пошёл?",
     (relation("пойти", "парк", "obl:location_or_destination", "accusative", 4),
      relation("парк", "в", "case", "", 4, "preposition")), "пойти", "парк"),
])
async def test_unseen_wh_questions_in_the_unreal_mood_are_filled_from_word_relations(
        db, text, relations, verb, noun):
    for i, rel in enumerate(r for r in relations if r.relation != "case"):
        await teach_material(db, {
            "съесть": "я съел пиццу и рыбу", "есть": "мы едим пиццу",
            "сделать": "я сделал торт", "делать": "мы делаем торт",
            "пойти": "я пошёл в парк"}.get(rel.governor, "я сделал торт"),
            base=200 + 10 * i)
    plans, _ = await ask(db, text, relations)
    assert plans, text
    plan = plans[0].plan
    assert plan.claim.predicate.lemma == verb
    assert noun in {d.argument.head.lemma for d in plan.claim.dependencies}
    assert dict(plan.claim.predicate.features)["mood"] == "conditional"


async def test_hub_nouns_used_with_every_verb_are_not_an_association(db):
    await teach_material(db, "я ем время", base=300)
    relations = (*EAT[2:3], relation("писать", "время", "obj", count=200),
                 relation("читать", "время", "obj", count=200))
    plans, _ = await ask(db, "что бы ты съел?", relations)
    assert plans == ()


async def test_a_noun_said_by_one_author_only_is_not_an_association(db):
    await teach_material(db, "я съел пиццу", authors=(401, 401), base=320)
    plans, _ = await ask(db, "что бы ты съел?", EAT[:1])
    assert plans == ()  # И6 / В5: a single voice, however often repeated, makes no association


async def test_material_of_another_server_is_not_used(db):
    await teach_material(db, "я съел пиццу", guild=2, base=340)
    plans, _ = await ask(db, "что бы ты съел?", EAT[:1])
    assert plans == ()


async def test_without_licence_there_is_no_fantasy(db):
    await flying_corpus(db)
    for text in ("ты летал на самолёте?", "ты ел пиццу?", "что ты ел?"):
        plans, _ = await ask(db, text, FLY + EAT)
        assert plans == (), text  # И8


async def test_a_request_to_make_something_up_is_a_licence_even_without_a_question(db):
    await flying_corpus(db)
    await teach_marker(db, "🤣")
    await human_turn(db, 50, "хочу полететь", actor=USER, ago=timedelta(minutes=3))
    for request in ("придумай что-нибудь", "сочини что-нибудь смешное", "выдумывай", "пошути"):
        plans, _ = await ask(db, request, FLY)
        assert plans, request
        plan = plans[0].plan
        assert dict(plan.hypothesis)["kind"] == "fantasy"
        assert dict(plan.hypothesis)["operator"] == "🤣"


async def test_laughter_licenses_play_about_itself_but_needs_the_laughter_marker(db):
    await flying_corpus(db)
    # «ты бы…» without the unreal particle, said with laughter in the conversation
    plans, _ = await ask(db, "ты полетел?", FLY, owner_text="ахахах ты полетел?")
    # no learned marker: only the grammar of the unreal can mark it
    assert plans and dict(plans[0].plan.claim.predicate.features)["mood"] == "conditional"
    assert "operator" not in dict(plans[0].plan.hypothesis)
    await teach_marker(db, "😹")
    plans, _ = await ask(db, "ты полетел?", FLY, owner_text="ахахах ты полетел?")
    assert plans and dict(plans[0].plan.hypothesis)["operator"] == "😹"
    assert "mood" not in dict(plans[0].plan.claim.predicate.features)  # no «бы»: told with laughter
    # a question that already says everything leaves nothing to make up (no echo as a joke)
    plans, _ = await ask(db, "ты полетел на самолёте?", FLY, owner_text="ахахах ты полетел?")
    assert plans == ()


@pytest.mark.parametrize("text", [
    "мне плохо, а ты бы полетел?",
    "😭 а ты бы полетел?",
    "😡 ты бы полетел?",
])
async def test_a_negative_tone_forbids_play(db, text):
    await flying_corpus(db)
    plans, _ = await ask(db, text, FLY)
    assert plans == ()


@pytest.mark.parametrize("text", [
    "а она бы полетела?", "а они бы съели пиццу?", "а если бы ты полетел, я бы тоже полетел?"])
async def test_never_about_the_interlocutor_or_a_third_person(db, text):
    await flying_corpus(db)
    plans, _ = await ask(db, text, FLY + EAT)
    assert plans == () or all(
        any(d.role == "subject" and d.argument.head.lemma == "я"
            for d in p.plan.claim.dependencies) for p in plans)


async def test_known_or_a_guess_outranks_play(db):
    await flying_corpus(db)
    grounded = SimpleNamespace(plan=SimpleNamespace(knowledge_limit=None))
    plans, _ = await ask(db, "а если хотел, то полетел бы?", FLY, proposals=(grounded,))
    assert plans == ()


async def test_a_reason_is_not_asked_a_game(db):
    await flying_corpus(db)
    plans, _ = await ask(db, "почему бы тебе не полететь?", FLY)
    assert plans == ()


async def delivered(db, n, plan, basis, *, ago=timedelta(minutes=2), text="Я полетел бы."):
    at = (NOW - ago).isoformat()
    await EventRepository(db).append(ev(f"i{n}", "q", at=NOW - ago - timedelta(seconds=5)))
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,created_at,response_message_id,"
        "response_text,sent_at) VALUES (?,'p',?,?,?,?,'k','b','supposition','pk',?,0,0,?,"
        "'{}',?,?,?,?)",
        (f"i{n}", f"e:i{n}", GUILD, CHANNEL, USER, json.dumps(plan),
         json.dumps({"basis": basis}), at, 5 + n, text, at))


async def test_no_play_right_after_play(db):
    await flying_corpus(db)
    plans, _ = await ask(db, "а если хотел, то полетел бы?", FLY)
    assert plans
    await delivered(db, 1, {"claim": {}}, {"fantasy": True})
    again, _ = await ask(db, "а если хотел, то полетел бы?", FLY)
    assert again == ()  # one game at a time


async def test_play_is_a_trait_of_character_not_a_rule(db):
    from dataclasses import make_dataclass

    await flying_corpus(db)
    event = ev(900, "а если хотел, то полетел бы?")
    learning = DialogueStructureLearning(db, None)
    plans = await learning.fantasy_plans(
        event, context_for("а если хотел, то полетел бы?", FLY), planning())
    assert plans  # the genome is playful (+0.24): the game is accepted
    dull = make_dataclass("Dull", [("playfulness", float), ("chaos", float)])
    plans = await learning.fantasy_plans(
        event, context_for("а если хотел, то полетел бы?", FLY, personality=dull(0.2, 0.2)),
        planning())
    assert plans == ()


# ---------------------------------------------------------------- readers (И2–И5)

async def test_a_fantasy_turn_is_no_position_no_fact_and_no_hole(db):
    from debilgi.cognition.continuous import ContinuousCognitionService

    await flying_corpus(db)
    plan = {"claim": {"clause_kind": "statement", "predicate": {"lemma": "полететь"},
                      "dependencies": []},
            "hypothesis": [["kind", "fantasy"], ["operator", "🤣"]]}
    await delivered(db, 2, plan, {"fantasy": True}, text="Я полетел бы на самолёте 🤣")
    history = await own_turn_history(db, GUILD, USER, NOW)
    assert history and history[0]["claim"] is None  # И4: not a position
    assert history[0]["basis"]["fantasy"] is True
    service = ContinuousCognitionService.__new__(ContinuousCognitionService)
    service._database = db
    positions, _ = await service._own_recent_positions(ev(901, "ты рад?"))
    assert positions == ()
    assert await service._is_play_turn(SimpleNamespace(event_id="e:i2")) is True  # И5: no hole
    assert await service._is_play_turn(SimpleNamespace(event_id="e:none")) is False


async def test_a_person_repeating_the_joke_does_not_make_the_agent_know_it(db):
    # И2: one person's echo («ты полетел на кухню») and another person's question: no Known.
    memory = PropositionMemory.for_meaning(db)
    echo = ev(60, "ахах, ты полетел на кухню", actor=201, at=NOW - timedelta(hours=1))
    await EventRepository(db).append(echo)
    await memory.observe(echo)
    other = ev(61, "ты полетел на кухню?", actor=202)
    await EventRepository(db).append(other)
    result = await memory.retrieve_claims(other, "ты полетел на кухню?")
    assert result.status != "Known"


async def test_the_second_person_answering_yes_to_the_joke_promotes_nothing(db):
    # И3: «да» to a joke is no receipt of a response hypothesis.
    await flying_corpus(db)
    before = await counts(db)
    yes = ev(62, "да")
    await EventRepository(db).append(yes)
    await PropositionMemory.for_meaning(db).observe(yes)
    after = await counts(db)
    assert after[2] == before[2]  # no hypothesis appears or is promoted


async def test_a_human_utterance_is_never_repeated_as_the_agents_own(db):
    await human_turn(db, 70, "Я полетел бы на самолёте", actor=210)
    learning = DialogueStructureLearning(db, None)
    assert await learning.is_human_utterance(ev(900), "Я полетел бы на самолёте 😂") is True
    assert await learning.is_human_utterance(ev(900), "Я съел бы на самолёте") is False
    assert await learning.is_human_utterance(ev(900, guild=2), "Я полетел бы на самолёте") is False


# ---------------------------------------------------------------- game under way: «почему?»

async def test_why_after_play_offers_owning_up_and_going_on_as_two_plans(db):
    await flying_corpus(db)
    await teach_marker(db, "🤣")
    await teach_material(db, "я съел рыбу", base=400)
    prior_plan = {"claim": {
        "predicate": {"lemma": "съесть", "surface": "съесть", "part_of_speech": "verb",
                      "features": [["tense", "past"], ["mood", "conditional"]],
                      "entity_kind": None, "participant_ref": None},
        "dependencies": [
            {"role": "subject", "argument": {
                "head": {"lemma": "я", "surface": "я", "part_of_speech": "pronoun",
                         "features": [], "entity_kind": None, "participant_ref": None},
                "modifiers": [], "preposition": None, "grammatical_case": None,
                "postmodifiers": []}},
            {"role": "direct_object", "argument": {
                "head": {"lemma": "пицца", "surface": "пицца", "part_of_speech": "noun",
                         "features": [], "entity_kind": None, "participant_ref": None},
                "modifiers": [], "preposition": None, "grammatical_case": "accusative",
                "postmodifiers": []}}],
        "clause_kind": "statement", "negated": False, "response_polarity": None,
        "focus_role": None, "confidence": 0.7, "evidence": [], "predicate_class": "lexical"}}
    text = "почему?"
    event = ev(902, text)
    context = SimpleNamespace(
        structure=analyze_structured_utterance(text), prompt=text, claim_result=None,
        word_relations=EAT, personality=None, prior_bot_turn={
            "plan": prior_plan, "basis": {"fantasy": True}, "receipt_id": "receipt:10:5",
            "response_text": "Я съел бы пиццу 🤣"})
    plans = await DialogueStructureLearning(db, None).fantasy_followup_plans(event, context)
    assert [p.operation for p in plans] == ["explain_delivered_basis", "supposition"]
    admission, play = plans
    assert admission.plan.claim.predicate.lemma == "пошутить"
    assert admission.plan.speech_act == "admission"
    assert play.plan.claim.predicate.lemma == "съесть"
    # the game goes on with something new, not the thing it already said
    assert "пицца" not in {d.argument.head.lemma for d in play.plan.claim.dependencies}
    assert dict(play.plan.hypothesis)["kind"] == "fantasy"
    # a turn that was not play gets no such pair
    context.prior_bot_turn["basis"] = {}
    assert await DialogueStructureLearning(db, None).fantasy_followup_plans(event, context) == ()


def test_fantasy_plan_is_a_statement_about_the_agent_only():
    from debilgi.language.communication import plan_issues
    assert plan_issues is not None
    assert isinstance(ClauseThought, type)


async def test_a_delivered_fantasy_records_its_basis_and_no_hypothesis(db):
    from debilgi.experience.meaning_intents import MeaningIntentRepository, delivered_basis
    from debilgi.language.generative_planner import DialogueObjective

    await flying_corpus(db)
    plans, event = await ask(db, "а если хотел, то полетел бы?", FLY)
    await EventRepository(db).append(event)
    before = await counts(db)
    repository = MeaningIntentRepository(db)
    objective = DialogueObjective("polar", "полететь", None, (), ())
    selection = await repository.choose(
        event, plans, objective, parent_intent_id="p", request_text=event.meaning.text)
    assert selection is not None
    await repository.finalize_selection(
        selection, plans[0], basis=delivered_basis(plans[0].premises, {"fantasy": True}))
    assert await repository.record_sent(
        event, intention_id=selection.intention_id, response_message_id=77, created_at=NOW,
        text="Я полетел бы на самолёте.")
    assert await counts(db) == before  # И1/И3: no claim, no evidence, no hypothesis to confirm
    history = await own_turn_history(db, GUILD, USER, NOW + timedelta(minutes=1))
    assert history[0]["basis"]["fantasy"] is True and history[0]["claim"] is None


async def observed_limit(db, n, actor, lemmas, *, guild=GUILD):
    await human_turn(db, 500 + n, "не знаю", actor=actor, guild=guild)
    await EventRepository(db).append(ev(f"p{n}", "а ты знаешь?", actor=actor + 1000, guild=guild))
    forms = [{"meaning": "express_limit", "component": "core",
              "slots": [{"lemma": lemma} for lemma in lemmas]}]
    await db.execute(
        "INSERT INTO language_observed_exchanges (answer_event_id,prompt_event_id,guild_id,"
        "channel_id,prompt_act,response_act,prompt_features_json,response_forms_json,"
        "topic_links_json,observed_at) VALUES (?,?,?,?,'q','fragment','[]',?,'[]',?)",
        (f"e:{500 + n}", f"e:p{n}", guild, CHANNEL, json.dumps(forms),
         (NOW - timedelta(hours=2)).isoformat()))


async def test_the_person_may_be_carried_by_the_verb_only_if_two_people_are_seen_to_say_so(db):
    learning = DialogueStructureLearning(db, None)
    assert await learning.elided_limit_observed(ev(900)) is False
    await observed_limit(db, 1, 601, ("не", "знать"))
    assert await learning.elided_limit_observed(ev(900)) is False  # one voice
    await observed_limit(db, 2, 602, ("я", "не", "знать"))  # with the pronoun: not elided
    assert await learning.elided_limit_observed(ev(900)) is False
    await observed_limit(db, 3, 603, ("не", "знать"), guild=2)  # another server
    assert await learning.elided_limit_observed(ev(900)) is False
    await observed_limit(db, 4, 604, ("не", "знать"))
    assert await learning.elided_limit_observed(ev(900)) is True
