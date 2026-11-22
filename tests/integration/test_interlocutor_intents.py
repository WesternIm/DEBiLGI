import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.generative_planner import objective_for, plan_from_evidence
from debilgi.language.interlocutor_intent import (
    MAX_INTENT_CHARACTERS,
    intent_features,
    lexical_slot_supported,
    realize_clarification,
)
from debilgi.language.planning import ReplyContext
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.word_relations import WordRelation, observe_word_relations
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 10, tzinfo=UTC)


def turn(n, text, *, actor=100, **payload):
    event = normalize_message_created(
        message_id=n, guild_id=1, channel_id=10, author_id=actor,
        content=text, created_at=NOW + timedelta(seconds=n), author_is_bot=False,
        bot_user_id=999,
    )
    return replace(event, payload={**event.payload, **payload})


@pytest.fixture
async def memory(tmp_path):
    db = Database(tmp_path / "intent.sqlite3", Path("migrations"))
    await db.initialize()
    assimilation = SemanticAssimilationService(db, SelfDatasetRepository(db))
    learner = DialogueStructureLearning(db, assimilation)

    async def observe(event):
        await EventRepository(db).append(event)
        await DiscourseMemoryRepository(db).observe(event)
        await learner.observe(event)

    yield db, learner.interlocutor, observe
    await db.close()


async def test_observed_intentions_are_grounded_in_response_and_continuation_not_rewards(memory):
    db, partner, observe = memory
    for event in (
        turn(1, "Чем занимаешься?"),
        turn(2, "Я читаю книгу.", actor=200),
        turn(3, "А какую книгу читаешь?"),
        turn(4, "Ты любишь собак?", actor=300),
        turn(5, "Я люблю собак.", actor=400),
    ):
        await observe(event)
    hypotheses = await partner.infer(turn(100, "Чем занимаешься?"), "Чем занимаешься?")
    assert hypotheses
    best = hypotheses[0]
    assert best.signature["self_directed"]
    assert best.signature["response_predicates"] == ["читать"]
    assert dict(best.outcomes)["topic_continued"] == 1
    assert best.observed_count == 1 and best.live_count == 0 and best.confidence <= 0.64
    other = await partner.infer(turn(101, "Ты любишь собак?"), "Ты любишь собак?")
    assert other[0].cluster_id != best.cluster_id
    assert not await db.fetchone("SELECT 1 FROM language_meaning_intentions")
    assert not await MeaningIntentRepository(db).learned_goals(turn(102, "что дальше?"))


async def test_question_slots_transfer_user_focus_and_pymorphy_exception(memory):
    _, partner, observe = memory
    assert not await partner.clarification_plans(turn(100, "про собак"), "про собак", reason="test")
    await observe(turn(1, "Мы живём в селе."))
    await observe(turn(2, "Что за село?", actor=200))
    plans = await partner.clarification_plans(
        turn(100, "про собак"), "про собак", reason="uncertain",
    )
    assert plans and {realize_clarification(p) for p in plans} == {"Что за собаки?"}
    for plan in plans:
        assert plan.focus_lemma == "собака" and plan.evidence_ids
        assert not evaluate_surface(realize_clarification(plan)).hard_issues
    # Suppletive Russian plural is supplied by pymorphy, not by appending a suffix.
    plans = await partner.clarification_plans(
        turn(101, "про детей"), "про детей", reason="uncertain",
    )
    assert {realize_clarification(p) for p in plans} == {"Что за дети?"}
    await observe(turn(3, "Я читаю книги."))
    await observe(turn(4, "Какие книги?", actor=200))
    plans = await partner.clarification_plans(
        turn(102, "про собак"), "про собак", reason="uncertain",
    )
    assert "Какие собаки?" in {realize_clarification(p) for p in plans}


async def test_different_demonstrated_activities_share_a_goal_not_a_ready_answer(memory):
    _, partner, observe = memory
    for event in (
        turn(1, "Чем занимаешься?"), turn(2, "Я читаю книгу.", actor=200),
        turn(3, "Чем занимаешься?"), turn(4, "Я рисую картину.", actor=200),
    ):
        await observe(event)
    hypotheses = await partner.infer(turn(100, "Чем занимаешься?"), "Чем занимаешься?")
    best = hypotheses[0]
    assert best.observed_count == 2
    assert set(best.signature["response_predicates"]) == {"читать", "рисовать"}
    assert "книга" not in repr(best.signature) and "картина" not in repr(best.signature)


async def test_detached_operator_is_learned_from_raw_multiturn_context(memory):
    _, partner, observe = memory
    for event in (
        turn(1, "Я читаю книгу.", actor=200),
        turn(2, "А ты?", actor=200),
        turn(3, "В смысле, читаешь или пишешь?"),
    ):
        await observe(event)
    plans = await partner.clarification_plans(
        turn(100, "Чем занимаешься?"), "Чем занимаешься?", reason="test",
    )
    assert "В смысле «занимаешься»?" in {realize_clarification(p) for p in plans}


async def test_confirmed_sending_limits_repair_chain_and_bot_input_never_trains(memory):
    db, partner, observe = memory
    await observe(turn(1, "Я читаю книгу."))
    await observe(turn(2, "В смысле, читаешь или пишешь?", actor=200))
    prompt = turn(100, "Я рисовал.")
    await EventRepository(db).append(prompt)
    plans = await partner.clarification_plans(prompt, "Я рисовал.", reason="test")
    assert plans
    meanings = MeaningIntentRepository(db)
    selected = await meanings.choose(
        prompt, (plans[0].proposal,), objective_for(analyze_structured_utterance("Я рисовал.")),
        parent_intent_id="parent",
    )
    assert await partner.clarification_allowed(turn(101, "Рисовал?"))
    assert await meanings.record_sent(
        prompt, intention_id=selected.intention_id, response_message_id=1000,
        created_at=NOW + timedelta(seconds=101), text=realize_clarification(plans[0]),
    )
    for n, text in enumerate(("Рисовал?", "Что за рисовал?", "В смысле?"), start=102):
        assert not await partner.clarification_allowed(turn(n, text, reply_to_message_id=1000))
    assert await partner.clarification_allowed(turn(105, "что?", actor=300))
    assert await partner.clarification_allowed(turn(1000, "новая тема"))
    assert not await partner.clarification_allowed(
        turn(1000, "Та же цепочка", reply_to_message_id=1000)
    )
    count = await db.fetchone("SELECT COUNT(*) AS n FROM language_interlocutor_examples")
    await observe(turn(110, "Что за книга?", actor=888, author_is_bot=True))
    await observe(turn(111, "Расскажи подробнее."))
    await observe(turn(112, "В смысле?", actor=999, author_is_bot=True))
    assert not await partner.clarification_allowed(turn(113, "В смысле?", author_is_bot=True))
    after = await db.fetchone("SELECT COUNT(*) AS n FROM language_interlocutor_examples")
    assert after["n"] == count["n"]


async def test_live_correction_overrides_corpus_and_follow_up_does_not_reward(memory):
    db, partner, observe = memory
    await observe(turn(1, "Я читаю книгу."))
    await observe(turn(2, "В смысле, читаешь или пишешь?", actor=200))
    prompt = turn(100, "Я рисовал.")
    await EventRepository(db).append(prompt)
    plans = await partner.clarification_plans(prompt, "Я рисовал.", reason="test")
    meanings = MeaningIntentRepository(db)
    selection = await meanings.choose(
        prompt, (plans[0].proposal,), objective_for(analyze_structured_utterance("Я рисовал.")),
        parent_intent_id="parent",
    )
    await meanings.record_sent(
        prompt, intention_id=selection.intention_id, response_message_id=1000,
        created_at=NOW + timedelta(seconds=101), text=realize_clarification(plans[0]),
    )
    continuation = turn(102, "Я рисовал портрет.", reply_to_message_id=1000)
    await EventRepository(db).append(continuation)
    assert not await meanings.observe(continuation)
    live = await db.fetchone(
        "SELECT * FROM language_interlocutor_examples WHERE source_kind='live'"
    )
    assert live["outcome"] == "topic_continued"
    assert not await meanings.learned_goals(turn(103, "что дальше?"))
    correction = turn(104, "Чё за бред?", reply_to_message_id=1000)
    await EventRepository(db).append(correction)
    assert await meanings.observe(correction)
    hypotheses = await partner.infer(turn(105, "Я рисовал."), "Я рисовал.")
    assert hypotheses and hypotheses[0].live_count == 2
    assert hypotheses[0].observed_count == 0
    assert dict(hypotheses[0].outcomes)["rejected"] == 1
    assert not await meanings.learned_goals(turn(106, "что дальше?"))
    rows = await db.fetchall("SELECT features_json FROM language_interlocutor_examples")
    assert all(isinstance(json.loads(r["features_json"]), list) for r in rows)


async def test_partner_activity_slot_uses_actual_action_and_not_corpus_biography(memory):
    db, partner, observe = memory
    await observe(turn(1, "Чем занимаешься?"))
    await observe(turn(2, "Я читаю книгу.", actor=200))
    prompt = turn(100, "Чем занимаешься?")
    await EventRepository(db).append(prompt)
    hypotheses = await partner.infer(prompt, "Чем занимаешься?")
    assert not await partner.grounded_agent_plans(prompt, "Чем занимаешься?", hypotheses)
    # A recorded clarification supplies the true operational fact 'answering'.
    await observe(turn(3, "Я рисую."))
    await observe(turn(4, "В смысле, рисуешь или пишешь?", actor=200))
    earlier = turn(90, "Я рисовал.")
    await EventRepository(db).append(earlier)
    repair = (await partner.clarification_plans(earlier, "Я рисовал.", reason="test"))[0]
    meanings = MeaningIntentRepository(db)
    selection = await meanings.choose(
        earlier, (repair.proposal,), objective_for(analyze_structured_utterance("Я рисовал.")),
        parent_intent_id="parent",
    )
    await meanings.record_sent(
        earlier, intention_id=selection.intention_id, response_message_id=1000,
        created_at=NOW + timedelta(seconds=91), text=realize_clarification(repair),
    )
    plans = await partner.grounded_agent_plans(prompt, "Чем занимаешься?", hypotheses)
    # A completed receipt cannot establish an ongoing present-tense action.
    assert not plans
    assert "книга" not in repr(plans)
    # A topical question about thoughts cannot use an instrumental activity slot.
    await observe(turn(5, "О чём думаешь?"))
    await observe(turn(6, "Думаю о собаке.", actor=200))
    topical = await partner.infer(prompt, "О чём думаешь?")
    assert not await partner.grounded_agent_plans(prompt, "О чём думаешь?", topical)


async def test_actual_correction_vetoes_even_an_observed_lexical_combination(memory):
    db, partner, _ = memory
    prompt = turn(100, "Кот любит рыбу?")
    await EventRepository(db).append(prompt)
    planning = plan_from_evidence(ReplyContext(
        prompt="Кот любит рыбу?", knowledge_facts=("Кот любит рыбу.",),
    ))
    meanings = MeaningIntentRepository(db)
    selection = await meanings.choose(
        prompt, planning.proposals[:1], planning.objective, parent_intent_id="parent",
    )
    assert selection is not None
    await meanings.record_sent(
        prompt, intention_id=selection.intention_id, response_message_id=1000,
        created_at=NOW + timedelta(seconds=101), text="Кот любит рыбу.",
    )
    correction = turn(102, "Неправильно.", reply_to_message_id=1000)
    await EventRepository(db).append(correction)
    assert await meanings.observe(correction)
    rejected = await partner.rejected_relations(turn(103, "Кот любит рыбу?"))
    assert ("любить", "рыба", "obj") in rejected
    assert not lexical_slot_supported(
        "любить", "рыба", "obj", observe_word_relations("Кот любит рыбу."), rejected=rejected,
    )


def test_exact_lexical_exception_precedes_neighbor_class_and_no_class_recursion():
    links = tuple(WordRelation(g, d, "obj", "noun", "accusative") for g, ds in (
        ("любить", ("собака", "кошка")),
        ("обожать", ("собака", "кошка", "попугай")),
        ("кормить", ("попугай", "лошадь")),
    ) for d in ds)
    assert lexical_slot_supported("любить", "попугай", "obj", links)
    assert not lexical_slot_supported("любить", "стол", "obj", links)
    assert not lexical_slot_supported("любить", "лошадь", "obj", links)
    assert not lexical_slot_supported(
        "любить", "собака", "obj", links,
        rejected=frozenset({("любить", "собака", "obj")}),
    )
    assert intent_features("Ты любишь собак?") != intent_features("Ты кормишь собак?")


def test_intent_input_budget_does_not_infer_an_intention_of_an_intention():
    text = "Чем занимаешься?".ljust(MAX_INTENT_CHARACTERS)
    assert intent_features(text) == intent_features(text + "Ты любишь собак?" * 1000)
