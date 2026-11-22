import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository
from debilgi.language.canonical_claims import activity_thought
from debilgi.language.communication import ResponseComposer
from debilgi.language.dialogue_learning import normalized_response, response_forms
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.generative_planner import (
    decide_response_mode,
    plan_from_evidence,
    response_act_plan,
)
from debilgi.language.planning import ReplyContext
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 14, 14, tzinfo=UTC)


def event(number, text, *, actor=3, parent=None, addressed=True):
    return normalize_message_created(
        message_id=number, guild_id=1, channel_id=2, author_id=actor,
        created_at=NOW + timedelta(seconds=number), content=text,
        author_is_bot=actor == 99, bot_user_id=99, bot_mentioned=addressed,
        reply_to_message_id=parent,
    )


@pytest.fixture
async def learned(tmp_path):
    db = Database(tmp_path / "forms.sqlite3", Path("migrations"))
    await db.initialize()
    learning = SemanticAssimilationService(db, SelfDatasetRepository(db)).dialogue_learning

    async def observe(e):
        await EventRepository(db).append(e)
        await DiscourseMemoryRepository(db).observe(e)
        await learning.observe(e)

    yield db, learning, observe
    await db.close()


async def surfaces(learning, e, *, hook=True):
    forms = await learning.forms_for(e, e.meaning.text)
    c = ReplyContext(prompt=e.meaning.text, input_meaning=e.meaning,
                     turn_form=e.turn_form, source_event_id=e.event_id)
    planning = plan_from_evidence(c)
    decision = decide_response_mode(c, planning,
                                    observed_reaction=any(f.component == "core" for f in forms),
                                    hook_allowed=hook, observed_forms=forms)
    proposal = response_act_plan(c, decision, forms)
    assert decision.mode == "free_form" and proposal
    await EventRepository(learning._database).append(e)
    selected = await MeaningIntentRepository(learning._database).choose(
        e, (proposal,), planning.objective, parent_intent_id="test",
        response_decision=decision,
    )
    assert selected and selected.proposal == proposal
    composer = ResponseComposer(c.structure)
    result = composer.compose(proposal.plan, exchange_forms=forms, variation_key=e.event_id,
                              source_answers=await learning.source_answers(e))
    return decision, forms, result, composer


@pytest.mark.parametrize("text", ["дырка?", "а что есть у тебя?", "аузын жап котакбас",
                                 "а ты попробуй дать наугад"])
async def test_unknown_content_does_not_block_witnessed_nonfactual_turn(learned, text):
    _, learning, observe = learned
    for e in (event(1, "Ну привет"), event(2, "Ну ага!", actor=4, parent=1),
              event(3, "что ещё расскажешь?"),
              event(4, "Даже не знаю, а у тебя?", actor=4, parent=3)):
        await observe(e)
    e = event(100, text)
    forms = await learning.forms_for(e, text)
    c = ReplyContext(prompt=text, turn_form=e.turn_form)
    decision = decide_response_mode(c, plan_from_evidence(c), observed_forms=forms,
                                    observed_reaction=any(f.component == "core" for f in forms))
    assert decision.confidence_level == "silence"
    assert not any(f.meaning == "express_limit" for f in forms)


@pytest.mark.parametrize('text', ['лошара', 'ты дурак', 'ну ты и тупой'])
async def test_unknown_evaluation_uses_human_tone_without_inventing_an_insult(learned, text):
    _, learning, observe = learned
    for i, prompt in enumerate(('Я полицейский', 'У тебя есть кошка?', 'Я инженер')):
        await observe(event(i * 3 + 1, prompt, actor=3 + i))
        await observe(event(i * 3 + 2, 'Ого', actor=7 + i, parent=i * 3 + 1))
    decision, forms, result, composer = await surfaces(learning, event(100, text))
    assert result, composer.records
    assert composer.metadata_for(result[0].text)['reverse_check']['passed']
    assert decision.confidence_level == 'hypothesis'
    assert decision.target_meaning == 'acknowledge'
    assert decision.mixture == (('reaction', 1.0),) and not decision.hypothesis
    assert all(f.event_valence is None for f in forms if f.component == 'core')
    assert all(f.support == 3 for f in forms if f.component == 'core')
    assert not any(f.meaning == 'retort' for f in forms)
    assert decision.input_features and 'valence:negative' not in decision.input_features
    assert all('ого' in s.text.casefold() for s in result)
    # Restart and repeated retrieval don't turn the interpretation into fact.
    assert forms == await learning.forms_for(event(100, text), text)


async def test_directed_retort_requires_a_sign_and_preserves_human_evidence(learned):
    _, learning, observe = learned
    for i in range(3):
        await observe(event(i * 3 + 1, 'ты лох', actor=3 + i))
        await observe(event(i * 3 + 2, 'ты сам плохой', actor=7 + i, parent=i * 3 + 1))
    await observe(event(20, 'ты плохой'))
    await observe(event(21, 'сам такой', actor=4, parent=20))
    forms = await learning.forms_for(event(100, 'лошара'), 'лошара')
    retorts = [f for f in forms if f.component == 'core' and f.meaning == 'retort']
    assert retorts and all(f.event_valence == -1 for f in retorts)
    assert all('evaluation_target:addressee' in f.input_features for f in retorts)
    assert all('discord:message:2' in f.activation_source_ids for f in retorts)
    for text in ('он лошара', 'я лошара', 'ну ты бля крутой'):
        forms = await learning.forms_for(event(101, text), text)
        assert not any(f.meaning == 'retort' for f in forms)


async def test_epistemic_negation_cannot_become_a_tone_prefix_or_a_positive_claim(learned):
    _, learning, observe = learned
    await observe(event(1, "Что ещё расскажешь?"))
    await observe(event(2, "Ну не знаю, а ты?", actor=4, parent=1))
    forms = await learning.forms_for(event(100, "дырка?"), "дырка?")
    assert not any(f.meaning == "express_limit" for f in forms)
    raw, _ = response_forms("Что ещё расскажешь?", "Ну не знаю, а ты?")
    assert not any(f.get("component") == "prefix" and any(s["lemma"] == "не" for s in f["slots"])
                   for f in raw)
    assert any(f.get("meaning") == "express_limit" for f in raw)


async def test_one_negative_event_appraisal_vetoes_many_positive_pairs(learned):
    _, learning, observe = learned
    for i in range(20):
        await observe(event(1 + i * 2, "я потерял зумбрак"))
        await observe(event(2 + i * 2, "Это круто!", actor=4, parent=1 + i * 2))
    await observe(event(50, "я потерял фыркун"))
    await observe(event(51, "Печально.", actor=4, parent=50))
    await observe(event(52, "как дела?"))
    await observe(event(53, "а как ты?", actor=4, parent=52))
    decision, forms, result, _ = await surfaces(learning, event(100, "я потерял книгу"))
    assert decision.act == "self_report" and result
    cores = [f for f in forms if f.component == "core"]
    assert cores and all(f.event_valence < 0 for f in cores)
    assert not any(f.meaning == "appraise" for f in cores)
    assert not any("круто" in s.text for s in result)


async def test_unfamiliar_word_reaction_is_raw_idempotent_and_can_bind_a_new_hook(learned):
    db, learning, observe = learned
    await observe(event(1, "зумбрак"))
    answer = event(2, "зумбрак?", actor=4, parent=1)
    await observe(answer)
    before = dict(await db.fetchone("SELECT * FROM language_observed_exchanges"))
    links = json.loads(before["topic_links_json"])
    assert any(link.get("kind") == "word_reaction" and link["derivation_depth"] == 0
               for link in links)
    await learning.observe(answer)
    assert before == dict(await db.fetchone("SELECT * FROM language_observed_exchanges"))
    _, _, result, _ = await surfaces(learning, event(100, "аузын жап котакбас"))
    assert result and any("котакбас" in s.text.casefold() and "?" in s.text for s in result)
    # Generated bot language is never a new human reaction observation.
    await observe(event(101, result[0].text, actor=99, parent=2))
    assert (await db.fetchone("SELECT COUNT(*) n FROM language_observed_exchanges"))["n"] == 1


async def test_self_question_and_requested_attribute_never_repeat_generic_activity(learned):
    db, _, observe = learned
    memory = PropositionMemory(db)
    source = event(1, "я учусь")
    await observe(source)
    await memory.record_agent_activity(
        source, source_key="raw:study:1", operation="study",
        thought=activity_thought("изучить", "конструкция"), occurred_at=source.utc_timestamp,
    )
    generic = await memory.retrieve_claims(event(100, "что ты сегодня делал?"))
    assert generic.status == "Known"
    for text in ("потому что ты?", "так какую?", "какую конструкцию ты изучил?"):
        result = await memory.retrieve_claims(event(101, text))
        assert result.status == "Unknown" and not result.claims, text


async def test_unaddressed_mode_is_chosen_before_generation(learned):
    _, _, _ = learned
    e = event(100, "ты зумбрастый", addressed=False)
    c = ReplyContext(prompt=e.meaning.text, input_meaning=e.meaning, turn_form=e.turn_form)
    decision = decide_response_mode(c, plan_from_evidence(c))
    assert decision.mode == "silence_or_unknown" and decision.reason == "not_addressed"
    assert not normalized_response("")


@pytest.mark.parametrize("answer", ["ты больной", "тебе холодно", "тебе нельзя", "я на море"])
def test_predicative_and_personal_assertions_are_not_free_reactions(answer):
    assert not any(f.get("meaning") for f in response_forms("как дела", answer)[0])


def test_emotional_predicative_with_degree_can_be_learned_without_a_reply_template():
    forms, _ = response_forms("я провалил экзамен", "Очень жаль.")
    cores = [f for f in forms if f.get("component") == "core"]
    assert len(cores) == 1 and cores[0]["meaning"] == "empathize"
    assert [s["surface"] for s in cores[0]["slots"]] == ["очень", "жаль"]


async def test_predicate_question_respects_transitivity_and_experiencer(learned):
    _, learning, observe = learned
    for e in (event(1, "я читаю книгу"), event(2, "ну что ты читаешь?", actor=4, parent=1),
              event(3, "мне нравится рисовать"),
              event(4, "ой здорово!", actor=4, parent=3)):
        await observe(e)
    _, forms, result, _ = await surfaces(learning, event(100, "мне нравится лепить"))
    assert result and not any("нравишься" in r.text for r in result)
    assert not any(f.component == "core" and any(s["binding"] == "prompt" and s["part"] == "verb"
                   for s in f.slots) for f in forms)


async def test_observed_laughter_remains_amusement_instead_of_generic_acknowledgement(learned):
    _, learning, observe = learned
    for e in (event(1, "я читаю анекдот"), event(2, "ахаха))", actor=4, parent=1),
              event(3, "я танцую"), event(4, "ну здорово!", actor=4, parent=3)):
        await observe(e)
    _, _, result, composer = await surfaces(learning, event(100, "ахахах"))
    assert result
    assert all(composer.metadata_for(r.text)["reaction_meaning"] == "mirror_amusement"
               for r in result)


async def test_question_adjective_agrees_with_bound_head_among_multiple_source_nouns(learned):
    _, learning, observe = learned
    await observe(event(1, "Люблю цветы, выращиваю на балконе"))
    await observe(event(2, "какие цветы?", actor=4, parent=1))
    _, _, result, _ = await surfaces(learning, event(100, "я читаю журнал"))
    assert result and all("какой журнал" in r.text for r in result)


def test_activation_decays_stops_cycles_and_respects_threshold():
    from debilgi.language.semantic_graph import spread_activation

    graph = (("word", "rule", 1.0), ("rule", "meaning", 0.5),
             ("meaning", "word", 1.0), ("meaning", "late", 1.0))
    currents = spread_activation({"word": 1.0}, graph, decay=0.5, max_depth=2)
    assert currents == {"word": 1.0, "rule": 0.5, "meaning": 0.125}
    assert spread_activation({"word": 0.01}, graph, threshold=0.1) == {"word": 0.01}


async def test_transformation_generalizes_content_and_reverse_check_rejects_bad_agreement(learned):
    from debilgi.language.dialogue_learning import reverse_reaction_check

    _, learning, observe = learned
    await observe(event(1, "я читаю книги"))
    await observe(event(2, "ну какие книги читаешь?", actor=4, parent=1))
    e = event(100, "я читаю журнал")
    decision, forms, result, composer = await surfaces(learning, e)
    assert decision.rule_id and decision.rule_source_ids == ("discord:message:2",)
    assert decision.confidence_level in {"answer", "hypothesis"}
    assert decision.activation > 0 and decision.rule_weight > 0
    assert "slot:direct_object:gender:masculine" in decision.input_features
    proposal = response_act_plan(ReplyContext(prompt=e.meaning.text), decision, forms)
    assert result and all("какой журнал" in r.text for r in result)
    for r in result:
        metadata = composer.metadata_for(r.text)
        assert metadata["reverse_check"]["passed"]
        parts = tuple(type(forms[0])(**p) for p in metadata["reaction_parts"])
        bad = reverse_reaction_check(proposal.plan, parts, e.meaning.text, "какие журнал читаешь?")
        assert not bad["passed"] and "agreement_mismatch" in bad["issues"]
    links = json.loads((await learning._database.fetchone(
        "SELECT topic_links_json FROM language_observed_exchanges"))[0])
    rule = next(link for link in links if link.get("rule_id") == decision.rule_id)
    assert rule["source_ids"] == ["discord:message:1", "discord:message:2"]
    assert rule["added"] and rule["removed"] and "журнал" not in str(rule["input"])


@pytest.mark.parametrize("feedback_text,direction", [
    ("да", 1), ("не, ты чё", -1), ("концерт отличный", 1),
])
async def test_hypothesis_feedback_updates_rule_once_and_scopes_delivery(learned, feedback_text,
                                                                       direction):
    from dataclasses import asdict, replace

    from debilgi.language.response_trace import ResponseTraceRepository

    db, learning, observe = learned
    await observe(event(1, "как дела?"))
    await observe(event(2, "а как ты?", actor=4, parent=1))
    prompt_text = ("я вчера на концерте был"
                   if feedback_text == "концерт отличный" else "я вчера болел")
    prompt = event(100, prompt_text)
    await observe(prompt)
    decision, _, result, _ = await surfaces(learning, prompt)
    assert decision.confidence_level == "hypothesis" and result
    trace = ResponseTraceRepository(db)
    await trace.begin(prompt, prompt=prompt.meaning.text)
    await trace.step(prompt.event_id, stage="intent", component="response_mode", status="selected",
                     summary="committed before realization", details={"decision": asdict(decision)},
                     created_at=prompt.utc_timestamp)
    intention = await db.fetchone(
        "SELECT intention_id FROM language_meaning_intentions WHERE prompt_event_id = ?",
        (prompt.event_id,),
    )
    await db.execute("UPDATE language_meaning_intentions SET status='prepared' "
                     "WHERE intention_id=?",
                     (intention[0],))
    assert await MeaningIntentRepository(db).record_sent(
        prompt, intention_id=intention[0], response_message_id=700,
        created_at=prompt.utc_timestamp + timedelta(seconds=1), text=result[0].text,
    )
    next_prompt = event(110, prompt_text)
    assert not await learning.hypothesis_allowed(next_prompt)
    assert await learning.hypothesis_allowed(replace(next_prompt, actor_id=500))
    assert await learning.hypothesis_allowed(replace(next_prompt, channel_id=500))

    def matching_rule(row):
        return next(link for link in json.loads(row[0])
                    if link.get("rule_id") == decision.rule_id)

    sql = "SELECT topic_links_json FROM language_observed_exchanges WHERE answer_event_id=?"
    before = matching_rule(await db.fetchone(sql, ("discord:message:2",)))
    feedback = event(102, feedback_text, parent=700)
    assert not await learning.observe_hypothesis_feedback(replace(feedback, actor_id=500))
    assert await learning.observe_hypothesis_feedback(feedback)
    after = matching_rule(await db.fetchone(sql, ("discord:message:2",)))
    assert (after["weight"] - before["weight"]) * direction > 0
    assert after["feedback_ids"] == [feedback.event_id]
    assert bool(after["exceptions"]) == (direction < 0)
    assert not await learning.observe_hypothesis_feedback(feedback)
    assert after == matching_rule(await db.fetchone(sql, ("discord:message:2",)))
    assert not await learning.observe_hypothesis_feedback(event(103, prompt_text, parent=700))
    assert after == matching_rule(await db.fetchone(sql, ("discord:message:2",)))
    from debilgi.language.dialogue_learning import DialogueStructureLearning

    restarted = DialogueStructureLearning(db, learning._assimilation)
    assert await restarted.forms_for(
        next_prompt, next_prompt.meaning.text) == await learning.forms_for(
            next_prompt, next_prompt.meaning.text)
    forms = await learning.forms_for(next_prompt, next_prompt.meaning.text)
    context = ReplyContext(prompt=next_prompt.meaning.text, turn_form=next_prompt.turn_form)
    withheld = decide_response_mode(
        context, plan_from_evidence(context), observed_forms=forms,
        observed_reaction=any(f.component == "core" for f in forms), hypothesis_allowed=False,
    )
    assert withheld.confidence_level == "silence" and withheld.act == "withhold"


async def test_hypothesis_can_be_a_reaction_without_an_unknown_factual_assertion(learned):
    _, learning, observe = learned
    await observe(event(1, "я вчера работал"))
    await observe(event(2, "ну жиза", actor=4, parent=1))
    decision, _, result, composer = await surfaces(learning, event(100, "я вчера работал"))
    assert decision.confidence_level == "hypothesis" and result
    assert all("?" not in s.text for s in result)
    assert all(composer.metadata_for(s.text)["message_plan"]["claim"]["dependencies"] == ()
               for s in result)


async def test_popular_generic_appraisal_cannot_outvote_a_bound_argument(learned):
    _, learning, observe = learned
    for i in range(20):
        await observe(event(1 + 2 * i, "я читаю текст"))
        await observe(event(2 + 2 * i, "ого круто!", actor=4, parent=1 + 2 * i))
    await observe(event(50, "я читаю книги"))
    await observe(event(51, "какие книги читаешь?", actor=4, parent=50))
    decision, _, result, _ = await surfaces(learning, event(100, "я читаю журнал"))
    assert decision.target_meaning == "slot_question"
    assert result and all("какой журнал" in s.text and "круто" not in s.text for s in result)


async def test_copula_does_not_transfer_a_strangers_negative_event(learned):
    _, learning, observe = learned
    for e in (event(1, "я был на концерте"), event(2, "Очень печально.", actor=4, parent=1),
              event(3, "как дела?"), event(4, "а как ты?", actor=4, parent=3),
              event(5, "я был женат, но развёлся"),
              event(6, "Очень жаль.", actor=4, parent=5),
              event(7, "Неее. Люблю шашлык, из животных"),
              event(8, "Грубовато,", actor=4, parent=7)):
        await observe(e)
    forms = await learning.forms_for(event(100, "я вчера на шашлыках был"),
                                     "я вчера на шашлыках был")
    cores = [f for f in forms if f.component == "core"]
    assert cores and all(f.event_valence is None for f in cores)
    assert not any(f.meaning == "empathize" for f in cores)
    assert not any(f.meaning == "appraise" for f in cores)


async def test_circumstance_adverb_is_not_an_appraisal_of_a_later_evaluation(learned):
    """Live 17.11: «но ты рад» → «рядом!)». No sound similarity was involved:
    the donor's locative answer «Рядом.» to «Из дождливой Кубани а ты» was
    stored as an appraisal core because every adverb defaulted to appraisal,
    and it fired on any later addressed evaluation."""
    from debilgi.language.dialogue_learning import _reaction_parts

    _, learning, observe = learned
    for e in (event(1, "Ты откуда?"), event(2, "Из дождливой Кубани а ты", actor=4, parent=1),
              event(3, "Рядом. Красноярск. Живу в пещере", actor=5, parent=2),
              event(4, "ты молодец"), event(5, "хорошо!", actor=4, parent=4)):
        await observe(e)
    forms = await learning.forms_for(event(100, "но ты рад"), "но ты рад")
    assert not any("рядом" in s.get("surface", "").casefold()
                   for f in forms for s in f.slots)
    assert not _reaction_parts("Из дождливой Кубани а ты", "Рядом. Красноярск.")
    # Grounded matches stay: qualitative adverbs are still appraisals.
    for prompt, answer in (("ты молодец", "хорошо!"), ("как дела", "нормально)"),
                           ("круто", "отлично")):
        assert [p["meaning"] for p in _reaction_parts(prompt, answer)
                if p["component"] == "core"] == ["appraise"]


async def test_hypothesis_budget_lapses_with_the_episode_window(learned):
    db, learning, observe = learned
    prompt = event(800, "я купил машину")
    await observe(prompt)
    sent = NOW + timedelta(seconds=805)
    await db.execute(
        "INSERT INTO language_meaning_intentions (intention_id,parent_intent_id,prompt_event_id,"
        "guild_id,channel_id,user_id,context_key,behavior_key,operation,plan_key,plan_json,"
        "expected_reward,sample_count,selection_json,state_json,created_at,response_message_id,"
        "response_text,sent_at) VALUES ('i1','root',?,1,2,3,'c','b','o','k','{}',0,0,?,'{}',?,77,"
        "'вау!',?)",
        (prompt.event_id, json.dumps({"response_decision": {"confidence_level": "hypothesis"}}),
         sent.isoformat(), sent.isoformat()))
    soon = event(900, "я вчера болел")
    later = normalize_message_created(
        message_id=901, guild_id=1, channel_id=2, author_id=3,
        created_at=sent + timedelta(minutes=11), content="я вчера болел", author_is_bot=False,
        bot_user_id=99, bot_mentioned=True)
    assert await learning.hypothesis_allowed(soon) is False
    assert await learning.hypothesis_allowed(later) is True
