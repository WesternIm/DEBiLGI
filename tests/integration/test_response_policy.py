"""Stage 4: query execution, bounded ambiguity, and one-hop action retrieval."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.models import Event, EventType
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository
from debilgi.language.canonical_claims import Unknown, query_from_event
from debilgi.language.communication import ResponseComposer, plan_issues
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.generative_planner import (
    decide_response_mode,
    knowledge_limit_plan,
    plan_from_evidence,
    resolve_evidence_context,
    response_act_plan,
)
from debilgi.language.interlocutor_intent import InterlocutorHypothesis
from debilgi.language.planning import ReplyContext
from debilgi.language.proposition_memory import PropositionMemory
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 14, 12, tzinfo=UTC)


def event(n, text, *, actor=3, **payload):
    return Event(str(n), 1, 2, actor, 99, EventType.MESSAGE_CREATED,
                 NOW + timedelta(seconds=n),
                 {"content": text, "bot_user_id": 99, "bot_mentioned": True, **payload}, "test")


@pytest.fixture
async def memory(tmp_path):
    db = Database(tmp_path / "policy.sqlite3", Path("migrations"))
    await db.initialize()
    memory = PropositionMemory(db)
    learning = SemanticAssimilationService(db, SelfDatasetRepository(db)).dialogue_learning

    async def observe(e):
        await EventRepository(db).append(e)
        await DiscourseMemoryRepository(db).observe(e)
        await learning.observe(e)
        await memory.observe(e)

    yield memory, db, learning.interlocutor, observe
    await db.close()


async def context(memory, query, **kwargs):
    return ReplyContext(prompt=query.meaning.text, input_meaning=query.meaning,
                        source_event_id=query.event_id,
                        claim_result=await memory.retrieve_claims(query), **kwargs)


@pytest.mark.parametrize("report,predicate", [
    ("я вчера брил ноги", "брить"), ("я вчера мыл окно", "мыть"),
    ("я вчера красил стену", "красить"), ("я вчера читал книгу", "читать"),
    ("я вчера ремонтировал велосипед", "ремонтировать"),
])
async def test_general_action_question_keeps_fact_and_one_hop_proof(memory, report, predicate):
    memory, db, _, observe = memory
    await observe(event(1, report))
    q = event(2, "что я делал вчера?")
    c = await context(memory, q)
    assert c.claim_result.status == "Known"
    claim, = c.claim_result.claims
    assert claim.predicate == predicate
    assert claim.proof.rule_ids[0] == f"predicate_parent:{predicate}:делать:depth=1"
    assert claim.valid_from and claim.valid_to and claim.subject_ref == "discord:user:3"
    planning = plan_from_evidence(c)
    decision = decide_response_mode(c, planning)
    assert decision.mode == "answer"
    await EventRepository(db).append(q)
    chosen = await MeaningIntentRepository(db).choose(q, planning.proposals, planning.objective,
                                                     parent_intent_id="test",
                                                     query_result=c.claim_result)
    assert chosen and chosen.proposal.plan.claim.predicate.lemma == predicate
    assert ResponseComposer(c.structure).compose(chosen.proposal.plan)
    assert (await db.fetchone("SELECT COUNT(*) n FROM meaning_claims"))["n"] == 1


@pytest.mark.parametrize("report", [
    "я вчера болел", "я вчера спал", "я вчера не брил ноги",
    "я вчера хотел брить ноги", "я завтра буду брить ноги", "я завтра брею ноги",
])
async def test_states_and_unexecuted_actions_do_not_fill_action_slot(memory, report):
    memory, _, _, observe = memory
    await observe(event(1, report))
    c = await context(memory, event(2, "что я делал вчера?"))
    assert c.claim_result.status == "Unknown" and not c.claim_result.claims
    assert decide_response_mode(c, plan_from_evidence(c)).mode == "silence_or_unknown"


async def test_correction_replaces_patient_after_generic_question(memory):
    memory, _, _, observe = memory
    await observe(event(1, "я вчера брил ноги"))
    first = await memory.retrieve_claims(event(2, "что я делал вчера?"))
    assert first.status == "Known"
    await observe(event(3, "нет, я брил руки"))
    corrected = await memory.retrieve_claims(event(4, "что я брил?"))
    claim, = corrected.claims
    assert any(d.role == "direct_object" and d.argument.head.lemma == "рука"
               for d in claim.thought.dependencies)
    assert claim.supersedes == (first.claims[0].claim_id,)


async def test_raw_dialogue_can_license_intransitive_action_but_rejection_vetoes_default(memory):
    memory, db, _, observe = memory
    await observe(event(1, "что ты делал вчера?", actor=10, message_id=1))
    await observe(event(2, "я вчера гулял", actor=11, reply_to_message_id=1,
                        reply_to_author_id=10, message_id=2))
    await observe(event(3, "я вчера гулял"))
    result = await memory.retrieve_claims(event(4, "что я делал вчера?"))
    assert result.status == "Known" and result.claims[0].predicate == "гулять"
    # Explicit rejection is an exception to this edge, not a new class to generalize.
    await db.execute("UPDATE language_interlocutor_examples SET outcome = 'rejected'")
    assert (await memory.retrieve_claims(event(5, "что я делал вчера?"))).status == "Unknown"
    # No transitive closure: a demonstrated parent is not itself a raw answer edge.
    assert (await memory.retrieve_claims(event(6, "чем я занимался вчера?"))).status == "Unknown"


@pytest.mark.parametrize("text,focus", [
    ("что ты делаешь?", "что я делаю"), ("че делаешь", "что я делаю"),
    ("ты где?", "где я"), ("какие книги ты читал?", "какие книги я читал"),
    ("что я покупаю?", "что ты покупаешь"),
    ("что Петя делал вчера?", "петя"),
])
def test_unknown_is_a_scoped_assertion_even_when_intent_experience_is_below_point65(text, focus):
    q = event(1, text)
    c = ReplyContext(prompt=q.meaning.text, input_meaning=q.meaning,
                     claim_result=Unknown(query_from_event(q)))
    p = plan_from_evidence(c)
    hypothesis = InterlocutorHypothesis("cluster", {}, 0.64, 1.0, 100, 0, (), ())
    decision = decide_response_mode(c, p, (hypothesis,))
    assert decision.mode == "silence_or_unknown"
    assert decision.fact_confidence is None and decision.intent_confidence == 0.64
    assert decision.expression_confidence is None
    proposal = knowledge_limit_plan(c)
    assert proposal and proposal.plan.knowledge_limit.query == c.claim_result.query
    assert not plan_issues(proposal.plan)
    surfaces = ResponseComposer(c.structure).compose(proposal.plan)
    assert surfaces and "не знаю" in surfaces[0].text.casefold()
    assert focus in surfaces[0].text.casefold() and not surfaces[0].text.endswith("?")
    assert proposal.plan.claim.negated and not proposal.plan.knowledge_limit.question.negated


@pytest.mark.parametrize("text", ["ку бро", "але", "ауу", "че делаешь", "ты где?"])
def test_current_contact_or_question_never_inherits_an_old_polar_focus(text):
    c = ReplyContext(prompt=text, consultation_messages=("user: так ты согласен или нет?",))
    assert resolve_evidence_context(c).prompt == text


@pytest.mark.parametrize("prior,question", [
    ("Дракон встретил рыцаря.", "Он ушёл?"),
    ("Отец встретил брата.", "Он спал?"),
    ("Кот увидел пса.", "Он убежал?"),
    ("Маша встретила Олю.", "Она читала?"),
    ("Мама увидела сестру.", "Она ушла?"),
    ("Учитель встретил ученика.", "Он работал?"),
    ("Врач увидел пациента.", "Он болел?"),
    ("Писатель встретил художника.", "Он рисовал?"),
])
def test_real_ambiguity_has_discriminating_subject_alternatives(prior, question):
    c = ReplyContext(prompt=question, dialogue_messages=(prior,))
    p = plan_from_evidence(c)
    decision = decide_response_mode(c, p)
    assert decision.mode == "clarification" and decision.ambiguous_slot == "subject"
    assert len(decision.alternatives) >= 2 and decision.expected_information_gain > 0
    proposal = response_act_plan(c, decision)
    assert proposal and not plan_issues(proposal.plan)
    surfaces = ResponseComposer(c.structure).compose(proposal.plan)
    assert surfaces and " или " in surfaces[0].text and surfaces[0].text.endswith("?")
    assert decide_response_mode(c, p, clarification_allowed=False).act == "withhold"


async def test_reward_cannot_make_exploration_replace_a_clear_fact(memory):
    memory, db, _, observe = memory
    await observe(event(1, "я читаю книгу"))
    q = event(10, "что я читаю?")
    await EventRepository(db).append(q)
    c = await context(memory, q)
    p = plan_from_evidence(c)
    answer, = p.proposals
    question = replace(answer, operation="request_missing_argument",
                       plan=replace(answer.plan, claim=replace(answer.plan.claim,
                                                              clause_kind="question")))
    unrelated = replace(answer, plan=replace(answer.plan, claim=replace(answer.plan.claim,
                        predicate=replace(answer.plan.claim.predicate, lemma="летать"))),
                        premises=())
    repo = MeaningIntentRepository(db, exploration_rate=1)
    selected = await repo.choose(q, (question, unrelated, answer), p.objective,
                                 parent_intent_id="test", query_result=c.claim_result)
    assert selected and selected.proposal == answer
    assert len(selected.alternatives) == 1


async def test_clarification_receipt_and_slot_outcome_do_not_reward_or_assert_alternatives(memory):
    memory, db, partner, observe = memory
    await observe(event(1, "Дракон встретил рыцаря."))
    q = event(2, "Он ушёл?")
    await EventRepository(db).append(q)
    c = await context(memory, q, dialogue_messages=("Дракон встретил рыцаря.",))
    planning = plan_from_evidence(c)
    mode = decide_response_mode(c, planning)
    proposal = response_act_plan(c, mode)
    assert proposal.plan.claim.predicate.lemma == "спрашивать"
    assert all(t.predicate.lemma != "уйти" for t in proposal.plan.additional_claims)
    repo = MeaningIntentRepository(db)
    selected = await repo.choose(q, (proposal,), planning.objective, parent_intent_id="test")
    assert selected
    text = ResponseComposer(c.structure).compose(proposal.plan)[0].text
    assert await repo.record_sent(q, intention_id=selected.intention_id,
                                  response_message_id=700, created_at=q.timestamp, text=text)
    assert not await partner.clarification_allowed(event(3, "Он ушёл?"))
    answer = event(4, "Дракон", reply_to_message_id=700)
    await EventRepository(db).append(answer)
    assert not await repo.observe(answer)
    import json

    effect = await db.fetchone("SELECT * FROM language_meaning_consequences")
    assert effect["outcome_type"] == "requested_information_received"
    assert effect["external_value"] is None
    assert json.loads(effect["semantic_effect_json"])["value"] == "дракон"
    assert (await memory.retrieve_claims(event(5, "Дракон ушёл?"))).status == "Unknown"


async def test_gateway_identity_is_not_a_current_action(memory):
    memory, _, _, observe = memory
    naming = event(2, "как тебя зовут?", bot_identity={"id": 99, "username": "Дебильги"})
    await observe(naming)
    assert (await memory.retrieve_claims(naming)).status == "Known"
    assert (await memory.retrieve_claims(event(3, "что ты делаешь?"))).status == "Unknown"


def test_optional_response_acts_preserve_legacy_plan_keys():
    from debilgi.language.communication import MessagePlan, ReaderGoal
    from debilgi.language.productive_nlg import ClauseThought, ConceptLexeme

    plan = MessagePlan(ReaderGoal("respond", None, (), None, False),
                       ClauseThought(ConceptLexeme("читать", "читать", "verb"), ()))
    # Computed with the stage-3 implementation, not from a new hash algorithm.
    assert plan.key == "f158ac0152f06d7f4c49c412"


def test_dative_corpus_experiencer_cannot_become_an_agent_preference():
    planning = plan_from_evidence(ReplyContext(
        prompt="сегодня идёт дождь", knowledge_facts=("мне нравится дождь",),
    ))
    assert not any(p.plan.claim.predicate.lemma == "нравиться" for p in planning.proposals)


def test_unsupported_embedded_reference_does_not_request_an_unrelated_old_antecedent():
    q = event(1, "если бы рыба умела летать, что бы она делала")
    c = ReplyContext(prompt=q.meaning.text, input_meaning=q.meaning,
                     dialogue_messages=("Птица летает над водой.", "Рыба летает над домом."),
                     claim_result=Unknown(query_from_event(q)))
    assert decide_response_mode(c, plan_from_evidence(c)).mode != "clarification"


@pytest.mark.parametrize('text', ['так какую?', 'а че так?', 'потому что ты?'])
async def test_missing_ellipsis_asks_argument_from_observed_human_grammar(memory, text):
    memory, db, _, observe = memory
    from debilgi.language.dialogue_learning import DialogueStructureLearning

    await observe(event(1, 'О чём ты спрашиваешь?', actor=10, message_id=1))
    query = event(10, text)
    ctx = await context(memory, query, turn_form=query.turn_form)
    learning = DialogueStructureLearning(db, None)
    proposal = await learning.variable_plan(query, ctx)
    assert proposal and proposal.operation == 'ask_unbound_variable'
    assert not plan_issues(proposal.plan)
    composer = ResponseComposer(ctx.structure)
    surfaces = composer.compose(proposal.plan)
    assert surfaces and surfaces[0].text.endswith('?')
    assert 'спрашиваешь' in surfaces[0].text
    assert all(p.kind == 'human_question_grammar' for p in proposal.premises)
