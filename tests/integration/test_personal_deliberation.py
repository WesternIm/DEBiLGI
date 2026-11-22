"""Actual observed grammar, attributed experience and scoped personal preferences."""
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from debilgi.dataset.self_dataset import SelfDatasetRepository
from debilgi.events.normalizer import normalize_message_created
from debilgi.events.repository import EventRepository
from debilgi.experience.meaning_intents import MeaningIntentRepository
from debilgi.language.communication import ResponseComposer
from debilgi.language.discourse import DiscourseMemoryRepository
from debilgi.language.generative_planner import (
    compatible_response_decisions,
    decide_response_mode,
    plan_from_evidence,
    response_act_plan,
)
from debilgi.language.planning import ReplyContext
from debilgi.language.semantic_assimilation import SemanticAssimilationService
from debilgi.personality.expression import PersonalityExpression
from debilgi.storage.database import Database

NOW = datetime(2026, 11, 14, 14, tzinfo=UTC)


def event(n, text='ты крутой', *, actor=3, parent=None):
    return normalize_message_created(
        message_id=n, guild_id=1, channel_id=2, author_id=actor,
        created_at=NOW + timedelta(seconds=n), content=text, author_is_bot=False,
        bot_user_id=99, bot_mentioned=True, reply_to_message_id=parent,
    )


@pytest.fixture
async def personal(tmp_path):
    db = Database(tmp_path / 'personal.sqlite3', Path('migrations'))
    await db.initialize()
    learning = SemanticAssimilationService(db, SelfDatasetRepository(db)).dialogue_learning
    for i, answer in enumerate(('Ого!', 'Ну а то)', 'Да да!!!')):
        for e in (event(i * 3 + 1, actor=10 + i),
                  event(i * 3 + 2, answer, actor=20 + i, parent=i * 3 + 1)):
            await EventRepository(db).append(e)
            await DiscourseMemoryRepository(db).observe(e)
            await learning.observe(e)
    try:
        yield db, learning, MeaningIntentRepository(db, exploration_rate=0)
    finally:
        await db.close()


async def options(personal, e, *, mood=0, supplied_forms=None):
    db, learning, _ = personal
    await EventRepository(db).append(e)
    ctx = ReplyContext(
        prompt=e.meaning.text, input_meaning=e.meaning, turn_form=e.turn_form,
        source_event_id=e.event_id,
        personality=PersonalityExpression('neutral', .5, .5, .5, .5, .5, .5, .5, 20, 32),
        affect_features=(('emotion_valence', mood), ('mood_valence', mood)),
    )
    forms = supplied_forms or await learning.forms_for(e, ctx.prompt)
    planning = plan_from_evidence(ctx)
    decision = decide_response_mode(ctx, planning, observed_forms=forms, observed_reaction=True)
    composer, proposals, surfaces = ResponseComposer(ctx.structure), [], {}
    for option in compatible_response_decisions(decision, forms):
        matching = tuple(f for f in forms if f.component != 'core' or (
            f.meaning == option.target_meaning and (f.activation >= .4 and not f.hypothesis)
            == (option.confidence_level == 'answer')))
        p = response_act_plan(ctx, option, matching)
        if p is None:
            continue
        rendered = composer.compose(p.plan, exchange_forms=matching,
                                    source_answers=await learning.source_answers(e))
        if rendered:
            proposals.append(p)
            surfaces[p.plan.key] = rendered[0].text
    return decision, planning.objective, tuple(proposals), surfaces


async def select(personal, e, *, mood=0, supplied_forms=None):
    decision, objective, proposals, surfaces = await options(
        personal, e, mood=mood, supplied_forms=supplied_forms)
    selected = await personal[2].choose(e, proposals, objective, parent_intent_id='test',
                                         response_decision=decision)
    assert selected and selected.proposal.plan.key in surfaces
    return selected, surfaces[selected.proposal.plan.key]


async def test_mood_and_own_sent_memory_change_the_verified_answer(personal):
    warm, warm_text = await select(personal, event(80), mood=1)
    cautious, cautious_text = await select(personal, event(100), mood=-1)
    assert warm.proposal.plan.claim.predicate.lemma == 'agree'
    assert cautious.proposal.plan.claim.predicate.lemma == 'acknowledge'
    assert warm_text != cautious_text
    assert cautious.choice['decisive_source'] == 'taste'
    assert cautious.choice['considered'] >= 2
    # Prepared alternatives don't count as a memory of speaking.
    repeated, _ = await select(personal, event(110), mood=1)
    assert repeated.proposal.plan.claim.predicate.lemma == 'agree'
    await personal[2].record_sent(event(80), intention_id=warm.intention_id,
                                 response_message_id=900, created_at=NOW + timedelta(seconds=81),
                                 text=warm_text)
    remembered, text = await select(personal, event(120), mood=1)
    assert remembered.proposal.plan.claim.predicate.lemma == 'acknowledge'
    assert remembered.choice['decisive_source'] == 'memory' and text != warm_text
    assert warm.intention_id in remembered.choice['reasons'][0]['source_ids']
    stranger, _ = await select(personal, event(121, actor=4), mood=1)
    assert stranger.proposal.plan.claim.predicate.lemma == 'agree'


async def test_attributed_human_confirmation_can_outweigh_current_taste(personal):
    learned, text = await select(personal, event(80), mood=1)
    await personal[2].record_sent(event(80), intention_id=learned.intention_id,
                                 response_message_id=900, created_at=NOW + timedelta(seconds=81),
                                 text=text)
    feedback = event(82, 'Отлично', parent=900)
    await EventRepository(personal[0]).append(feedback)
    assert await personal[2].observe(feedback)
    selected, _ = await select(personal, event(100), mood=-1)
    assert selected.proposal.plan.claim.predicate.lemma == 'agree'
    assert selected.choice['decisive_source'] == 'experience'
    assert learned.intention_id in selected.choice['reasons'][0]['source_ids']
    transferred, _ = await select(personal, event(101, actor=4), mood=-1)
    assert transferred.proposal.plan.claim.predicate.lemma == 'agree'
    assert any(a['value_origin'] == 'shared_operation_experience'
               for a in transferred.alternatives)
    past, _ = await select(personal, event(79), mood=-1)
    assert past.proposal.plan.claim.predicate.lemma == 'acknowledge'


async def test_independent_coalition_changes_the_meaning_without_double_counting(personal):
    _, learning, _ = personal
    e = event(100)
    raw = await learning.forms_for(e, e.meaning.text)
    forms = tuple(replace(f, activation=.35 if f.meaning == 'agree' else .52)
                  if f.component == 'core' else f for f in raw if not f.recovery_only)
    selected, _ = await select(personal, e, supplied_forms=forms)
    assert selected.proposal.plan.claim.predicate.lemma == 'agree'
    assert selected.choice['decisive_source'] == 'coalition'
    assert selected.choice['not_first_by_score'] is False  # coalition is already in plan confidence
    members = selected.proposal.plan.deliberation['coalitions']['agree']['members']
    assert len(members) == 2
    assert set(members[0]['source_ids']).isdisjoint(members[1]['source_ids'])
    one = next(f for f in forms if f.component == 'core' and f.meaning == 'agree')
    reduced = tuple(f for f in forms if f.component != 'core' or f.meaning != 'agree') + (one, one)
    changed, _ = await select(personal, event(101), supplied_forms=reduced)
    assert changed.proposal.plan.claim.predicate.lemma == 'acknowledge'


async def test_failed_fact_surface_reactivates_a_nonasserting_verification(personal, monkeypatch):
    from debilgi.language import communication
    from debilgi.language.generative_planner import response_self_questions
    from debilgi.language.proposition_memory import PropositionMemory

    db = personal[0]
    source, query = event(30, 'я читаю книгу'), event(100, 'что я читаю?')
    await EventRepository(db).append(source)
    await EventRepository(db).append(query)
    memory = PropositionMemory(db)
    await memory.observe(source)
    known = await memory.retrieve_claims(query)
    assert known.status == 'Known'
    ctx = ReplyContext(prompt=query.meaning.text, input_meaning=query.meaning,
                       turn_form=query.turn_form, claim_result=known)
    planning = plan_from_evidence(ctx)
    decision = decide_response_mode(ctx, planning)
    ctx = replace(ctx, response_decision=decision)
    response_self_questions(ctx, proposals=planning.proposals)
    original = communication.realize_thought

    def refuse_statement(thought, *args, **kwargs):
        return () if thought.clause_kind == 'statement' else original(thought, *args, **kwargs)

    monkeypatch.setattr(communication, 'realize_thought', refuse_statement)
    composer = ResponseComposer(ctx.structure)
    plan = planning.proposals[0].plan
    result = composer.compose(plan)
    assert result and result[0].text.endswith('?')
    assert plan.review['questions'][0]['slot'] == 'realization'
    assert plan.review['reactivation']['level'] == 'hypothesis'
    assert [c['passed'] for c in plan.review['considered']] == [False, True]
    expressed = composer.metadata_for(result[0].text)['message_plan']
    assert expressed['claim']['evidence'] == plan.claim.evidence
    assert expressed['claim']['clause_kind'] == 'question'
    assert expressed['hypothesis'] == (('kind', 'verify_known_content'),)
    from debilgi.cognition.dialogue_tasks import task_candidate_metadata

    metadata = {**composer.metadata_for(result[0].text),
                'premises': planning.proposals[0].premises, 'locally_composed': True}
    checked = task_candidate_metadata(None, result[0].text, metadata)
    assert checked['genuine_clarification'] and checked['hypothetical_assertions'] == 0
    contradictory = {**metadata, 'message_plan': {**expressed, 'claim': {
        **expressed['claim'], 'negated': not expressed['claim']['negated']}}}
    assert not task_candidate_metadata(None, result[0].text, contradictory).get(
        'genuine_clarification')
    selected = await personal[2].choose(query, planning.proposals, planning.objective,
                                       parent_intent_id='test', query_result=known,
                                       response_decision=decision)
    assert selected
    await personal[2].record_sent(query, intention_id=selected.intention_id,
        response_message_id=900, created_at=NOW + timedelta(seconds=101), text=result[0].text)
    sent = await db.fetchone('SELECT plan_json,operation FROM language_meaning_intentions '
                            'WHERE intention_id=?', (selected.intention_id,))
    assert json.loads(sent['plan_json'])['claim']['clause_kind'] == 'question'
    assert sent['operation'] == 'verify_known_content'
    feedback = event(102, 'Да', parent=900)
    await EventRepository(db).append(feedback)
    assert not await personal[2].observe(feedback)
    effect = await db.fetchone(
        'SELECT outcome_type,external_value FROM language_meaning_consequences'
    )
    assert effect['outcome_type'] == 'requested_information_received'
    assert effect['external_value'] is None
    row = await db.fetchone('SELECT reward FROM language_meaning_intentions WHERE intention_id=?',
                           (selected.intention_id,))
    assert row['reward'] is None  # an answer to the actual question is not approval


async def test_human_dossier_changes_topic_without_importing_a_biography(personal):
    from debilgi.language.communication import MessagePlan, ReaderGoal
    from debilgi.language.generative_planner import DerivedPlan, DialogueObjective, ResponseDecision
    from debilgi.language.productive_nlg import (
        ClauseThought,
        ConceptArgument,
        ConceptLexeme,
        SemanticDependency,
    )

    db, _, repo = personal
    for e in (event(40, 'я люблю музыку', actor=12), event(41, 'я люблю книги', actor=13)):
        await EventRepository(db).append(e)
        await DiscourseMemoryRepository(db).observe(e)
    query = event(100, 'эй')
    await EventRepository(db).append(query)
    proposals = tuple(DerivedPlan(MessagePlan(ReaderGoal('respond', None, (), None, True),
        ClauseThought(ConceptLexeme('любить', 'любить', 'verb'), (
            SemanticDependency('subject', ConceptArgument(ConceptLexeme('ты', 'ты', 'pronoun'))),
            SemanticDependency('direct_object', ConceptArgument(ConceptLexeme(word, word, 'noun'),
                                                                grammatical_case='accusative')),
        ), clause_kind='question', evidence=('language:human-observations',))),
        'request_missing_argument', ()) for word in ('музыка', 'книга'))
    decision = ResponseDecision('free_form', 'observed', 'greeting', 1, 1, None,
        selection_context={'enabled': True, 'personality': {'memory_attachment': 1,
                           'novelty_seeking': 0, 'novelty_drive': 0}, 'affect': {}})
    objective = DialogueObjective('respond', None, None, (), ())
    before = await repo.choose(query, proposals, objective, parent_intent_id='test',
                               response_decision=decision)
    assert before.proposal == proposals[0]
    remembered = event(50, 'я люблю книги')
    await EventRepository(db).append(remembered)
    await DiscourseMemoryRepository(db).observe(remembered)
    after = await repo.choose(query, proposals, objective, parent_intent_id='test',
                              response_decision=decision)
    assert after.proposal == proposals[1] and after.choice['decisive_source'] == 'memory'
    assert remembered.event_id in after.choice['features'][1]['memory_sources']
    # The selected content is still an unasserted question.
    composer = ResponseComposer(query.meaning)
    rendered = composer.compose(after.proposal.plan)
    assert rendered and rendered[0].text.endswith('?')
    assert after.proposal.plan.claim.clause_kind == 'question'
