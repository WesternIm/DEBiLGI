"""Night variants must repair the requested meaning, not just fit a form."""
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from debilgi.cognition.continuous import ContinuousCognitionService as Sleep
from debilgi.events.models import Event, EventType
from debilgi.language.communication import ResponseComposer
from debilgi.language.dialogue_learning import DialogueStructureLearning
from debilgi.language.generative_planner import ReplyContext


def event(text):
    return Event('quality', 1, 2, 3, 9, EventType.MESSAGE_CREATED,
                 datetime(2026, 11, 17, tzinfo=UTC), {'content': text}, 'discord')


def candidate(text, **metadata):
    return {'text': text, 'metadata': metadata}


@pytest.mark.parametrize(('prompt', 'text', 'metadata', 'reason'), [
    ('Ты плаваешь?', 'Чем плаваешь?', {}, 'predicate_echo_question'),
    ('Ты танцуешь?', 'Я не знаю.', {'communicative_status': 'express_limit',
                                   'sleep_previous_text': 'Я не знаю, танцую ли я.'},
     'unknown_instead_of_unknown'),
    ('Ты танцуешь?', 'Возможно, ты танцуешь.', {'sleep_opinion': True}, 'wrong_addressee'),
    ('Ты танцуешь?', 'Возможно, танцуешь.', {'sleep_opinion': True}, 'wrong_addressee'),
    ('Ты хочешь отдыхать?', 'Кто их не хочет?', {}, 'unbound_reference'),
    ('Что ты хотел?', 'Я сказал «кто их не хочет?».',
     {'derivation': 'retrieve_delivered_act', 'reasoning_evidence': ['receipt:2:10']},
     'quote_instead_of_explanation'),
    ('Что ты читаешь?', 'Как дела?', {}, 'unrelated_counterquestion'),
])
def test_rejects_false_repairs(prompt, text, metadata, reason):
    assert Sleep.sleep_rejection(event(prompt), candidate(text, **metadata)) == reason


def row(n, actor, prompt, answer):
    return {'prompt': prompt, 'answer': answer, 'actor_id': actor,
            'answer_event_id': str(n), 'observed_at': '2026-11-16'}


def test_human_answer_meaning_and_one_vote_per_author():
    rows = [row(n, 1, 'Ты хочешь читать?', 'Нет, я не хочу читать.') for n in range(12)]
    rows += [row(13, 2, 'Ты хочешь читать?', 'Да, я хочу читать.'),
             row(14, 3, 'Ты хочешь читать?', 'Я хочу читать.'),
             row(15, 4, 'Ты умеешь читать?', 'Я умею читать.'),
             row(16, 5, 'Ты хочешь читать?', 'Наверное.')]
    context = ReplyContext(prompt='Ты хочешь плавать?')
    plans = DialogueStructureLearning._sleep_answer_plans(context, rows, {})
    assert plans
    proposal, metadata = plans[0]
    assert metadata['human_answer_people'] == 2
    assert not proposal.plan.claim.negated
    assert proposal.plan.claim.predicate.lemma == 'хотеть'
    assert 'плавать' in {d.argument.head.lemma for d in proposal.plan.claim.dependencies}
    composer = ResponseComposer(context.structure)
    surfaces = composer.compose(proposal.plan)
    assert surfaces and all(s.text.startswith('Наверное,') for s in surfaces)
    for surface in surfaces:
        assert Sleep.sleep_surface_allowed(event(context.prompt),
                                           candidate(surface.text, **metadata))
    # Removing the epistemic source cannot turn an opinion into a licensed fact.
    broken = replace(proposal.plan, claim=replace(proposal.plan.claim, evidence=()))
    assert not ResponseComposer(context.structure).compose(broken)


async def test_reask_explains_missing_reference_instead_of_quoting_it():
    context = ReplyContext(prompt='Что не хочет?', prior_bot_turn={
        'response_text': 'Кто их не хочет?', 'receipt_id': 'receipt:2:10', 'plan': {}})
    learning = DialogueStructureLearning(None, None)
    proposal, metadata = await learning.sleep_repair_plan(event(context.prompt), context, ())
    assert metadata['repair_reference_slot'] == 'их'
    assert metadata['repair_reference_binding'] is None
    composer = ResponseComposer(context.structure)
    surfaces = composer.compose(proposal.plan)
    assert surfaces
    assert all('«' not in s.text and 'сказать' in s.text for s in surfaces)
    for s in surfaces:
        assert Sleep.sleep_surface_allowed(event(context.prompt), candidate(s.text,
            **metadata, communicative_status='express_limit',
            reasoning_evidence=['receipt:2:10']))


def test_checked_low_weight_trial_can_beat_unknown_without_becoming_fact():
    from debilgi.language.arbitration import (
        ArbitrationProfile,
        ResponseArbitrator,
        ResponseCandidate,
    )

    context = ReplyContext(prompt='Ты хочешь плавать?')
    rows = [row(1, 10, 'Хочешь читать?', 'Хочу!'),
            row(2, 11, 'Как дела?', 'Наверное.')]
    proposal, metadata = DialogueStructureLearning._sleep_answer_plans(context, rows, {})[0]
    composer = ResponseComposer(context.structure)
    surface = composer.compose(proposal.plan)[0]
    metadata.update(composer.metadata_for(surface.text))
    trial = ResponseCandidate.create(
        event_id='e', source='semantic_reasoning', renderer='sleep_replay:human_answer_crossover',
        text=surface.text, speech_act='opinion', source_confidence=.45,
        metadata={**metadata, 'locally_composed': True, 'sleep_hypothesis_id': 'h',
                  'epistemic_status': 'hypothesis', 'sleep_weight': .5,
                  'reasoning_evidence': ['1', '2']})
    old = ResponseCandidate.create(
        event_id='e', source='semantic_reasoning', renderer='old',
        text='Я не знаю, хочу ли я плавать.', speech_act='brief_statement', source_confidence=.9,
        metadata={'communicative_status': 'express_limit'})
    decision = ResponseArbitrator().decide(
        event_id='e', prompt=context.prompt, focus_concepts=(), candidates=(old, trial),
        recent_replies=(), profile=ArbitrationProfile.baseline(1))
    assert decision.winner.candidate_id == trial.candidate_id
    score = next(s for s in decision.scores if s.candidate.candidate_id == trial.candidate_id)
    assert score.answer_status == 'hypothesis_answer'
    assert decision.winner.source_confidence <= .5
