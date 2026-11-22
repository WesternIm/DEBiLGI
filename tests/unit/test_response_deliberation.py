"""Bounded reasoning retains the original response and independent evidence."""

from dataclasses import replace

import pytest

from debilgi.discord.delivery import reply_pacing_seconds
from debilgi.language.dialogue_learning import (
    ObservedDialogueForm,
    _reaction_parts,
    ellipsis_variable,
    learned_evaluation,
    learned_evaluation_exchange,
)
from debilgi.language.generative_planner import ResponseDecision, compatible_response_decisions
from debilgi.language.morphology import DEFAULT_MORPHOLOGY


@pytest.mark.parametrize('text,lemma,part', [
    ('Неее.', 'не', 'particle'), ('у неё', 'она', 'pronoun'),
    ('у неее', 'она', 'pronoun'), ('Нееет.', 'нет', 'predicate'),
])
def test_elongation_preserves_contextual_alternatives(text, lemma, part):
    readings = [r for r in DEFAULT_MORPHOLOGY.analyze_sentence(text).readings if r]
    reading = readings[-1]
    assert reading.lemma == lemma
    if lemma != 'нет':
        assert reading.part_of_speech == part


@pytest.mark.parametrize('prefix', ['ого', 'ну', 'бля'])
@pytest.mark.parametrize('core', ['да', 'а то', 'вот тоже верно'])
def test_optional_prefix_does_not_relabel_agreement(prefix, core):
    forms = _reaction_parts('ну ты бля крутой', f'{prefix} {core}')
    assert any(f['meaning'] == 'agree' and f['component'] == 'core' for f in forms)
    # An interjection on its own remains acknowledgment.
    if prefix == 'ого':
        assert any(f['meaning'] == 'acknowledge' for f in _reaction_parts('ты крутой', prefix))


def test_causal_conjunction_is_not_an_object_question():
    assert ellipsis_variable('потому что ты?')['role'] == 'cause_predicate'
    assert ellipsis_variable('а че так?')['role'] == 'cause'
    assert ellipsis_variable('так какую?')['features'] == {
        'case': 'accusative', 'gender': 'feminine', 'number': 'singular'}
    assert ellipsis_variable('какую книгу ты читал?') is None


def test_evaluation_requires_independent_explicit_human_evidence():
    def rows(n, people=2, text='лошара это плохо'):
        return [dict(prompt=text, prompt_actor_id=i % people, prompt_event_id=str(i))
                for i in range(n)]

    assert learned_evaluation('лошара', rows(3))[0] == -1
    assert learned_evaluation('лошара', rows(2))[0] is None
    assert learned_evaluation('лошара', rows(100, people=1))[0] is None
    assert learned_evaluation('лошара', rows(100, text='я гулял это плохо'))[0] is None
    assert learned_evaluation('лошара', [*rows(3), dict(
        prompt='лошара это хорошо', prompt_actor_id=3, prompt_event_id='extra')])[0] is None


def test_evaluation_family_needs_reciprocal_human_evidence_not_reply_sentiment():
    def rows(n=3, *, people=2, prompt='лох', answer='ты сам плохой'):
        return [dict(prompt=prompt, answer=answer, prompt_actor_id=i % people,
                     prompt_event_id=str(i), answer_event_id='answer:' + str(i))
                for i in range(n)]

    sign, target, sources = learned_evaluation_exchange('лошара', rows())
    assert (sign, target) == (-1, 'addressee')
    assert len(sources) == 6  # both sides of every human exchange
    for insufficient in (rows(2), rows(100, people=1), rows(answer='это плохо'),
                         rows(answer='ахаха'),
                         rows(prompt='он лох'), rows(prompt='я лох'), rows(prompt='ты не лох')):
        assert learned_evaluation_exchange('лошара', insufficient)[:2] == (None, None)
    assert learned_evaluation_exchange('он лошара', rows())[:2] == (None, None)
    assert learned_evaluation_exchange('лошара', rows(answer='сам такой'))[:2] == (
        None, 'addressee')
    contrary = dict(prompt='ты лох', answer='ты сам хороший', prompt_actor_id=3,
                    prompt_event_id='contrary', answer_event_id='contrary:answer')
    assert learned_evaluation_exchange('лошара', [*rows(), contrary])[:2] == (None, 'addressee')


def test_candidate_budget_keeps_primary_and_filters_hypotheses():
    decision = ResponseDecision('free_form', 'activated', 'evaluation', 1, 1, None,
                                target_meaning='agree', activation=0.6)
    def form(meaning, strength, hypothesis=()):
        return ObservedDialogueForm((), 'fragment', '', '', ('human:1',), 1, strength,
                                    meaning=meaning, activation=strength,
                                    hypothesis=hypothesis)
    forms = (form('agree', 0.6), form('acknowledge', 0.5),
             form('appraise', 0.3, (('predicate', 'гулять'),)))
    choices = tuple(compatible_response_decisions(decision, forms, hypothesis_allowed=False))
    assert [c.target_meaning for c in choices] == ['agree', 'acknowledge']
    assert tuple(compatible_response_decisions(decision, forms, limit=1)) == (decision,)
    assert tuple(compatible_response_decisions(replace(decision, mode='answer'), forms)) == (
        replace(decision, mode='answer'),)
    reserve = replace(forms[1], recovery_only=True, component='reserve')
    ready = replace(decision, selection_context={'considered': [{'passed': True}]})
    assert tuple(compatible_response_decisions(ready, (reserve,))) == (ready,)
    screen = ready.selection_context['scope_candidates'][0]
    assert screen['source_ids'] == ('human:1',) and not screen['surface_checked']
    assert screen['reason'] == 'lower_level_not_needed'


def test_pacing_credits_reasoning_and_bounds_total_time():
    assert reply_pacing_seconds('Привет', 2) < reply_pacing_seconds('Привет' * 12, 2)
    assert reply_pacing_seconds('Привет', 2) < reply_pacing_seconds('Привет', 8)
    assert reply_pacing_seconds('Привет', 8, elapsed=20) == 0
    assert 0 < reply_pacing_seconds('x' * 10000, 1000) <= 6
    target = reply_pacing_seconds('Ответ', 4)
    assert reply_pacing_seconds('Ответ', 4, elapsed=target / 2) == target / 2


def test_deliberation_coverage_gate_rejects_lost_deliveries():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        'coverage_probe', Path('scripts/probe_response_contract.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    delivered = [dict(scenario='same', turn=1, prompt='так какую?', sent_text='О чём?')]
    silent = [{**delivered[0], 'sent_text': None}]
    probe.verify_deliberation_coverage(delivered, silent)
    with pytest.raises(AssertionError, match='consumed'):
        probe.verify_deliberation_coverage(silent, delivered)


def test_hypothesis_budget_does_not_erase_a_supported_answer():
    from debilgi.language.dialogue_acts import analyze_turn_form
    from debilgi.language.generative_planner import decide_response_mode, plan_from_evidence
    from debilgi.language.planning import ReplyContext

    ctx = ReplyContext(prompt='ты крутой', turn_form=analyze_turn_form('ты крутой', addressed=True))
    def form(meaning, activation, hypothesis=()):
        return ObservedDialogueForm((), 'fragment', '', '', ('human:1',), 1, activation,
                                    meaning=meaning, activation=activation, hypothesis=hypothesis)
    forms = (form('appraise', 0.9, (('quality', 'unknown'),)), form('acknowledge', 0.5))
    decision = decide_response_mode(ctx, plan_from_evidence(ctx), observed_reaction=True,
                                    observed_forms=forms, hypothesis_allowed=False)
    assert decision.mode == 'free_form' and decision.target_meaning == 'acknowledge'
    assert decision.confidence_level == 'answer'


def test_own_speech_uses_the_delivered_meaning_instead_of_the_input_act():
    from dataclasses import asdict

    from debilgi.language.dialogue_learning import own_turn_plan
    from debilgi.language.planning import ReplyContext
    from debilgi.language.productive_nlg import ClauseThought, ConceptLexeme

    claim = ClauseThought(
        ConceptLexeme('reciprocal_question', 'reciprocal_question', 'dialogue_act'),
        (), predicate_class='dialogue_act', clause_kind='fragment')
    prior = {'plan': {'claim': asdict(claim)}, 'prompt_event_id': 'prior',
             'receipt_id': 'receipt:1:2', 'response_text': 'как тебя зовут?',
             'decision': {'act': 'greeting', 'target_meaning': 'reciprocal_question'},
             'operations': {'greeting': ('приветствовать', 'human:grammar')}}
    ctx = ReplyContext(prompt='и к чему это?', prior_bot_turn=prior)
    result = own_turn_plan(ctx, None)
    assert result.plan.claim.predicate.lemma == 'сказать'
    assert 'как тебя зовут?' in result.plan.claim.dependencies[1].argument.head.surface
    # A historical fixture is a real receipt: its actual greeting still counts.
    ctx = replace(ctx, prior_bot_turn={**prior, 'response_text': 'хай!'})
    assert own_turn_plan(ctx, None).plan.claim.predicate.lemma == 'приветствовать'
