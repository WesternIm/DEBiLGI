from dataclasses import asdict, replace

import pytest

from debilgi.language.communication import ResponseComposer
from debilgi.language.generative_planner import plan_from_evidence, statement_thought
from debilgi.language.planning import ReplyContext
from debilgi.language.semantic_graph import (
    MeaningGoal,
    extract_meanings,
    goal_effect,
    infer_meanings,
    meaning_from_data,
    meaning_query,
    parse_proposition,
    proposition_key,
)
from debilgi.language.structured_utterance import analyze_structured_utterance

DEFINITION = (
    "Быть выключенным для тебя, это когда ты не можешь отвечать мне, даже думать ты не можешь"
)


def records(text, event_id="human"):
    result = extract_meanings(text, event_id)
    assert result, text
    return result


def proposition(text):
    result = parse_proposition(text)
    assert result, text
    return result


def rendered(context):
    result = plan_from_evidence(context)
    composer = ResponseComposer(analyze_structured_utterance(context.prompt))
    return result, tuple(
        surface.text for p in result.proposals for surface in composer.compose(p.plan)
    )


def test_actual_definition_retains_both_actions_modality_recipient_and_source():
    learned = records(DEFINITION)
    assert len(learned) == 1
    assert learned[0].kind == "definition"
    assert len(learned[0].conclusions) == 2
    assert all(t.negated and t.predicate.lemma == "мочь" for t in learned[0].conclusions)
    first = {d.role: d.argument.head.lemma for d in learned[0].conclusions[0].dependencies}
    assert first == {"subject": "я", "action_complement": "отвечать", "indirect_object": "ты"}
    assert infer_meanings(learned) == ()  # A definition doesn't assert the current state.
    assert meaning_from_data(asdict(learned[0])) == learned[0]
    result, text = rendered(ReplyContext(prompt=DEFINITION, source_event_id="definition"))
    assert result.proposals[0].operation == "learn_semantic_relation"
    assert text == ("Если я выключен, я не могу отвечать тебе и я не могу думать.",)
    assert result.proposals[0].premises[0].thought.evidence == ("definition",)


@pytest.mark.parametrize(
    "text",
    [
        "Ты выключен?",
        "Если ты выключен?",
        "Возможно ты выключен.",
        "Ты можешь отвечать или думать.",
        "Ты можешь только отвечать.",
        "Ты мог бы отвечать.",
        "Почему ты не можешь отвечать?",
        "«Ты выключен».",
        "Если ты выключен, ты можешь не отвечать.",
    ],
)
def test_queries_quotes_uncertainty_and_unresolved_scope_do_not_become_facts(text):
    assert extract_meanings(text, "input") == ()


def test_outer_negation_is_not_negation_of_the_embedded_action():
    assert proposition("Я не могу отвечать.").negated
    assert parse_proposition("Я могу не отвечать.") is None
    assert parse_proposition("Я хочу не говорить.") is None


@pytest.mark.parametrize(
    "rule,state,effect",
    [
        ("Если бот выключен, он не может говорить.", "Бот выключен.", "Бот не может говорить."),
        ("Если кот голоден, кот ест рыбу.", "Кот голоден.", "Кот ест рыбу."),
        ("Если робот заряжен, робот может двигаться.", "Робот заряжен.", "Робот может двигаться."),
    ],
)
def test_same_inference_engine_works_for_different_learned_states(rule, state, effect):
    learned = records(rule, "rule")
    assert infer_meanings(learned) == ()
    facts = records(state, "state")
    proofs = infer_meanings((*learned, *facts))
    found = next(
        p for p in proofs if proposition_key(p.thought) == proposition_key(proposition(effect))
    )
    assert set(found.source_event_ids) == {"rule", "state"}
    assert found.rule_ids == (learned[0].meaning_id,)
    assert infer_meanings(facts)[0].rule_ids == ()


def test_condition_is_not_reversible_but_definition_requires_all_reverse_premises():
    condition = records("Если бот выключен, бот не может говорить.")
    consequence = records("Бот не может говорить.", "actual")
    assert not any(
        p.thought.predicate_class == "copular" for p in infer_meanings((*condition, *consequence))
    )
    definition = records(DEFINITION, "definition")
    partial = records("Ты не можешь думать.", "thinking")
    assert not any(
        p.thought.predicate_class == "copular" for p in infer_meanings((*definition, *partial))
    )
    complete = (*partial, *records("Ты не можешь отвечать мне.", "replying"))
    reversed_state = next(
        p
        for p in infer_meanings((*definition, *complete))
        if p.thought.predicate_class == "copular"
    )
    assert set(reversed_state.source_event_ids) == {"definition", "thinking", "replying"}


def test_capability_is_not_actual_action_and_recipient_is_not_changed():
    learned = (*records(DEFINITION), *records("Ты выключен.", "state"))
    proofs = infer_meanings(learned)
    assert not any(p.thought.predicate.lemma == "отвечать" for p in proofs)
    result, _ = rendered(ReplyContext(prompt="Ты можешь отвечать Маше?", meaning_records=learned))
    assert result.proposals == ()
    result, text = rendered(ReplyContext(prompt="Ты можешь отвечать мне?", meaning_records=learned))
    assert result.proposals[0].operation == "infer_from_learned_relations"
    assert text == ("Я не могу отвечать тебе.",)


def test_multi_step_proof_keeps_every_source_and_does_not_invent_premises():
    learned = (
        *records("Если бот выключен, бот не может говорить.", "one"),
        *records("Если бот не может говорить, бот молчит.", "two"),
        *records("Бот выключен.", "state"),
    )
    found = next(p for p in infer_meanings(learned) if p.thought.predicate.lemma == "молчать")
    assert set(found.source_event_ids) == {"one", "two", "state"}
    assert len(found.rule_ids) == 2
    assert infer_meanings(learned[:-1]) == ()


def test_conflicting_rules_retract_their_downstream_conclusions():
    learned = (
        *records("Если бот выключен, бот не может говорить.", "no"),
        *records("Если бот выключен, бот может говорить.", "yes"),
        *records("Если бот может говорить, бот поёт.", "downstream"),
        *records("Бот выключен.", "state"),
    )
    proofs = infer_meanings(learned)
    assert not any(p.thought.predicate.lemma in {"мочь", "петь"} for p in proofs)


def test_hypothetical_consequences_remain_inside_the_condition():
    context = ReplyContext(
        prompt="Что будет, если ты выключен?",
        source_event_id="question",
        meaning_records=records(DEFINITION, "definition"),
    )
    result, text = rendered(context)
    assert text == ("Если я выключен, я не могу отвечать тебе и я не могу думать.",)
    assert result.proposals[0].plan.condition is not None
    assert len(result.proposals[0].plan.additional_claims) == 1
    assert any(p.kind == "hypothesis" for p in result.proposals[0].premises)
    assert infer_meanings(context.meaning_records) == ()
    assert meaning_query(context.prompt).kind == "consequence"


def test_state_preference_depends_on_learned_consequences_and_an_active_goal():
    goal = MeaningGoal(proposition("Ты отвечаешь мне."), "reply_intent:question")
    context = ReplyContext(
        prompt="Ты хочешь быть выключенным?",
        source_event_id="question",
        meaning_records=records(DEFINITION),
        meaning_goals=(goal,),
    )
    result, text = rendered(context)
    assert result.proposals[0].operation == "appraise_learned_consequence"
    assert text == (
        "Я не хочу быть выключенным, потому что если я выключен, я не могу отвечать тебе.",
    )
    assert any(p.kind == "operational_goal" for p in result.proposals[0].premises)
    assert result.proposals[0].plan.reason_condition is not None
    assert rendered(replace(context, meaning_goals=()))[0].proposals == ()
    assert rendered(replace(context, meaning_records=()))[0].proposals == ()
    changed = replace(
        context,
        meaning_records=records("Быть выключенным для тебя — это когда ты можешь отвечать мне."),
    )
    assert not rendered(changed)[0].proposals[0].plan.claim.negated
    assert goal_effect(proposition("Ты не можешь отвечать Маше."), goal) == 0


def test_appraisal_has_no_shutdown_specific_branch():
    goal = MeaningGoal(proposition("Робот двигается."), "task:move")
    context = ReplyContext(
        prompt="Робот хочет быть заряженным?",
        meaning_records=records("Если робот заряжен, робот может двигаться."),
        meaning_goals=(goal,),
    )
    result, text = rendered(context)
    assert result.proposals[0].operation == "appraise_learned_consequence"
    assert text == (
        "Робот хочет быть заряженным, потому что если робот заряжен, робот может двигаться.",
    )


def test_full_property_and_short_predication_are_realized_without_placeholders():
    result, text = rendered(ReplyContext(prompt="Быть занятым — это когда не можешь говорить."))
    assert result.proposals[0].operation == "learn_semantic_relation"
    assert text == ("Если кто-то занят, кто-то не может говорить.",)
    assert "@subject" not in text[0]


def test_full_and_short_state_forms_bind_to_the_same_learned_definition():
    context = ReplyContext(
        prompt="Что будет, если ты занят?",
        meaning_records=records("Быть занятым — это когда не можешь говорить."),
    )
    result, text = rendered(context)
    assert result.proposals[0].operation == "infer_conditional_consequences"
    assert text == ("Если я занят, я не могу говорить.",)


@pytest.mark.parametrize("prompt", ["Что будет, если я тебя выключу?", "Если я тебя выключу?"])
def test_completed_action_uses_a_learned_passive_state_and_keeps_its_actor(prompt):
    context = ReplyContext(
        prompt=prompt,
        source_event_id="hypothesis",
        meaning_records=records(DEFINITION, "definition"),
    )
    result, text = rendered(context)
    assert result.proposals[0].operation == "infer_conditional_consequences"
    assert text == (
        "Если ты выключишь меня, я выключен и я не могу отвечать тебе и я не могу думать.",
    )
    assert any(p.kind == "learned_event_result" for p in result.proposals[0].premises)
    assert rendered(replace(context, meaning_records=()))[0].proposals == ()


def test_incomplete_or_negated_action_does_not_assert_its_passive_result():
    for prompt in ("Если я тебя не выключу?", "Если я выключаю тебя?"):
        result = plan_from_evidence(
            ReplyContext(prompt=prompt, meaning_records=records(DEFINITION))
        )
        assert not any(p.operation == "infer_conditional_consequences" for p in result.proposals)


def test_every_conjunct_in_a_condition_is_required():
    learned = records("Если бот заряжен и бот свободен, бот может двигаться.", "rule")
    facts = records("Бот заряжен.", "charged")
    assert not any(p.thought.predicate.lemma == "мочь" for p in infer_meanings((*learned, *facts)))
    complete = (*learned, *facts, *records("Бот свободен.", "free"))
    capability = next(p for p in infer_meanings(complete) if p.thought.predicate.lemma == "мочь")
    assert set(capability.source_event_ids) == {"rule", "charged", "free"}


@pytest.mark.parametrize(
    "report,question,expected",
    [
        ("Гигик тупой.", "Кто тупой?", "Гигик тупой."),
        ("Маша голодна.", "Кто голоден?", "Маша голодна."),
        ("Маша читает книгу.", "Кто читает книгу?", "Маша читает книгу."),
    ],
)
def test_actor_is_bound_to_the_learned_property_or_action(report, question, expected):
    context = ReplyContext(prompt=question, meaning_records=records(report, "human"))
    result, text = rendered(context)
    assert result.proposals[0].operation == "infer_from_learned_relations"
    assert text == (expected,)
    assert result.proposals[0].plan.claim.evidence == ("human",)
    assert rendered(replace(context, meaning_records=()))[0].proposals == ()


def test_active_goal_answers_only_its_own_action_and_participant():
    goal = MeaningGoal(proposition("Ты отвечаешь мне."), "reply_intent:request")
    context = ReplyContext(prompt="Ты не хочешь отвечать мне?", meaning_goals=(goal,))
    result, text = rendered(context)
    assert result.proposals[0].operation == "express_active_goal"
    assert text == ("Я хочу отвечать тебе.",)
    assert result.proposals[0].premises[0].kind == "operational_goal"
    assert rendered(replace(context, meaning_goals=()))[0].proposals == ()
    assert rendered(replace(context, prompt="Ты хочешь отвечать Маше?"))[0].proposals == ()
    assert rendered(replace(context, prompt="Ты хочешь петь?"))[0].proposals == ()
    assert meaning_query("Почему ты хочешь отвечать мне?").kind == "cause"


def test_open_next_step_is_composed_from_active_goal_instead_of_a_stored_reply():
    action = statement_thought("Я отвечаю тебе.")
    assert action is not None
    goal = MeaningGoal(action, "learned:replying", 1.0)
    context = ReplyContext(prompt="ну и что теперь", meaning_goals=(goal,))

    result, text = rendered(context)

    assert result.proposals[0].operation == "continue_active_goal"
    assert text == ("Я отвечаю тебе теперь.",)
    assert result.proposals[0].premises[0].kind == "operational_goal"
    assert result.proposals[0].premises[0].text == ""
    assert rendered(replace(context, meaning_goals=()))[0].proposals == ()
