from dataclasses import replace
from unittest.mock import Mock

import pytest

import debilgi.language.communication as communication
from debilgi.language.communication import (
    MessagePlan,
    PendingText,
    ResponseComposer,
    plan_issues,
    reader_goal,
    surface_issues,
)
from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    SemanticDependency,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def thought(subject="маша", obj="петя", *, negated=False):
    return ClauseThought(
        ConceptLexeme("видеть", "видеть", "verb"),
        tuple(SemanticDependency(role, ConceptArgument(ConceptLexeme(lemma, lemma, "noun")))
              for role, lemma in (("subject", subject), ("direct_object", obj))),
        negated=negated, confidence=.85, evidence=("event:observed",),
    )


def test_content_can_be_compared_and_selected_without_calling_nlg(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    composer.deferred = True
    nlg = Mock(wraps=communication.realize_thought)
    monkeypatch.setattr(communication, "realize_thought", nlg)
    requested = composer.express(thought())[0].text
    unrelated = composer.express(thought("коля", "дима"))[0].text
    assert isinstance(requested, PendingText)
    # Fluency preferences and an unrelated producer's confidence cannot choose
    # the topic. No candidate has received words at this stage.
    assert composer.priority(requested, {}, .5) > composer.priority(unrelated, {}, .99)
    nlg.assert_not_called()
    result = composer.realize(requested)
    assert result
    assert nlg.call_count == 1
    metadata = composer.metadata_for(result[0].text)
    assert metadata["message_plan_key"] == requested.plan.key
    assert metadata["reader_checked"]


@pytest.mark.parametrize("text,issue", [
    ("Петя видит Машу.", "swapped_actor_and_object"),
    ("Маша не видит Петю.", "changed_negation"),
    ("Маша видит.", "missing_direct_object"),
    ("Маша видит Диму.", "missing_direct_object"),
    ("Маша слышит Петю.", "missing_predicate"),
])
def test_reader_check_catches_lost_or_changed_propositions(text, issue):
    assert issue in surface_issues(thought(), text)


def test_word_order_can_change_without_changing_intended_roles():
    assert not surface_issues(thought(), "Петю видит Маша.")


def test_actor_object_and_negation_are_part_of_message_identity():
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    plan = MessagePlan(composer.goal, thought())
    assert plan.key != replace(plan, claim=thought("петя", "маша")).key
    assert plan.key != replace(plan, claim=thought(negated=True)).key
    assert plan.key == replace(plan, claim=replace(plan.claim, confidence=.1)).key


def test_content_selection_compares_roles_not_a_bag_of_identical_words():
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    composer.deferred = True
    correct = composer.express(thought())[0].text
    reversed_roles = composer.express(thought("петя", "маша"))[0].text
    assert composer.priority(correct, {}, .5) > composer.priority(reversed_roles, {}, .99)


def test_known_question_arguments_cannot_be_swapped_before_generation():
    composer = ResponseComposer(analyze_structured_utterance("Кого видит Маша?"))
    assert "question_argument_changed_role" in plan_issues(
        MessagePlan(composer.goal, thought("петя", "маша")),
    )


@pytest.mark.parametrize("subject", ["погода", "машина", "музыка"])
def test_nominal_question_subject_cannot_become_an_assessed_quality(subject):
    composer = ResponseComposer(analyze_structured_utterance(f"Какая сегодня {subject}?"))
    malformed = ClauseThought(
        ConceptLexeme("считать", "считать", "verb"),
        (
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("сегодня", "сегодня", "adverb"),
            )),
            SemanticDependency("predicate_complement", ConceptArgument(
                ConceptLexeme(subject, subject, "noun"),
            )),
        ), confidence=.95,
    )
    assert "question_subject_became_property" in plan_issues(MessagePlan(composer.goal, malformed))


def test_incomplete_meaning_is_rejected_before_surface_generation(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Кто видит Петю?"))
    malformed = replace(thought(), dependencies=thought().dependencies[1:])
    plan = MessagePlan(composer.goal, malformed)
    assert "missing_subject" in plan_issues(plan)
    nlg = Mock()
    monkeypatch.setattr(communication, "realize_thought", nlg)
    assert composer.compose(plan) == ()
    nlg.assert_not_called()


def test_reader_goal_recovers_quantity_word_when_parser_did_not_mark_target():
    goal = reader_goal(analyze_structured_utterance("У тебя сколько бабушек?"))

    assert goal.kind == "answer_slot"
    assert goal.question_word == "сколько"
    assert "бабушка" in goal.topics


def test_failed_verbalization_does_not_silently_drop_the_reason(monkeypatch):
    composer = ResponseComposer(analyze_structured_utterance("Маша видит Петю?"))
    nlg = communication.realize_thought
    reason = thought("коля", "дима")
    monkeypatch.setattr(communication, "realize_thought",
                        lambda t, *a, **k: () if t is reason else nlg(t, *a, **k))
    assert composer.compose(MessagePlan(composer.goal, thought(), reason=reason)) == ()
