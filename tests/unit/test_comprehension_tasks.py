from dataclasses import replace

import pytest

from debilgi.language.comprehension_tasks import (
    TaskDependency,
    dependency_frontier,
    plan_comprehension,
)
from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.sentence_understanding import understand_utterance


def word(term: str, meaning: str | None = None, *, part: str = "NOUN") -> WordAnalysis:
    return WordAnalysis(
        surface_form=term, normalized_form=term, lemma=term, part_of_speech=part,
        grammatical_features={}, dictionary_meanings=(meaning,) if meaning else (),
        style_labels=(), source="test", confidence=ConfidenceProfile(.8, .7, .5, .5, .2),
        unknown_kind=None if meaning else UnknownKind.UNKNOWN_MEANING,
    )


def test_familiar_words_do_not_resolve_a_missing_discourse_referent():
    words = (word("понимать", "постигать смысл", part="INFN"),)
    missing = understand_utterance("Как это понимать?", words)
    bound = understand_utterance("Как это понимать?", words, ("Я не знаю причины.",))
    assert not missing.sufficient
    assert {n.kind for n in missing.task.frontier.pending} == {"referent"}
    assert bound.sufficient
    assert bound.task.goal == "explain_previous"
    assert not bound.task.answer_ready
    assert {n.kind for n in bound.task.frontier.pending} == {"evidence"}
    assert bound.task.lookup_terms == ()
    assert not bound.task.research_allowed


def test_definition_does_not_prove_a_property_or_expand_every_word_in_its_text():
    unknown = plan_comprehension("Можно ли есть фепку?", (word("фепка"),))
    defined = plan_comprehension(
        "Можно ли есть фепку?", (word("фепка", "разновидность трошки с кварговиной"),)
    )
    assert unknown.lookup_terms == ("фепка",)
    assert not unknown.interpretation_ready
    assert defined.interpretation_ready
    assert defined.lexical_states == (("фепка", "defined"),)
    assert defined.lookup_terms == ()
    assert not defined.answer_ready
    assert {n.kind for n in defined.frontier.pending} == {"evidence"}
    assert not any(n.target in {"трошка", "кварговина"} for n in defined.nodes)
    assert not defined.with_answer_evidence(()).answer_ready
    assert defined.with_answer_evidence(("verified_property:inedible:source-7",)).answer_ready


@pytest.mark.parametrize("prompt", [
    "Я вчера потерял фепку", "Когда я потерял фепку?", "Где я потерял фепку?",
])
def test_opaque_event_object_is_not_a_blocking_meaning_dependency(prompt):
    task = plan_comprehension(prompt, (word("фепка"),))
    assert task.interpretation_ready
    assert "фепка" in task.incidental_terms
    assert not task.lookup_terms


def test_unresolved_person_is_a_context_gap_not_a_word_to_research():
    task = plan_comprehension("Где он живёт?", (word("жить", part="VERB"),))
    assert not task.interpretation_ready
    assert not task.lookup_terms
    assert {n.kind for n in task.frontier.pending} == {"referent"}


def test_ambiguous_sense_requires_selection_not_another_dictionary_request():
    item = replace(word("лук", "растение; оружие"), unknown_kind=UnknownKind.AMBIGUOUS_CONTEXT)
    task = plan_comprehension("Лук съедобный?", (item,))
    assert task.lexical_states == (("лук", "ambiguous"),)
    assert not task.interpretation_ready
    assert not task.lookup_terms
    assert {n.kind for n in task.frontier.pending} == {"sense_selection"}


def test_only_dependencies_of_the_requested_goal_are_visited_once():
    nodes = (
        TaskDependency("answer", "answer", "x", ("left", "right")),
        TaskDependency("left", "relation", "x", ("word",)),
        TaskDependency("right", "relation", "y", ("word",)),
        TaskDependency("word", "lexical_meaning", "фепка"),
        TaskDependency("unrelated", "lexical_meaning", "трошка"),
    )
    result = dependency_frontier(nodes, "answer")
    assert tuple(n.target for n in result.pending) == ("фепка",)
    assert not result.cycles and not result.limited


def test_circular_definition_does_not_count_as_new_evidence():
    nodes = (
        TaskDependency("answer", "answer", "x", ("a",)),
        TaskDependency("a", "relation", "a", ("b",)),
        TaskDependency("b", "relation", "b", ("a",)),
    )
    result = dependency_frontier(nodes, "answer")
    assert result.cycles == (("a", "b", "a"),)
    assert not result.pending


@pytest.mark.parametrize("limits", [{"max_depth": 1}, {"max_nodes": 2}])
def test_exhausted_dependency_budget_is_not_success(limits):
    nodes = tuple(TaskDependency(str(i), "relation", str(i), (str(i + 1),)) for i in range(10))
    result = dependency_frontier(nodes, "0", **limits)
    assert result.limited


def test_missing_dependency_is_reported_instead_of_silently_completed():
    result = dependency_frontier((TaskDependency("answer", "answer", "x", ("missing",)),),
                                 "answer")
    assert result.pending[0].kind == "missing_dependency"
