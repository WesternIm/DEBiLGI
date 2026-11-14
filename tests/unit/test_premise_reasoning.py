from dataclasses import replace

import pytest

from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.premise_reasoning import (
    infer_from_premises,
    lexical_premise,
    realize_premise_inference,
)
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.surface_coherence import evaluate_surface


def word(lemma, *definitions):
    return WordAnalysis(
        lemma, lemma, lemma, "NOUN", {}, definitions, (), "dictionary_cache",
        ConfidenceProfile(1, .8, .34, .8, .34),
        unknown_kind=UnknownKind.AMBIGUOUS_CONTEXT if len(definitions) > 1 else None,
    )


@pytest.mark.parametrize(("prompt", "analysis", "expected"), [
    ("Что думаешь о ножницах?", word("ножницы",
     "режущий инструмент из двух раздвигающихся лезвий", "технический приём в футболе"),
     ("Если", "инструмент", "полезными", "резать")),
    ("У тебя нос капает?", word("нос",
     "выступающая часть лица человека", "передняя часть судна"),
     ("Если", "лица", "носа", "тела")),
    ("У тебя желудок болит?", word("желудок", "внутренний орган пищеварения"),
     ("желудка", "тела")),
    ("Что думаешь о грелке?", word("грелка", "согревающий прибор"),
     ("грелку", "полезной", "согревать")),
])
def test_inference_builds_a_conclusion_and_reason(prompt, analysis, expected):
    inference = infer_from_premises(analyze_structured_utterance(prompt), (analysis,))
    assert inference is not None
    surface = realize_premise_inference(inference)
    assert surface is not None
    assert all(part in surface.text for part in expected)
    assert not evaluate_surface(surface.text).hard_issues
    assert not any(surface.text == definition for definition in analysis.dictionary_meanings)


@pytest.mark.parametrize("prompt", [
    "У меня нос капает?", "У него нос капает?", "У тебя виртуальный нос капает?",
    "У тебя на экране нос капает?", "Если у тебя нос капает, что делать?",
])
def test_ownership_deduction_respects_actor_and_hypothetical_scope(prompt):
    analysis = word("нос", "часть лица человека")
    assert infer_from_premises(analyze_structured_utterance(prompt), (analysis,)) is None


def test_unrelated_dictionary_mentions_are_not_premises():
    assert lexical_premise("идея о режущем инструменте") is None
    assert lexical_premise("не режущий инструмент") is None
    assert lexical_premise("режущий инструмент, который не работает") is None
    assert lexical_premise("орган законодательной власти") is None
    assert lexical_premise("орган управления") is None


@pytest.mark.parametrize("prompt", [
    "Почему ты думаешь о ножницах?", "Ты думаешь о ножницах?",
    "Что он думает о ножницах?", "Когда ты думал о ножницах?",
])
def test_appraisal_does_not_replace_a_question_about_someones_thinking(prompt):
    assert infer_from_premises(analyze_structured_utterance(prompt),
                               (word("ножницы", "режущий инструмент"),)) is None


def test_explicit_local_sense_takes_precedence_over_dictionary_senses():
    analysis = replace(word("ножницы", "режущий инструмент"),
                       personal_interpretation="приём в футболе")
    assert infer_from_premises(analyze_structured_utterance("Что думаешь о ножницах?"),
                               (analysis,)) is None
