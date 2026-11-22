from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.question_reasoning import (
    analyze_semantic_question,
    reason_from_internal_knowledge,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_extracts_comparison_behind_stance_clause() -> None:
    prompt = (
        "Как ты думаешь, что полезнее для исследования Марса: "
        "роботы или люди? Почему?"
    )

    frame = analyze_semantic_question(analyze_structured_utterance(prompt))

    assert frame is not None
    assert frame.kind == "comparison"
    assert frame.focus == "полезный"
    assert frame.alternatives == ("робот", "человек")
    assert frame.context_terms == ("исследование", "марс")
    assert "думать" not in frame.study_terms
    assert frame.complete is True


def test_complete_comparison_reports_knowledge_gap_not_missing_question() -> None:
    prompt = (
        "Кто полезнее при исследовании Марса — роботы или люди? "
        "Назови один вариант и объясни причину."
    )
    frame = analyze_semantic_question(analyze_structured_utterance(prompt))

    assert frame is not None
    result = reason_from_internal_knowledge(frame, ())

    assert result.needs_study is True
    assert result.text == ""


def test_selects_alternative_from_internalized_meaning_links() -> None:
    prompt = "Что полезнее для исследования Марса: роботы или люди?"
    frame = analyze_semantic_question(analyze_structured_utterance(prompt))
    robot = WordAnalysis(
        surface_form="роботы",
        normalized_form="роботы",
        lemma="робот",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=(
            "автоматическое устройство для исследования опасной среды Марса",
        ),
        style_labels=(),
        source="internal_dictionary",
        confidence=ConfidenceProfile(0.95, 0.90, 0.88, 0.90, 0.88),
    )
    person = WordAnalysis(
        surface_form="люди",
        normalized_form="люди",
        lemma="человек",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=("живое разумное существо",),
        style_labels=(),
        source="internal_dictionary",
        confidence=ConfidenceProfile(0.95, 0.90, 0.70, 0.90, 0.70),
    )

    assert frame is not None
    result = reason_from_internal_knowledge(frame, (robot, person))

    assert result.needs_study is False
    assert "робот" in result.text
    assert set(result.used_evidence) >= {"исследование", "марс"}
    assert "автоматическое устройство" not in result.text


def test_post_study_reasoning_answers_without_exposing_the_study_process() -> None:
    frame = analyze_semantic_question(
        analyze_structured_utterance("Что полезнее: роботы или люди?")
    )
    robot = WordAnalysis(
        surface_form="роботы",
        normalized_form="роботы",
        lemma="робот",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=("автоматическое устройство, выполняющее заданные действия",),
        style_labels=(),
        source="ru_wiktionary",
        confidence=ConfidenceProfile(0.95, 0.90, 0.80, 0.90, 0.80),
    )
    person = WordAnalysis(
        surface_form="люди",
        normalized_form="люди",
        lemma="человек",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=("живое разумное общественное существо",),
        style_labels=(),
        source="ru_wiktionary",
        confidence=ConfidenceProfile(0.95, 0.90, 0.80, 0.90, 0.80),
    )

    assert frame is not None
    result = reason_from_internal_knowledge(
        frame,
        (robot, person),
        study_completed=True,
    )

    assert result.text
    assert result.needs_study is False
    assert "доизуч" not in result.text
    assert "словар" not in result.text


def test_bare_alternative_is_a_complete_preference_question() -> None:
    frame = analyze_semantic_question(
        analyze_structured_utterance("Краба выбираешь или рака?")
    )

    assert frame is not None
    assert frame.kind == "choice"
    assert frame.alternatives == ("краб", "рак")
    assert frame.complete is True

    result = reason_from_internal_knowledge(frame, ())

    assert result.needs_study is False
    assert result.text in {"Я выбираю краба.", "Я выбираю рака."}
