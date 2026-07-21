from debilgi.language.conversational_repair import (
    repair_for_utterance,
    unknown_meaning_repair,
)
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.structured_utterance import analyze_structured_utterance


def _repair(text: str):
    return repair_for_utterance(
        analyze_structured_utterance(text),
        analyze_pragmatics(text),
    )


def test_first_person_state_asks_for_its_missing_semantic_argument() -> None:
    repair = _repair("Просто удивляюсь")

    assert repair.reason == "missing_semantic_role:indirect_object"
    assert repair.realize() == "Чему ты удивляешься?"
    assert "действ" not in repair.realize().casefold()


def test_interjection_is_continued_as_social_meaning_not_parser_failure() -> None:
    repair = _repair("ого")

    assert not repair.truly_unresolved
    assert repair.realize() == "Что тебя удивило?"


def test_isolated_topic_gets_a_natural_intention_question() -> None:
    repair = _repair("футбол")

    assert repair.truly_unresolved
    assert repair.realize() == "Что ты хочешь сказать про футбол?"


def test_unknown_word_question_is_built_from_semantic_roles() -> None:
    repair = unknown_meaning_repair("бебра")

    assert repair.realize() == "Что значит «бебра»?"
    assert repair.thought.evidence == ("meaning_unknown_after_internal_study",)


def test_personal_evaluation_continues_with_a_reason_question() -> None:
    repair = _repair("Я умный")

    assert not repair.truly_unresolved
    assert repair.realize() == "Почему ты так считаешь?"
