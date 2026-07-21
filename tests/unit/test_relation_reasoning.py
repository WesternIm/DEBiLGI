import pytest

from debilgi.language.relation_reasoning import (
    analyze_relation_question,
    reason_about_relation,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


@pytest.mark.parametrize(
    "text,holder_kind,object_surface",
    (
        ("У тебя есть носки?", "self", "носки"),
        ("Есть ли у тебя носки?", "self", "носки"),
        ("Хочу узнать, есть ли у тебя носки", "self", "носки"),
        ("Ты имеешь носки?", "self", "носки"),
        ("Носки у тебя есть?", "self", "носки"),
        ("У меня есть ключ?", "speaker", "ключ"),
        ("У Маши есть ключ?", "other", "ключ"),
    ),
)
def test_extracts_general_possession_frame(
    text: str,
    holder_kind: str,
    object_surface: str,
) -> None:
    frame = analyze_relation_question(analyze_structured_utterance(text))

    assert frame is not None
    assert frame.relation == "possession"
    assert frame.holder_kind == holder_kind
    assert frame.object_surface == object_surface
    assert frame.complete


def test_keeps_ambiguous_object_lemmas_until_meaning_can_resolve_them() -> None:
    frame = analyze_relation_question(analyze_structured_utterance("У тебя есть носки?"))

    assert frame is not None
    assert {"носка", "носок"} <= set(frame.object_lemmas)

    result = reason_about_relation(
        frame,
        (),
        (
            ("носка", "действие или процесс ношения чего-либо"),
            ("носок", "предмет одежды, короткое изделие из ткани"),
        ),
        study_completed=True,
    )

    assert result.object_lemma == "носок"
    assert result.object_class == "physical"
    assert result.text == "Нет, у меня нет носков."


@pytest.mark.parametrize(
    "text,concepts,expected_prefix,expected_class",
    (
        (
            "У тебя есть память?",
            (("память", "способность сохранять и использовать информацию"),),
            "Да,",
            "cognitive",
        ),
        (
            "У тебя нет памяти?",
            (("память", "способность сохранять и использовать информацию"),),
            "Нет,",
            "cognitive",
        ),
        (
            "У тебя есть куртка?",
            (("куртка", "предмет верхней одежды, изделие из ткани"),),
            "Нет,",
            "physical",
        ),
    ),
)
def test_subject_ontology_answers_by_object_class_not_by_named_item(
    text: str,
    concepts: tuple[tuple[str, str], ...],
    expected_prefix: str,
    expected_class: str,
) -> None:
    frame = analyze_relation_question(analyze_structured_utterance(text))

    assert frame is not None
    result = reason_about_relation(frame, (), concepts, study_completed=True)

    assert result.text.startswith(expected_prefix)
    assert result.object_class == expected_class
    assert result.needs_study is False


def test_unknown_third_party_possession_is_not_invented() -> None:
    frame = analyze_relation_question(analyze_structured_utterance("У Маши есть ключ?"))

    assert frame is not None
    result = reason_about_relation(
        frame,
        (),
        (("ключ", "металлический инструмент для отпирания замка"),),
        study_completed=True,
    )

    assert result.text.startswith("Не знаю")
    assert "есть" in result.text
