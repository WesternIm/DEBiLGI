from debilgi.language.personal_action_reasoning import (
    analyze_personal_action,
    reason_about_personal_action,
)
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_external_past_action_is_answered_from_owned_history() -> None:
    frame = analyze_personal_action(
        analyze_structured_utterance("Ты ножницы покупал?")
    )
    resolution = reason_about_personal_action(frame)

    assert frame is not None
    assert frame.action_lemma == "покупать"
    assert frame.object is not None
    assert frame.object.head.lemma == "ножницы"
    assert resolution is not None
    assert resolution.actual is False
    assert realize_thought(resolution.thought)[0].text == (
        "Нет, я не покупал ножницы."
    )


def test_unobserved_internal_action_is_not_invented() -> None:
    frame = analyze_personal_action(
        analyze_structured_utterance("Ты идею обдумывал?")
    )

    assert frame is not None
    assert reason_about_personal_action(frame) is None


def test_instrumental_object_keeps_verb_governed_case() -> None:
    frame = analyze_personal_action(
        analyze_structured_utterance("Ты кусачками пользуешься?")
    )
    resolution = reason_about_personal_action(frame)

    assert resolution is not None
    assert realize_thought(resolution.thought)[0].text == (
        "Нет, я не пользуюсь кусачками."
    )


def test_prepositional_action_complement_is_preserved() -> None:
    frame = analyze_personal_action(
        analyze_structured_utterance("Ты играл в гта 5?")
    )
    resolution = reason_about_personal_action(frame)

    assert resolution is not None
    assert realize_thought(resolution.thought)[0].text == (
        "Нет, я не играл в гта 5."
    )
