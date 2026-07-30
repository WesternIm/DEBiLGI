from debilgi.language.runtime_state_reasoning import (
    analyze_runtime_state,
    reason_about_runtime_state,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_language_layout_question_is_resolved_from_runtime_observation() -> None:
    frame = analyze_runtime_state(
        analyze_structured_utterance(
            "У тебя сейчас какая раскладка языка стоит?"
        )
    )
    resolution = reason_about_runtime_state(frame)

    assert frame is not None
    assert resolution is not None
    assert [
        (observation.fact_id, observation.value)
        for observation in resolution.observations
    ] == [
        (
            "runtime.keyboard_layout.observable",
            (
                "predicate=наблюдать;subject=agent;"
                "direct_object=раскладка;"
                "direct_object_modifiers=системный;"
                "tense=present;truth=false. "
                "Агент не способен наблюдать системную раскладку клавиатуры."
            ),
        ),
        (
            "message.language",
            (
                "predicate=приходить;subject=сообщение;"
                "manner=по-русски;tense=present;truth=true. "
                "Текущее сообщение пользователя написано по-русски."
            ),
        ),
    ]


def test_runtime_repair_owns_the_previous_topic_error() -> None:
    frame = analyze_runtime_state(
        analyze_structured_utterance(
            "У тебя сейчас какая раскладка языка стоит?"
        )
    )
    resolution = reason_about_runtime_state(
        frame,
        repairs_previous_answer=True,
    )

    assert resolution is not None
    assert [item.fact_id for item in resolution.observations] == [
        "previous_answer.topic_match",
        "runtime.keyboard_layout.observable",
        "message.language",
    ]
    assert not hasattr(resolution, "thoughts")
