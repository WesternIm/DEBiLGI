from debilgi.language.productive_nlg import realize_thought
from debilgi.language.self_state_reasoning import (
    analyze_self_state,
    reason_about_self_state,
)
from debilgi.language.sentence_understanding import UtteranceUnderstanding
from debilgi.language.structured_utterance import analyze_structured_utterance


def _understanding(*known: str) -> UtteranceUnderstanding:
    return UtteranceUnderstanding(
        content_terms=known,
        known_terms=known,
        unknown_terms=(),
        relations=(),
        context_links=known,
        comprehension_score=0.92,
        sufficient=True,
    )


def test_self_understanding_question_is_resolved_from_context_then_realized() -> None:
    frame = analyze_self_state(analyze_structured_utterance("А ты понял идею?"))

    assert frame is not None
    result = reason_about_self_state(
        frame,
        _understanding("понять", "идея"),
        context_messages=("Идея состоит в том, чтобы сначала собрать мысль.",),
    )

    assert result.actual is True
    assert result.thought is not None
    assert realize_thought(result.thought)[0].text == "Да, я понял идею."


def test_same_reasoner_handles_knowledge_and_memory_predicates() -> None:
    knowledge = analyze_self_state(analyze_structured_utterance("Ты знаешь термин?"))
    memory = analyze_self_state(analyze_structured_utterance("Ты помнишь разговор?"))

    assert knowledge is not None and memory is not None
    known = reason_about_self_state(
        knowledge,
        _understanding("знать", "термин"),
        knowledge_facts=("Термин обозначает специальное понятие.",),
    )
    remembered = reason_about_self_state(
        memory,
        _understanding("помнить", "разговор"),
        context_messages=("Этот разговор посвящён построению предложений.",),
    )

    assert realize_thought(known.thought)[0].text == "Да, я знаю термин."
    assert realize_thought(remembered.thought)[0].text == "Да, я помню разговор."


def test_ungrounded_idea_does_not_create_a_fake_confident_answer() -> None:
    frame = analyze_self_state(analyze_structured_utterance("Ты понял идею?"))

    assert frame is not None
    result = reason_about_self_state(frame, _understanding("понять", "идея"))

    assert result.actual is None
    assert result.thought is None


def test_misunderstanding_report_creates_a_semantic_repair_question() -> None:
    frame = analyze_self_state(analyze_structured_utterance("Я не понял"))

    assert frame is not None
    result = reason_about_self_state(frame, _understanding("понять"))

    assert result.thought is not None
    assert realize_thought(result.thought)[0].text == "Что осталось непонятным?"


def test_open_self_reflection_is_answered_from_the_learning_architecture() -> None:
    frame = analyze_self_state(
        analyze_structured_utterance("Что ты думаешь о себе?")
    )

    assert frame is not None
    assert frame.state == "self_reflection"
    assert frame.complete
    result = reason_about_self_state(frame, _understanding("думать"))

    assert result.thought is not None
    assert realize_thought(result.thought)[0].text == (
        "Я считаю себя обучающимся собеседником."
    )
    assert "local_learning_architecture" in result.evidence


def test_open_self_reflection_recovers_omitted_second_person_subject() -> None:
    frame = analyze_self_state(analyze_structured_utterance("Что думаешь о себе"))

    assert frame is not None
    assert frame.experiencer_kind == "self"
    assert frame.complete
    result = reason_about_self_state(frame, _understanding("думать"))
    assert result.thought is not None


def test_self_location_is_derived_from_runtime_environment() -> None:
    for prompt in (
        "Где ты находишься?",
        "Ты где находишься?",
        "Ладно. А где ты вообще находишься?",
    ):
        frame = analyze_self_state(analyze_structured_utterance(prompt))

        assert frame is not None
        assert frame.state == "self_location"
        assert frame.complete
        result = reason_about_self_state(frame, _understanding("находиться"))

        assert result.thought is not None
        assert realize_thought(result.thought)[0].text == "Я работаю в Discord."
        assert "discord_runtime_environment" in result.evidence
