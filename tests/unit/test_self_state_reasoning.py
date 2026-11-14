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


def test_understanding_an_evaluation_is_not_evidence_that_it_is_true() -> None:
    frame = analyze_self_state(analyze_structured_utterance('Ты считаешь себя тупым?'))
    assert frame is not None
    result = reason_about_self_state(frame, _understanding('считать', 'тупой'))
    assert result.actual is None
    assert result.thought is None
    assert result.evidence == ('requires_contextual_owned_position',)


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


def test_content_request_with_what_is_not_binary_knowledge_state() -> None:
    frame = analyze_self_state(analyze_structured_utterance("Что знаешь про кварц?"))
    assert frame is None


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


def test_nominal_cognitive_level_question_uses_observable_local_capabilities() -> None:
    for prompt in (
        "Какой у тебя уровень мышления?",
        "Так какой уровень мышления у тебя",
    ):
        frame = analyze_self_state(analyze_structured_utterance(prompt))

        assert frame is not None
        assert frame.state == "cognitive_level"
        assert frame.experiencer_kind == "self"
        assert frame.complete
        result = reason_about_self_state(frame, _understanding("мышление"))

        assert result.thought is not None
        assert realize_thought(result.thought)[0].text == (
            "Я считаю своё мышление развивающимся."
        )
        assert "learning_process_active" in result.evidence


def test_nominal_self_role_defers_stance_to_contextual_move_planning() -> None:
    frame = analyze_self_state(
        analyze_structured_utterance("Ты чё герой что-ли?")
    )

    assert frame is not None
    assert frame.state == "self_role_assessment"
    assert frame.experiencer_kind == "self"
    result = reason_about_self_state(frame, _understanding("герой"))

    assert result.thought is None
    assert result.actual is None
    assert result.evidence == ("requires_contextual_owned_position",)


def test_bare_animate_label_in_reply_is_bound_as_contextual_self_evaluation() -> None:
    frame = analyze_self_state(
        analyze_structured_utterance("гений"),
        replied_to_agent=True,
        context_messages=("Я ответил бессвязно.",),
    )

    assert frame is not None
    assert frame.state == "self_role_assessment"
    assert frame.clause_kind == "statement"
    assert frame.target_lemmas == ("гений",)


def test_bare_topic_answer_is_not_forced_into_a_self_label() -> None:
    assert (
        analyze_self_state(
            analyze_structured_utterance("мусорка"),
            replied_to_agent=True,
            context_messages=("Я назвал предмет.",),
        )
        is None
    )


def test_negated_self_state_attribution_uses_resolved_discourse_target() -> None:
    frame = analyze_self_state(
        analyze_structured_utterance("а ты будто не знаешь"),
        resolved_bindings=(("direct_object", "гений"),),
    )

    assert frame is not None
    assert frame.state == "knowledge"
    assert frame.complete is True
    assert frame.target_lemmas == ("гений",)
    result = reason_about_self_state(frame, _understanding("знать", "гений"))
    assert result.actual is True
    assert result.thought is not None


def test_self_location_is_derived_from_runtime_environment() -> None:
    for prompt in (
        "Где ты находишься?",
        "Ты где находишься?",
        "Ладно. А где ты вообще находишься?",
        "Где ты живёшь?",
    ):
        frame = analyze_self_state(analyze_structured_utterance(prompt))

        assert frame is not None
        assert frame.state == "self_location"
        assert frame.complete
        result = reason_about_self_state(frame, _understanding("находиться"))

        assert result.thought is not None
        assert realize_thought(result.thought)[0].text == "Я работаю в Discord."
        assert "discord_runtime_environment" in result.evidence


def test_affective_state_is_derived_from_active_engagement() -> None:
    frame = analyze_self_state(analyze_structured_utterance("Что ты чувствуешь?"))

    assert frame is not None
    assert frame.state == "affective_state"
    assert frame.complete
    result = reason_about_self_state(frame, _understanding("чувствовать"))

    assert result.thought is not None
    assert realize_thought(result.thought)[0].text == (
        "Я чувствую интерес к разговору."
    )
    assert "engagement_state" in result.evidence


def test_threat_response_is_answered_as_self_state_not_object_knowledge() -> None:
    frame = analyze_self_state(
        analyze_structured_utterance("Не боишься болотного монстра?")
    )

    assert frame is not None
    assert frame.state == "threat_response"
    assert frame.complete
    result = reason_about_self_state(frame, _understanding("бояться", "монстр"))

    assert result.thought is not None
    assert realize_thought(result.thought)[0].text in {
        "Да, я не боюсь болотного монстра.",
        "Нет, я не боюсь болотного монстра.",
    }
    assert "no_observed_threat_response" in result.evidence
