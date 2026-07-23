from debilgi.language.response_alignment import assess_response_alignment
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_rejects_answer_about_another_subject_and_action() -> None:
    prompt = analyze_structured_utterance("Астрономы умеют летать?")

    assessment = assess_response_alignment(prompt, "Я умею смысл.")

    assert not assessment.allowed
    assert "answer_subject_mismatch" in assessment.reasons
    assert "answer_focus_missing" in assessment.reasons
    assert assessment.question_focus == ("летать",)


def test_second_person_question_accepts_first_person_perspective_shift() -> None:
    prompt = analyze_structured_utterance("А ты умеешь летать?")

    assessment = assess_response_alignment(
        prompt,
        "Нет, физически летать я не умею.",
    )

    assert assessment.allowed
    assert assessment.expected_subjects == ("ты",)
    assert "летать" in assessment.matched_focus


def test_fluent_but_unrelated_consciousness_answer_is_rejected() -> None:
    prompt = analyze_structured_utterance("Насколько считаешь ты себя осознанным?")

    assessment = assess_response_alignment(prompt, "Умник ответил на вопрос.")

    assert not assessment.allowed
    assert assessment.expected_subjects == ("ты",)
    assert "осознанный" in assessment.question_focus


def test_reflexive_topic_must_survive_an_open_self_question() -> None:
    prompt = analyze_structured_utterance("Что ты думаешь о себе?")

    unrelated = assess_response_alignment(
        prompt,
        "Правильно: думаю о новом смысле.",
    )
    bound = assess_response_alignment(
        prompt,
        "Я считаю себя обучающимся собеседником.",
    )

    assert not unrelated.allowed
    assert "себя" in unrelated.question_focus
    assert bound.allowed


def test_unknown_named_entity_is_a_required_question_focus() -> None:
    prompt = analyze_structured_utterance("Ты играл в гта 5?")

    unrelated = assess_response_alignment(prompt, "Свои игрульки играешь.")
    bound = assess_response_alignment(prompt, "Нет, я не играл в гта 5.")

    assert not unrelated.allowed
    assert set(unrelated.question_focus) >= {"гта", "5"}
    assert bound.allowed
