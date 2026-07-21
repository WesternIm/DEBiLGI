from dataclasses import replace

import debilgi.language.semantic_reply_plan as semantic_reply_module
from debilgi.language.semantic_reply_plan import (
    EXPRESS_LACK_OF_UNDERSTANDING,
    REQUEST_CLARIFICATION,
    ProcessingOutcomeType,
    build_clarification_plan,
    intentional_silence_outcome,
    realize_clarification_plan,
    validate_semantic_reply,
)


def clarification_plan():
    return build_clarification_plan(
        intention_id="intent:clarify",
        target_user_id=100,
        prompt="Э-э, это самое про штуку",
        confidence=0.72,
        supporting_memory_ids=("discord:message:1", "hypothesis:ambiguous"),
        context_messages=("Книга принадлежит Борису.",),
    )


def test_clarification_plan_stores_meaning_before_wording() -> None:
    plan = clarification_plan()

    assert plan.speech_act == "clarifying_question"
    assert plan.target_user_id == 100
    assert plan.requested_information == ("intended_meaning",)
    assert plan.required_elements == (
        EXPRESS_LACK_OF_UNDERSTANDING,
        REQUEST_CLARIFICATION,
    )
    assert tuple(item.predicate for item in plan.propositions) == (
        "understand",
        "clarify",
    )
    assert plan.propositions[0].qualifiers == (("polarity", "negative"),)
    assert plan.propositions[1].qualifiers == (("mode", "request"),)
    assert "hypothesis:ambiguous" in plan.causal_hypothesis_ids
    assert "книга" in plan.forbidden_drift_topics


def test_productive_realizer_preserves_the_clarification_plan() -> None:
    result = realize_clarification_plan(clarification_plan(), max_attempts=4)

    assert result.outcome.outcome_type is ProcessingOutcomeType.REPLIED
    assert result.outcome.was_intentional is False
    assert result.text is not None and result.text.endswith("?")
    assert result.validation.is_valid
    assert result.validation.coverage == 1.0
    assert result.used_fallback is False
    assert len(result.candidates) >= 2
    assert all(candidate.validation.is_valid for candidate in result.candidates)
    assert len({candidate.text for candidate in result.candidates}) >= 2


def test_validator_rejects_memory_drift_instead_of_rewarding_fluency() -> None:
    result = validate_semantic_reply(
        clarification_plan(),
        "Книга принадлежит Борису.",
    )

    assert result.is_valid is False
    assert result.coverage == 0.0
    assert set(result.missing_elements) == {
        EXPRESS_LACK_OF_UNDERSTANDING,
        REQUEST_CLARIFICATION,
    }
    assert "книга" in result.semantic_drift


def test_validator_rejects_a_question_turned_into_a_statement() -> None:
    result = validate_semantic_reply(
        clarification_plan(),
        "Я понимаю это сообщение и поясняю его.",
    )

    assert result.is_valid is False
    assert "claims_understanding" in result.contradictions
    assert "question_became_statement" in result.contradictions


def test_intentional_silence_is_not_a_language_failure() -> None:
    silence = intentional_silence_outcome(
        intention_id="intent:silence",
        reason="conversation does not require a reply",
    )
    failed = realize_clarification_plan(
        replace(clarification_plan(), speech_act="statement")
    ).outcome

    assert silence.outcome_type is ProcessingOutcomeType.INTENTIONAL_SILENCE
    assert silence.was_intentional is True
    assert silence.generated_text is None
    assert silence.failure_stage is None
    assert failed.outcome_type is ProcessingOutcomeType.FAILED_TO_FORM_PLAN
    assert failed.was_intentional is False
    assert failed.failure_stage == "semantic_planning"


def test_language_exception_is_recorded_and_uses_semantic_fallback(monkeypatch) -> None:
    def fail_realization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("local morphology failure")

    monkeypatch.setattr(semantic_reply_module, "realize_thought", fail_realization)

    result = realize_clarification_plan(clarification_plan())

    assert result.text == "Не понял. Пояснишь?"
    assert result.used_fallback is True
    assert result.validation.is_valid
    assert result.outcome.outcome_type is ProcessingOutcomeType.REPLIED
    assert result.outcome.was_intentional is False
    assert result.outcome.failure_stage == "language_realization"
    assert result.outcome.failure_reason == "productive NLG raised RuntimeError"
