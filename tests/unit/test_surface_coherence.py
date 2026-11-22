import pytest

from debilgi.language.surface_coherence import evaluate_surface
from debilgi.language.surface_policy import internal_surface_reason


@pytest.mark.parametrize(("text", "reason"), [
    ("Я любишь собак.", "subject_verb_person_mismatch"),
    ("Мы любит собак.", "subject_verb_number_mismatch"),
    ("Ты бы тоже любишь.", "conditional_requires_past_or_infinitive"),
    ("Я думаю о собаку.", "preposition_case_mismatch"),
])
def test_morphological_violations_are_arbitration_rejections(text, reason):
    from debilgi.language.arbitration import ResponseArbitrator, ResponseCandidate

    assert reason in evaluate_surface(text).hard_issues
    candidate = ResponseCandidate.create(
        event_id="test", source="learned_distribution", renderer="test", text=text,
        speech_act="brief_statement", source_confidence=1,
        metadata={"allow_rough_surface": True, "locally_composed": True},
    )
    assert f"surface:{reason}" in ResponseArbitrator._hard_rejections(candidate)


@pytest.mark.parametrize("text", [
    "Я люблю собак.", "Ты бы любил собак.", "Мне бы поспать.",
    "Я знаю, что ты любишь собак.", "Коты и собаки бегают.", "Привет!",
    "О ком ты говоришь?", "Я думаю о собаке.",
])
def test_clause_local_agreement_accepts_valid_forms(text):
    assert not evaluate_surface(text).hard_issues


def test_document_bullet_is_not_a_conversational_surface():
    result = evaluate_surface("* создание: первый том написан в тюрьме.")

    assert "document_fragment" in result.hard_issues


def test_rejects_logged_markov_splices() -> None:
    broken = (
        ". суть: ауу, но не буду но она ещё разобрал?.",
        "правил, памяти и не буду но она ещё разобрал?. затем этого её не брал книга на полке.",
        "узнай?. затем ошибку разобрать. значение слова зависит от контекста: оно может быть.",
        "карантине. затем какую ошибку разобрал?. затем ошибку разобрал?.",
    )

    for text in broken:
        result = evaluate_surface(text)
        assert result.hard_issues
        assert result.score <= 0.20
        assert not result.learnable


def test_accepts_complete_dialogue_surfaces() -> None:
    for text in (
        "Я здесь. Говори.",
        "Предыдущий ответ получился бессвязным. Я его отбрасываю.",
        "Дракон — мифическое существо с крыльями.",
    ):
        result = evaluate_surface(text)
        assert not result.hard_issues
        assert result.score >= 0.72
        assert result.learnable


def test_internal_consultant_narration_is_never_a_user_facing_surface() -> None:
    for text in (
        "Пользователь настойчиво интересуется, находится ли агент в Китае.",
        "Пользователь реагирует на странный ответ агента оскорблением и смехом.",
        "Реплика пользователя выражает недоумение.",
        "Реальность пишет собственный скрытый тест.",
    ):
        assert internal_surface_reason(text) is not None


def test_reverse_valency_rejects_a_fluent_looking_but_broken_clause() -> None:
    result = evaluate_surface("Нет, я не поперхнулся мозг.")

    assert "unexpected_valency_role" in result.hard_issues
    assert result.score <= 0.20


def test_government_rejects_accusative_after_control_predicate() -> None:
    result = evaluate_surface("Ненадёжность мешает его.")

    assert "unexpected_valency_role" in result.hard_issues
    assert result.score <= 0.20


def test_syncretic_pronoun_case_does_not_reject_a_valid_fallback() -> None:
    result = evaluate_surface("Я не могу подтвердить это.")

    assert "governed_case_mismatch" not in result.hard_issues
    assert result.learnable


def test_generic_data_counter_is_never_a_user_facing_answer() -> None:
    for text in (
        "У меня недостаточно данных.",
        "У меня нет достаточных данных о стране.",
        "Достаточных данных нет у меня о стране.",
        "У меня нет достаточных оснований.",
        "В памяти не хватает фактов.",
    ):
        assert internal_surface_reason(text) is not None


def test_position_reason_is_not_mistaken_for_internal_evidence_bookkeeping() -> None:
    text = "У меня нет оснований для такой оценки."

    assert internal_surface_reason(text) is None
    assert evaluate_surface(text).learnable


def test_genitive_object_is_licensed_by_negation_only() -> None:
    assert "governed_case_mismatch" not in evaluate_surface("Я не имею желудка.").hard_issues
    assert "governed_case_mismatch" in evaluate_surface("Я имею желудка.").hard_issues


def test_expressive_parenthesis_before_a_closing_quote_is_not_unbalanced():
    assert "unbalanced_delimiters" not in evaluate_surface("Я сказал «дадно!)».").hard_issues
    assert "unbalanced_delimiters" in evaluate_surface("Я (не знаю.").hard_issues
    assert "unbalanced_delimiters" in evaluate_surface("Я сказал «привет.").hard_issues
