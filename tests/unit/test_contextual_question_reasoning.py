from debilgi.language.contextual_question_reasoning import (
    epistemic_gap_candidates_for_question,
    epistemic_gap_for_question,
    reason_about_contextual_question,
)
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.structured_utterance import analyze_structured_utterance


def _resolve(prompt: str, *context: str) -> tuple[str, str]:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance(prompt),
        context,
    )
    assert resolution is not None
    surfaces = realize_thought(resolution.thought, resolution.patterns)
    assert surfaces
    return resolution.kind, surfaces[0].text


def test_epistemic_follow_ups_select_different_semantic_actions() -> None:
    previous = "Я не знаю ответа на этот вопрос."

    assert _resolve("Зачем?", previous) == (
        "epistemic_purpose",
        "Я не выдумываю ответ.",
    )
    assert _resolve("Почему же не знаешь?", previous) == (
        "overcautious_response_cause",
        "Я слишком осторожно отвечаю из-за завышенного порога.",
    )
    assert _resolve("А хотел бы знать?", previous) == (
        "epistemic_desire",
        "Да, я хочу это узнать.",
    )


def test_cognitive_method_is_answered_from_actual_local_process() -> None:
    kind, text = _resolve(
        "А как ты тогда думаешь?",
        "Нет, у меня нет сознания.",
    )

    assert kind == "cognitive_method"
    assert text == "Я анализирую контекст."


def test_elliptical_conditional_restores_the_previous_proposition() -> None:
    kind, text = _resolve(
        "А если всё же есть?",
        "Нет, у меня нет сознания.",
    )

    assert kind == "conditional_verification"
    assert text == "Это требует проверки."


def test_exhausted_question_sources_produce_competing_semantic_moves() -> None:
    structure = analyze_structured_utterance("Где ключ?")
    plans = epistemic_gap_candidates_for_question(structure)
    resolution = epistemic_gap_for_question(structure)

    assert resolution is not None
    assert len(plans) >= 3
    assert len({plan.strategy for plan in plans}) == len(plans)
    assert resolution.kind == "location_gap"
    assert realize_thought(resolution.thought)[0].text == "Я не могу найти ключ."


def test_person_attribution_gap_does_not_turn_a_label_into_missing_data() -> None:
    plans = epistemic_gap_candidates_for_question(
        analyze_structured_utterance("Кто тут шалава?")
    )

    assert plans[0].strategy == "withhold_unsupported_label"
    assert realize_thought(plans[0].thought, plans[0].patterns)[0].text == (
        "Я никого шалавой не называю."
    )


def test_embedded_content_clause_does_not_create_an_epistemic_gap() -> None:
    plans = epistemic_gap_candidates_for_question(
        analyze_structured_utterance(
            "Ой бля, не делай вид, что ты не понял, ахахахах"
        )
    )

    assert plans == ()


def test_contextual_reasoning_does_not_capture_an_independent_cause_question() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("Почему небо синее?"),
        ("Мы только начали разговор.",),
    )

    assert resolution is None


def test_understanding_follow_up_restores_previous_directive_content() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("Что понял?"),
        ("Выкинь штурвал",),
    )

    assert resolution is not None
    assert resolution.kind == "restated_directive"
    assert realize_thought(resolution.thought, resolution.patterns)[0].text == (
        "Ты просишь выкинуть штурвал."
    )


def test_reason_follow_up_uses_embodiment_instead_of_generic_evidence_gap() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("А почему так?"),
        ("Нет, я не ел яблоко.",),
    )

    assert resolution is not None
    assert resolution.kind == "embodiment_cause"
    assert realize_thought(resolution.thought)[0].text == "У меня нет тела."


def test_location_follow_up_exposes_the_missing_slot_instead_of_inventing_it() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("Куда орут?"),
        ("Люди орут.",),
    )

    assert resolution is not None
    assert resolution.kind == "antecedent_location_unspecified"
    assert realize_thought(resolution.thought, resolution.patterns)[0].text == (
        "В предыдущей фразе я не указал направление."
    )


def test_preference_cause_follow_up_does_not_repeat_the_preference_claim() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("А почему не любишь?"),
        ("Нет, я не люблю небо.",),
    )

    assert resolution is not None
    assert resolution.kind == "preference_cause_unestablished"
    assert "небо" in resolution.target_terms
    assert realize_thought(resolution.thought, resolution.patterns)[0].text == (
        "Причина остаётся загадкой."
    )


def test_epistemic_action_is_not_misclassified_as_bodily_action() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance("Почему не определил?"),
        ("Не определил я отношение к имени.",),
    )

    assert resolution is not None
    assert resolution.kind == "overcautious_response_cause"
    assert "external_action_requires_body" not in resolution.evidence
    assert realize_thought(resolution.thought, resolution.patterns)[0].text == (
        "Я слишком осторожно отвечаю из-за завышенного порога."
    )


def test_criticism_after_no_data_reply_gets_a_cause_not_another_no_data_reply() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance(
            "Да почему ты такой пугливый в ответах?"
        ),
        ("У меня нет достаточных данных.",),
    )

    assert resolution is not None
    assert resolution.kind == "overcautious_response_cause"
    surface = realize_thought(resolution.thought, resolution.patterns)[0].text
    assert surface == "Я слишком осторожно отвечаю из-за завышенного порога."
    assert "недостаточ" not in surface.casefold()


def test_counterevidence_revises_a_denied_internal_capability() -> None:
    resolution = reason_about_contextual_question(
        analyze_structured_utterance(
            "Почему? Ты сейчас оцениваешь и отвечаешь мне"
        ),
        ("Нет, я не умею иметь мнение.",),
    )

    assert resolution is not None
    assert resolution.kind == "internal_capability_correction"
    assert {"мнение", "оценивать", "отвечать"} <= set(resolution.target_terms)
    assert realize_thought(resolution.thought, resolution.patterns)[0].text == (
        "Да, я могу иметь мнение."
    )
