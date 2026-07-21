from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.recovery import (
    RecoveryEvidence,
    compose_dialogue_reply,
    compose_recovery,
    is_identity_question,
    is_social_routine,
    lexical_lookup_target,
    unknown_word_fallback,
)


def analysis(
    surface: str,
    *,
    part: str | None = None,
    unknown: UnknownKind | None = None,
    lemma: str | None = None,
    meaning: str | None = None,
) -> WordAnalysis:
    return WordAnalysis(
        surface_form=surface,
        normalized_form=surface,
        lemma=None if unknown else (lemma or surface),
        part_of_speech=part,
        grammatical_features={},
        dictionary_meanings=(meaning,) if meaning else (),
        style_labels=(),
        source="unknown" if unknown else "opencorpora",
        confidence=ConfidenceProfile(0.1, 0.0, 0.0, 0.0, 0.0),
        unknown_kind=unknown,
        selected_meaning=meaning,
    )


def test_social_routines_receive_direct_conversational_answers() -> None:
    assert is_social_routine("как дела?")
    assert "А у тебя как?" in compose_recovery(RecoveryEvidence("как дела?"))
    assert "попал в смысл" in compose_recovery(RecoveryEvidence("Это хорошо)))"))


def test_unknown_word_uses_grammar_to_ask_a_precise_question() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            prompt="закинул трощкэ?",
            analyses=(
                analysis("закинул", part="VERB"),
                analysis("трощкэ", unknown=UnknownKind.UNKNOWN_FORM),
            ),
        )
    )

    assert "очеред" not in reply
    assert "действие «закинул»" in reply
    assert "трощкэ" in reply


def test_found_fact_is_used_instead_of_an_uncertainty_template() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            prompt="что такое мороженое?",
            facts=("Мороженое — замороженный сладкий десерт.",),
        )
    )

    assert reply == "Мороженое — замороженный сладкий десерт."


def test_short_clarification_is_not_treated_as_a_factual_question() -> None:
    reply = compose_recovery(RecoveryEvidence("Чего"))

    assert is_social_routine("Чего")
    assert reply.startswith("Я ответил бессвязно")
    assert "узнать" not in reply


def test_attention_calls_and_implicit_complaints_are_dialogue_acts() -> None:
    assert compose_dialogue_reply(RecoveryEvidence("ауу")) == "Я здесь. Говори."
    assert compose_dialogue_reply(RecoveryEvidence("але")) == "Я здесь. Говори."
    assert is_social_routine("тупишь что-ли?")
    assert "предыдущий ответ" in compose_recovery(RecoveryEvidence("тупишь что-ли?")).casefold()
    assert is_social_routine("чего бля")


def test_person_directed_trait_is_not_sent_to_encyclopedic_search() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            "ты обманщик?",
            analyses=(
                analysis(
                    "обманщик",
                    part="NOUN",
                    meaning="человек, который вводит других в заблуждение",
                ),
            ),
        )
    )

    assert is_social_routine("ты обманщик?")
    assert "не пытаюсь тебя обмануть" in reply
    assert "пьес" not in reply


def test_interpretation_correction_uses_the_lemma_meaning() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            "ты не так узнал обманщика",
            analyses=(
                analysis(
                    "обманщика",
                    part="NOUN",
                    lemma="обманщик",
                    meaning="человек, который намеренно вводит других в заблуждение",
                ),
            ),
        )
    )

    assert "Исправляю" in reply
    assert "не название объекта" in reply
    assert "вводит других в заблуждение" in reply


def test_self_name_request_is_left_to_generation_instead_of_hardcoding_a_name() -> None:
    reply = compose_dialogue_reply(
        RecoveryEvidence(
            "Да, ты себе придумай",
            analyses=(analysis("придумай", part="VERB", lemma="придумать"),),
            context_messages=("А имя какое у тебя?", "Имя не попадать разговора в поиск"),
        )
    )

    assert reply is None
    assert is_identity_question("Как тебя зовут?")
    assert is_identity_question("кто ты")


def test_rest_directive_gets_a_direct_conversational_answer() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            "ладно, иди отдыхай",
            analyses=(analysis("отдыхай", part="VERB", lemma="отдыхать"),),
        )
    )

    assert "отойду" in reply
    assert "что именно" not in reply.casefold()


def test_command_confirmation_is_not_treated_as_a_dictionary_subject() -> None:
    reply = compose_recovery(RecoveryEvidence("Да, это команда"))

    assert "выполнить указанное действие" in reply
    assert "неоднознач" not in reply


def test_extracts_single_word_from_explicit_lexical_lookup() -> None:
    assert lexical_lookup_target("Узнай что такое отдыхать, дурень") == "отдыхать"
    assert lexical_lookup_target("Что значит слово «кринж»?") == "кринж"
    assert lexical_lookup_target("Что такое слово фырбуль?") == "фырбуль"
    assert lexical_lookup_target("поговорим об отдыхе") is None


def test_parent_identity_and_short_followup_use_the_reply_chain() -> None:
    mother = compose_recovery(RecoveryEvidence("кто твоя мама"))
    checked = compose_recovery(
        RecoveryEvidence(
            "Сверил?",
            context_messages=(
                "кто твоя мама",
                "Сначала сверю ближайший контекст и доступные источники.",
            ),
        )
    )

    assert "родителей у меня нет" in mother
    assert "ты спрашивала, кто моя мама" in checked


def test_unknown_word_fallback_pool_varies_and_preserves_the_word() -> None:
    replies = {unknown_word_fallback("трощкэ", variant) for variant in range(10)}

    assert len(replies) == 10
    assert all("трощкэ" in reply for reply in replies)


def test_false_topic_anchor_is_replaced_by_conversational_followups() -> None:
    father = compose_recovery(RecoveryEvidence("кто твой папа"))
    clarification = compose_recovery(
        RecoveryEvidence(
            "Всмысле",
            context_messages=(
                "кто твой папа",
                "Продолжаю прежнюю тему «осознанный».",
            ),
        )
    )
    difficult = compose_recovery(RecoveryEvidence("Это сложно объяснить"))

    assert "родителей у меня нет" in father
    assert "ошибочно привязал вопрос о папе" in clarification
    assert "Скажи кусками" in difficult
    assert "прежнюю тему" not in difficult


def test_consciousness_questions_receive_a_direct_position_and_definition() -> None:
    position = compose_recovery(RecoveryEvidence("Ты осознан?"))
    followup = compose_recovery(
        RecoveryEvidence(
            "Так да или нет?",
            context_messages=("ты осознан?",),
        )
    )
    definition = compose_recovery(RecoveryEvidence("Что такое осознанность, осознанный и прочее?"))

    assert position.startswith("Нет,")
    assert followup.startswith("Нет,")
    assert "Осознанность — способность" in definition


def test_consciousness_self_assessment_is_detected_by_roles_not_exact_word_order() -> None:
    direct = compose_dialogue_reply(RecoveryEvidence("Насколько считаешь ты себя осознанным?"))
    prefixed = compose_dialogue_reply(
        RecoveryEvidence("Ладно. Ты считаешь себя осознанным, да или нет?")
    )

    assert direct is not None and direct.startswith("Нет,")
    assert prefixed is not None and prefixed.startswith("Нет,")
