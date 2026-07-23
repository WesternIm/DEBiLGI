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
    meanings: tuple[str, ...] = (),
) -> WordAnalysis:
    dictionary_meanings = meanings or ((meaning,) if meaning else ())
    return WordAnalysis(
        surface_form=surface,
        normalized_form=surface,
        lemma=None if unknown else (lemma or surface),
        part_of_speech=part,
        grammatical_features={},
        dictionary_meanings=dictionary_meanings,
        style_labels=(),
        source="unknown" if unknown else "opencorpora",
        confidence=ConfidenceProfile(0.1, 0.0, 0.0, 0.0, 0.0),
        unknown_kind=unknown,
        selected_meaning=meaning,
    )


def test_only_irreducible_social_routines_keep_fixed_surfaces() -> None:
    assert compose_dialogue_reply(RecoveryEvidence("ауу")) == "Я здесь. Говори."
    assert compose_dialogue_reply(RecoveryEvidence("але")) == "Я здесь. Говори."
    assert compose_recovery(RecoveryEvidence("как дела?")) == "Я в работе. А ты как?"
    assert compose_recovery(RecoveryEvidence("спасибо")) == "Пожалуйста."
    assert compose_recovery(RecoveryEvidence("не смейся")) == "Хорошо, без шуток."


def test_person_label_is_a_dynamic_social_frame_not_encyclopedic_retrieval() -> None:
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
    assert "оцениваешь меня как «обманщик»" in reply
    assert "на каком ответе" in reply.casefold()
    assert "телесериал" not in reply


def test_unknown_word_failure_is_single_terminal_lookup_report() -> None:
    replies = {unknown_word_fallback("трощкэ", variant) for variant in range(10)}

    assert replies == {"Поиск не дал значения для «трощкэ». Уточни написание."}


def test_unknown_word_uses_found_spelling_candidates() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            prompt="закинул трощкэ?",
            analyses=(
                analysis("закинул", part="VERB"),
                analysis("трощкэ", unknown=UnknownKind.UNKNOWN_FORM),
            ),
            similar_words=(("трощкэ", ("трошки", "крошка")),),
        )
    )

    assert "после поиска" in reply.casefold()
    assert "«трошки»" in reply
    assert "«крошка»" in reply


def test_found_fact_is_used_instead_of_recovery_language() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            prompt="что такое мороженое?",
            facts=("Мороженое — замороженный сладкий десерт.",),
        )
    )

    assert reply == "Мороженое — замороженный сладкий десерт."


def test_ambiguous_meaning_asks_from_actual_found_variants() -> None:
    reply = compose_recovery(
        RecoveryEvidence(
            "что значит ключ?",
            analyses=(
                analysis(
                    "ключ",
                    unknown=UnknownKind.AMBIGUOUS_CONTEXT,
                    meanings=(
                        "инструмент для замка",
                        "источник воды",
                    ),
                ),
            ),
        )
    )

    assert "инструмент для замка" in reply
    assert "источник воды" in reply


def test_content_domains_are_not_answered_by_recovery_scripts() -> None:
    prompts = (
        "кто твоя мама",
        "Ты осознан?",
        "Что такое осознанность?",
        "ладно, иди отдыхай",
        "Да, это команда",
    )

    for prompt in prompts:
        assert compose_dialogue_reply(RecoveryEvidence(prompt)) is None
        assert not is_social_routine(prompt)


def test_extracts_single_word_from_explicit_lexical_lookup() -> None:
    assert lexical_lookup_target("Узнай что такое отдыхать, дурень") == "отдыхать"
    assert lexical_lookup_target("Что значит слово «кринж»?") == "кринж"
    assert lexical_lookup_target("Что такое слово фырбуль?") == "фырбуль"
    assert lexical_lookup_target("Что значит API?") == "api"
    assert lexical_lookup_target("Что значит т.е.?") == "те"
    assert lexical_lookup_target("поговорим об отдыхе") is None


def test_identity_detection_remains_routing_only() -> None:
    assert is_identity_question("Как тебя зовут?")
    assert is_identity_question("кто ты")
    assert not is_identity_question("кто твоя мама")
