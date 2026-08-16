from debilgi.language.grammar import role
from debilgi.language.pragmatics import analyze_pragmatics


def test_particles_keep_negation_modality_and_question_force() -> None:
    analysis = analyze_pragmatics("Неужели он бы вовсе не пришёл?")

    assert {"question_or_doubt", "hypothetical_or_softening", "reinforced_negation"} <= (
        analysis.functions
    )
    assert analysis.question_force >= 0.8
    assert role("неужели") == "particle"


def test_interjections_and_discourse_markers_are_not_topic_content() -> None:
    analysis = analyze_pragmatics("Ого, ну это, кажется, важно")

    assert {cue.category for cue in analysis.cues} >= {
        "interjection",
        "interjection_particle",
        "discourse_marker",
    }
    assert analysis.emotional_intensity >= 0.8
    assert analysis.uncertainty >= 0.6
    assert role("ого") == "interjection"


def test_stretched_chat_interjections_keep_their_pragmatic_function() -> None:
    hesitation = analyze_pragmatics("Хмм")
    laughter = analyze_pragmatics("Хехехех")
    attention = analyze_pragmatics("ауу")

    assert "hesitation" in hesitation.functions
    assert "amusement" in laughter.functions
    assert "attention_call" in attention.functions


def test_laughter_with_leading_vowel_and_chat_emotions_are_pragmatic() -> None:
    for text in ("ахахахах", "АХАХА", "hahaha", "лол", "кек"):
        assert "amusement" in analyze_pragmatics(text).functions

    assert "coarse_emphasis" in analyze_pragmatics("ой бля").functions
    assert "shock_or_frustration" in analyze_pragmatics("пиздец").functions


def test_emoji_and_repeated_punctuation_carry_emotional_force() -> None:
    analysis = analyze_pragmatics("СЕРЬЁЗНО??? 😂")

    assert {"intense_question", "amusement"} <= analysis.functions
    assert analysis.emotional_intensity >= 0.85


def test_chat_elongation_and_reduplication_keep_social_meaning() -> None:
    greeting = analyze_pragmatics("привеееет")
    realization = analyze_pragmatics("ооо")
    placeholder = analyze_pragmatics("бла-бла-бла")
    laughter = analyze_pragmatics("ха-ха-ха")

    assert "greeting" in greeting.functions
    assert "surprise_or_realization" in realization.functions
    assert "dismissive_placeholder" in placeholder.functions
    assert "amusement" in laughter.functions
    assert realization.emotional_intensity >= 0.7


def test_colloquial_profanity_is_emphasis_not_a_content_noun() -> None:
    analysis = analyze_pragmatics("Молодец хуле")

    assert "coarse_emphasis" in analysis.functions


def test_clause_final_imperative_can_be_a_parenthetical_attention_marker() -> None:
    analysis = analyze_pragmatics("Твой ответ логичный, прикинь")

    assert "parenthetical_attention_marker" in analysis.functions
    assert "parenthetical_attention_marker" not in analyze_pragmatics(
        "Прикинь сумму"
    ).functions
