from debilgi.language.entities import (
    analyze_entities,
    identity_reply,
    render_message_content,
)
from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.sentence_understanding import understand_utterance

PAYLOAD = {
    "bot_mentioned": True,
    "bot_user_id": 123,
    "author_identity": {
        "id": 100,
        "username": "ferzevina",
        "display_name": "Эвелина",
    },
    "bot_identity": {
        "id": 123,
        "username": "DEBiLGI",
        "display_name": "DEBiLGI",
    },
    "mentioned_users": [
        {"id": 456, "username": "furybam", "display_name": "FuryBaM"}
    ],
}


def unknown_word(surface: str) -> WordAnalysis:
    return WordAnalysis(
        surface_form=surface,
        normalized_form=surface.casefold(),
        lemma=None,
        part_of_speech=None,
        grammatical_features={},
        dictionary_meanings=(),
        style_labels=(),
        source="unknown",
        confidence=ConfidenceProfile(0.1, 0.0, 0.0, 0.0, 0.0),
        unknown_kind=UnknownKind.UNKNOWN_FORM,
    )


def test_rendering_removes_bot_address_but_keeps_other_users_semantic() -> None:
    rendered = render_message_content(
        "<@123> как думаешь, <@456> прав?",
        PAYLOAD,
    )

    assert "DEBiLGI" not in rendered
    assert "FuryBaM" in rendered
    assert "как думаешь" in rendered


def test_entity_analysis_distinguishes_nickname_action_and_object() -> None:
    analysis = analyze_entities("FuryBaM открыл сервер", PAYLOAD)

    assert any(
        item.canonical == "furybam"
        and item.kind == "person"
        and item.role == "mentioned_person"
        for item in analysis.mentions
    )
    assert "открыть" in analysis.action_terms
    assert "сервер" in analysis.object_terms


def test_known_nickname_is_not_treated_as_an_unknown_dictionary_word() -> None:
    entities = analyze_entities("FuryBaM ответил", PAYLOAD)
    understanding = understand_utterance(
        "FuryBaM ответил",
        (unknown_word("FuryBaM"), unknown_word("ответил")),
        entity_analysis=entities,
    )

    assert "furybam" in understanding.known_terms
    assert "furybam" not in understanding.unknown_terms
    assert "ответил" in understanding.unknown_terms


def test_identity_questions_use_discord_facts() -> None:
    assert identity_reply("как меня зовут?", PAYLOAD) == "Ты — Эвелина (@ferzevina)."
    assert identity_reply("какой у тебя ник?", PAYLOAD) == "В этом Discord я — @DEBiLGI."
    assert identity_reply("кто такой FuryBaM?", PAYLOAD) == "Это FuryBaM (@furybam)."


def test_grammatical_animacy_does_not_turn_an_object_into_a_person() -> None:
    analysis = analyze_entities("Краба выбираешь или рака?", PAYLOAD)

    assert not any(item.canonical in {"краб", "рак"} for item in analysis.mentions)
    assert {"краб", "рак"} <= set(analysis.object_terms)


def test_sentence_initial_capitalization_is_not_identity_evidence() -> None:
    analysis = analyze_entities("Книга лежит на столе", PAYLOAD)

    assert not analysis.mentions
    assert "книга" in analysis.object_terms
