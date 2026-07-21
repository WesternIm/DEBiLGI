from debilgi.language.entities import analyze_entities
from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.sense_resolution import (
    apply_sense_resolution,
    resolve_contextual_senses,
)
from debilgi.language.structured_utterance import analyze_structured_utterance

PAYLOAD = {
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
}


def word(
    surface: str,
    meanings: tuple[str, ...],
    *,
    selected: str | None = None,
    source: str = "dictionary_cache",
    local: str | None = None,
    personal: str | None = None,
) -> WordAnalysis:
    return WordAnalysis(
        surface_form=surface,
        normalized_form=surface.casefold(),
        lemma=surface.casefold(),
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=meanings,
        style_labels=(),
        source=source,
        confidence=ConfidenceProfile(0.95, 0.82, 0.72, 0.78, 0.72),
        selected_meaning=selected,
        local_meaning=local,
        personal_interpretation=personal,
    )


def chosen_meaning(text: str, analysis: WordAnalysis) -> str | None:
    structured = analyze_structured_utterance(text, (analysis,))
    resolution = resolve_contextual_senses(structured)
    decision = next(
        item
        for item in resolution.decisions
        if item.surface.casefold() == analysis.surface_form.casefold()
    )
    return decision.chosen.meaning if decision.chosen else None


def test_lowercase_person_label_beats_unrelated_media_title() -> None:
    analysis = word(
        "умник",
        (
            "сообразительный человек, иногда ироничное обращение",
            "американский телесериал о детективе",
        ),
    )

    meaning = chosen_meaning("ты что умник теперь?", analysis)

    assert meaning is not None
    assert "человек" in meaning
    assert "телесериал" not in meaning


def test_explicit_media_context_selects_media_sense() -> None:
    analysis = word(
        "умник",
        (
            "сообразительный человек, иногда ироничное обращение",
            "американский телесериал о детективе",
        ),
    )

    meaning = chosen_meaning("Сериал «Умник» вышел давно", analysis)

    assert meaning is not None
    assert "телесериал" in meaning


def test_server_slang_outranks_dictionary_meaning_in_its_scope() -> None:
    local = "игровой объект, который здесь считают чрезмерно сильным"
    analysis = word(
        "имба",
        ("устаревшее личное имя",),
        selected=local,
        source="server_slang",
        local=local,
    )

    meaning = chosen_meaning("этот меч имба", analysis)

    assert meaning == local


def test_discord_identity_beats_lexical_guess_and_is_applied_to_copy() -> None:
    text = "ferzevina открыла сервер"
    entities = analyze_entities(text, PAYLOAD)
    lexical = word("ferzevina", ("неизвестное название объекта",))
    structured = analyze_structured_utterance(text, (lexical,), entities)
    resolution = resolve_contextual_senses(structured)
    decision = next(item for item in resolution.decisions if item.surface == "ferzevina")

    assert decision.chosen is not None
    assert decision.chosen.source == "entity"
    assert decision.chosen.entity_kind == "person"

    applied = apply_sense_resolution(structured, resolution)
    token = next(item for item in applied.tokens if item.surface == "ferzevina")
    assert token.selected_meaning == "entity:person:ferzevina"
    assert token.lexical_source == "entity"


def test_all_candidates_and_numeric_components_remain_available_for_logging() -> None:
    analysis = word(
        "лук",
        (
            "оружие для стрельбы стрелами",
            "овощное растение, которое кладут в салат",
        ),
    )
    structured = analyze_structured_utterance("лук кладут в салат", (analysis,))
    resolution = resolve_contextual_senses(structured)
    decision = next(item for item in resolution.decisions if item.surface == "лук")

    assert len(decision.scores) == 2
    assert all(item.components for item in decision.scores)
    assert decision.scores[0].total >= decision.scores[1].total
    assert decision.chosen is not None
    assert "салат" in decision.chosen.meaning


def test_equal_unsupported_senses_remain_explicitly_unresolved() -> None:
    analysis = word(
        "лук",
        (
            "оружие для стрельбы стрелами",
            "огородное растение",
        ),
    )
    structured = analyze_structured_utterance("лук", (analysis,))
    resolution = resolve_contextual_senses(structured)
    decision = next(item for item in resolution.decisions if item.surface == "лук")

    assert decision.chosen is None
    assert decision.reason == "ambiguous_margin"
    assert decision.token_index in resolution.unresolved_indices


def test_personal_meaning_competes_with_reference_instead_of_replacing_its_record() -> None:
    personal = "личное название маленького складного фонаря"
    analysis = word(
        "фырбуль",
        ("редкое название растения",),
        source="dictionary_cache",
        personal=personal,
    )
    structured = analyze_structured_utterance("мой фырбуль снова светит", (analysis,))
    resolution = resolve_contextual_senses(structured)
    decision = next(item for item in resolution.decisions if item.surface == "фырбуль")

    assert len(decision.scores) == 2
    assert decision.chosen is not None
    assert decision.chosen.meaning == personal
    assert any("растения" in item.candidate.meaning for item in decision.scores)
