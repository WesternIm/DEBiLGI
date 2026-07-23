from debilgi.language.abbreviations import (
    abbreviation_meanings,
    abbreviation_mentions,
    extract_explicit_abbreviation_definition,
    generated_abbreviation_aliases,
    is_abbreviation_candidate,
)
from debilgi.language.tokenizer import tokenize


def test_recognizes_uppercase_dotted_and_known_lowercase_abbreviations() -> None:
    assert is_abbreviation_candidate("ИИ")
    assert is_abbreviation_candidate("т.е.")
    assert is_abbreviation_candidate("api")
    assert not is_abbreviation_candidate("кот")
    assert abbreviation_mentions("ИИ, API и т.е.") == ("ии", "api", "те")
    assert tokenize("т.е. это сокращение") == ("т.е.", "это", "сокращение")


def test_common_abbreviation_has_a_semantic_expansion() -> None:
    assert abbreviation_meanings("ИИ") == ("искусственный интеллект",)
    assert abbreviation_meanings("GTA") == ("Grand Theft Auto",)


def test_learns_short_first_and_parenthetical_expansions() -> None:
    assert extract_explicit_abbreviation_definition(
        "СУБД расшифровывается как система управления базами данных."
    ) == ("субд", "система управления базами данных")
    assert extract_explicit_abbreviation_definition(
        "Это система управления базами данных (СУБД)."
    ) == ("субд", "система управления базами данных")


def test_generates_reusable_aliases_from_a_full_concept_title() -> None:
    assert generated_abbreviation_aliases("Искусственный интеллект") == ("ии",)
    aliases = generated_abbreviation_aliases("Grand Theft Auto V")
    assert {"gtav", "gta", "gta 5"} <= set(aliases)
