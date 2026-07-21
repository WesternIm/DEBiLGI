from debilgi.language.lexical_matching import (
    fuzzy_match_allowed,
    same_lexical_family,
)


def test_false_friends_require_exact_identity() -> None:
    assert same_lexical_family("варенье", "варенье") is True
    assert same_lexical_family("варенье", "варение") is False
    assert same_lexical_family("стать", "статья") is False
    assert fuzzy_match_allowed("вол", "вёл") is False


def test_unprotected_related_forms_can_still_use_conservative_fuzzy_match() -> None:
    assert same_lexical_family("отвечать", "ответить") is True
