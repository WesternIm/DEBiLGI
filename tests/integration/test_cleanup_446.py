"""Stage 4.46: cleanup of the debts of 4.43–4.45 (flag ``DEBILGI_THOUGHT_CONTENT`` unless noted)."""

import pytest

from debilgi.cognition.affect import hostility_cues
from debilgi.language.dialogue_learning import ellipsis_variable, exchange_analysis


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "0")


def _lemmas(text):
    return [t.lemma for t in exchange_analysis(text)[0].tokens if t.surface[:1].isalpha()]


def test_the_three_spellings_of_the_particle_read_the_same(flag_on):
    expected = _lemmas("ты балбес что ли?")
    assert expected[-2:] == ["что", "ли"]
    assert _lemmas("ты балбес что-ли?") == expected
    assert _lemmas("ты балбес чтоли?") == expected


def test_a_hyphen_particle_after_any_word_is_split(flag_on):
    assert _lemmas("он-ли это сделал?")[:2] == ["он", "ли"]


def test_an_accusation_with_the_particle_is_an_attack_after_a_complaint(flag_on):
    for spelling in ("что-ли", "что ли", "чтоли"):
        assert hostility_cues(f"ты балбес {spelling}?", True, ()) == ("escalation:балбес",)
    # a plain question about the addressee stays a question
    assert hostility_cues("ты честный?", True, ()) == ()
    assert hostility_cues("ты значит у нас тут честный?", True, ()) == ()
    # nobody was hostile a moment ago: no attack by the particle alone
    assert hostility_cues("ты балбес что-ли?", False, ()) == ()


def test_purpose_without_content_is_the_variable_of_the_delivered_turn(flag_on):
    for text in ("для чего?", "а для чего?", "зачем?", "а зачем?"):
        assert ellipsis_variable(text) == {"role": "cause", "relation": "cause", "word": "зачем"}
    # a question about a thing is not an omission: the thing is named
    assert ellipsis_variable("а для чего он нужен?") is None
    assert ellipsis_variable("мыло для чего?") is None


def test_purpose_variable_is_off_with_the_flag_off(flag_off):
    assert ellipsis_variable("для чего?") is None
    assert ellipsis_variable("а зачем?") is None


def test_why_do_you_think_so_is_the_variable_of_the_view_just_delivered(flag_on):
    assert ellipsis_variable("а почему ты так думаешь?") == {"role": "cause", "relation": "cause"}
    assert ellipsis_variable("почему ты так думаешь?") == {"role": "cause", "relation": "cause"}
    assert ellipsis_variable("почему ты спишь?") is None


def test_why_do_you_think_so_is_not_read_with_the_flag_off(flag_off):
    assert ellipsis_variable("а почему ты так думаешь?") is None


def test_the_particle_spellings_are_untouched_with_the_flag_off(flag_off):
    from debilgi.language.tokenizer import tokenize

    assert "что-ли" in tokenize("ты балбес что-ли?")
    assert "чтоли" in tokenize("ты балбес чтоли?")
