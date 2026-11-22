"""Stage 4.35 (DEBILGI_THOUGHT_CONTENT): attacks on live profanity, lemmas of unknown words.

An attack is recognised from the dictionary register, the structure of the sentence, words the
dictionary does not know, and the irritation already in force; never from a phrase list.
"""

from datetime import timedelta

import pytest

from debilgi.cognition.affect import hostility_cues, irritation_reading
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from tests.integration.test_thought_content_434 import CHANNEL, NOW, OTHER, USER, db, ev, said  # noqa: F401


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("DEBILGI_THOUGHT_CONTENT", "1")


# ------------------------------------------------------------ lemmas of words the dictionary lacks
def test_an_unknown_word_keeps_its_own_lemma(flag_on):
    reading = DEFAULT_MORPHOLOGY.best("долбоёб")
    assert reading.lemma == "долбоёб" and reading.part_of_speech == "noun"
    assert DEFAULT_MORPHOLOGY.inflect("долбоёб", case="nominative", number="singular") == "долбоёб"


def test_the_old_lemma_is_kept_without_the_flag(monkeypatch):
    monkeypatch.delenv("DEBILGI_THOUGHT_CONTENT", raising=False)
    fresh = type(DEFAULT_MORPHOLOGY)()  # the parse cache is per instance
    assert fresh.best("долбоёб").lemma != "долбоёб"


@pytest.mark.parametrize("word,lemma", [
    ("расказала", "расказать"), ("подустал", "подустать"), ("фрилансом", "фриланс"),
    ("гандон", "гандон"), ("нервничаешь", "нервничать"),
])
def test_inflection_of_unknown_words_is_not_broken(flag_on, word, lemma):
    assert DEFAULT_MORPHOLOGY.best(word).lemma == lemma


def test_a_known_word_is_never_touched(flag_on):
    assert DEFAULT_MORPHOLOGY.best("шатал").lemma == "шатать"
    assert DEFAULT_MORPHOLOGY.best("петушком").lemma in {"петушком", "петушок"}


# ------------------------------------------------------------------ recognising an attack
@pytest.mark.parametrize("text", ["ты долбоёб?", "ну и ты петушарв", "але пидарас", "ты гандон",
                                  "пошел нахуй"])
def test_attacks_are_recognised_without_a_phrase_list(flag_on, text):
    assert hostility_cues(text)


@pytest.mark.parametrize("text", [
    "да и похуй", "пиздец погода", "хуйня какая-то",
    "Раздолбай обыкновенный, у меня брат такой же", "ты молодец", "чем ты занимаешся",
    "а ты где живеш?", "ты Вася?", "привет Лида, чем занимаешься",
    "ты студент?", "нравится быть программистом?",
])
def test_impersonal_profanity_typos_and_neutral_questions_are_not_attacks(flag_on, text):
    assert hostility_cues(text) == ()


def test_evidence_kinds_are_named(flag_on):
    assert hostility_cues("ты долбоёб?")[0].startswith("unplaced:")
    assert hostility_cues("ты гандон")[0].startswith("register:")


@pytest.mark.parametrize("text", ["ты дурак", "нравится быть петушком?", "ты балбес что-ли?"])
def test_a_word_of_unknown_sign_is_an_attack_only_after_an_attack(flag_on, text):
    assert hostility_cues(text) == ()
    assert hostility_cues(text, True)


def test_a_positive_word_is_never_an_attack(flag_on):
    assert hostility_cues("ты молодец", True) == ()
    assert hostility_cues("ты красава", True) == ()


# ------------------------------------------------------------------- R and the escalation
async def test_an_unknown_noun_after_an_attack_is_hostile_and_adds_the_irritation_in_force(
    db, flag_on,  # noqa: F811
):
    await said(db, 1, "ты долбоёб?", ago=timedelta(seconds=40))
    first = await irritation_reading(db, ev("a", "ты долбоёб?"), "ты долбоёб?")
    assert first.value > 0.5 and first.hostile
    reading = await irritation_reading(db, ev("b", "нравится быть петушком?"),
                                       "нравится быть петушком?")
    assert reading.hostile and any(
        "hostile_escalation" in i["kinds"] for i in reading.impulses)


async def test_the_same_words_from_a_calm_person_are_not_hostile(db, flag_on):  # noqa: F811
    reading = await irritation_reading(db, ev("b", "ты дурак"), "ты дурак")
    assert reading.hostile == () and reading.value == 0.0


async def test_the_escalation_fades_with_the_clock(db, flag_on):  # noqa: F811
    await said(db, 1, "пошел нахуй", ago=timedelta(seconds=20))
    later = ev("b", "ты дурак", at=NOW + timedelta(hours=1))
    assert (await irritation_reading(db, later, "ты дурак")).hostile == ()


async def test_impersonal_profanity_does_not_raise_irritation(db, flag_on):  # noqa: F811
    for n, text in enumerate(["да и похуй", "пиздец погода", "хуйня какая-то", "ахуеть как жарко"]):
        await said(db, n, text, ago=timedelta(seconds=60 - n))
    reading = await irritation_reading(db, ev("b", "похуй мне"), "похуй мне")
    assert reading.value == 0.0 and reading.hostile == ()


async def test_an_attack_by_someone_else_does_not_aggravate(db, flag_on):  # noqa: F811
    await said(db, 1, "пошел нахуй", actor=OTHER, ago=timedelta(seconds=20))
    reading = await irritation_reading(db, ev("b", "ты дурак"), "ты дурак")
    assert reading.hostile == () and CHANNEL and USER
