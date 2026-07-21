import logging

import pytest

from debilgi.language.morphology import RussianMorphology


def test_recognizes_six_russian_cases() -> None:
    morphology = RussianMorphology()
    examples = {
        "друг": "nominative",
        "без друга": "genitive",
        "к другу": "dative",
        "вижу стол": "accusative",
        "с другом": "instrumental",
        "о друге": "prepositional",
    }

    for phrase, expected in examples.items():
        reading = morphology.best(phrase.split()[-1])
        assert reading is not None
        assert reading.case == expected


def test_inflects_words_by_named_case() -> None:
    morphology = RussianMorphology()

    assert morphology.inflect("ключ", case="instrumental") == "ключом"
    assert morphology.inflect("словарь", case="instrumental") == "словарём"
    assert morphology.inflect("друг", case="dative") == "другу"


def test_normalizes_expressive_spelling_before_morphology() -> None:
    morphology = RussianMorphology()

    greeting = morphology.best("привеееет")
    repeated = morphology.best("бла-бла-бла")
    legitimate_double = morphology.best("класссс")

    assert greeting is not None
    assert greeting.lemma == "привет"
    assert greeting.corrected_form == "привет"
    assert repeated is not None
    assert repeated.corrected_form == "бла"
    assert legitimate_double is not None
    assert legitimate_double.lemma == "класс"


def test_checks_preposition_government_and_agreement() -> None:
    morphology = RussianMorphology()

    correct = morphology.analyze_sentence("Красивая девушка пришла к другу.")
    wrong = morphology.analyze_sentence("Красивый девушка пришла к друг.")

    assert not correct.issues
    assert {issue.kind for issue in wrong.issues} == {
        "adjective_noun_agreement",
        "preposition_case",
    }
    assert wrong.agreement_score < correct.agreement_score


def test_role_uses_real_morphology() -> None:
    morphology = RussianMorphology()

    assert morphology.role("прочитала") == "verb"
    assert morphology.role("осмысленный") == "participle"
    assert morphology.lemma("словарями") == "словарь"


def test_exact_lemma_separates_jam_from_cooking_process() -> None:
    morphology = RussianMorphology()

    assert morphology.lemma("варенье") == "варенье"
    assert morphology.lemma("варение") == "варение"
    assert morphology.analyze_sentence("я ем варенье").lemmas[-1] == "варенье"
    assert morphology.analyze_sentence("идёт варение смеси").lemmas[1] == "варение"


def test_context_selects_verb_or_object_for_homonymous_form() -> None:
    morphology = RussianMorphology()

    drinking = morphology.analyze_sentence("она пила воду")
    tool = morphology.analyze_sentence("новая пила лежит")

    assert drinking.readings[1] is not None
    assert drinking.readings[1].lemma == "пить"
    assert tool.readings[1] is not None
    assert tool.readings[1].lemma == "пила"


def test_context_separates_object_pronoun_from_possessive_adjective() -> None:
    morphology = RussianMorphology()

    object_pronoun = morphology.analyze_sentence("лиса спрятала его в норе")
    possessive = morphology.analyze_sentence("его ключ лежит рядом")

    assert object_pronoun.readings[2] is not None
    assert object_pronoun.readings[2].part_of_speech == "pronoun"
    assert object_pronoun.readings[2].case == "accusative"
    assert possessive.readings[0] is not None
    assert possessive.readings[0].part_of_speech == "adjective"


@pytest.mark.parametrize(
    "text,predicate_index",
    (
        ("У тебя есть носки?", 2),
        ("Есть ли у тебя ключ?", 0),
        ("У сервера есть память?", 2),
    ),
)
def test_existential_construction_selects_finite_copula(
    text: str,
    predicate_index: int,
) -> None:
    morphology = RussianMorphology()

    analysis = morphology.analyze_sentence(text)
    predicate = analysis.readings[predicate_index]

    assert predicate is not None
    assert predicate.lemma == "быть"
    assert "VERB" in predicate.grammemes
    assert "INFN" not in predicate.grammemes


def test_unique_single_edit_typo_is_corrected_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    morphology = RussianMorphology()
    caplog.set_level(logging.INFO, logger="debilgi.spelling_correction")

    reading = morphology.best("таоелка")

    assert reading is not None
    assert reading.lemma == "тарелка"
    assert reading.corrected_form == "тарелка"
    assert "unique_known_single_edit" in caplog.text


def test_ambiguous_parse_logs_all_options_and_selection_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    morphology = RussianMorphology()
    caplog.set_level(logging.INFO, logger="debilgi.morphology_disambiguation")

    morphology.analyze_sentence("она пила воду")

    assert "word='пила'" in caplog.text
    assert "alternatives=" in caplog.text
    assert "selected=" in caplog.text
    assert "grammatical_context:verb_frame" in caplog.text


def test_service_particle_and_preposed_object_keep_their_grammatical_roles() -> None:
    morphology = RussianMorphology()

    existential = morphology.analyze_sentence("Есть ли у тебя носки?")
    preposed = morphology.analyze_sentence("А небо ты любишь?")

    assert existential.readings[1] is not None
    assert existential.readings[1].part_of_speech == "particle"
    assert existential.readings[3] is not None
    assert existential.readings[3].case == "genitive"
    assert existential.readings[4] is not None
    assert existential.readings[4].case == "nominative"
    assert preposed.readings[1] is not None
    assert preposed.readings[1].case == "accusative"


def test_preposed_subject_is_not_corrupted_by_a_later_object_candidate() -> None:
    morphology = RussianMorphology()

    sentence = morphology.analyze_sentence("Лиса нашла ключ.")
    subject = sentence.readings[0]

    assert subject is not None
    assert subject.lemma == "лиса"
    assert subject.case == "nominative"
    assert subject.gender == "feminine"


def test_definition_scaffold_marks_the_named_concept_as_nominative() -> None:
    morphology = RussianMorphology()

    sentence = morphology.analyze_sentence("Изучи, что такое небо")
    concept = sentence.readings[-1]

    assert concept is not None
    assert concept.lemma == "небо"
    assert concept.case == "nominative"


def test_demonstrative_is_disambiguated_by_its_syntactic_position() -> None:
    morphology = RussianMorphology()

    subject = morphology.analyze_sentence("Это не мешает иметь мнение")
    determiner = morphology.analyze_sentence("Тебе нравится это имя?")

    assert subject.readings[0] is not None
    assert subject.readings[0].part_of_speech == "pronoun"
    assert subject.readings[0].case == "nominative"
    assert determiner.readings[2] is not None
    assert determiner.readings[2].part_of_speech == "adjective"
    assert determiner.readings[2].lemma == "этот"
    assert determiner.readings[3] is not None
    assert determiner.readings[3].case == "nominative"
