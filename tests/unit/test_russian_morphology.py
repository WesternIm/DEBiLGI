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
    assert morphology.inflect("Германия", case="accusative") == "Германию"


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


def test_numeric_literals_have_a_real_category_and_stay_atomic() -> None:
    morphology = RussianMorphology()

    integer = morphology.best("42")
    decimal = morphology.analyze_sentence("Версия 2.0 работает")
    percentage = morphology.best("10%")

    assert integer is not None
    assert integer.part_of_speech == "number"
    assert integer.analysis_kind == "numeric_literal"
    assert decimal.tokens == ("версия", "2.0", "работает")
    assert decimal.readings[1] is not None
    assert decimal.readings[1].part_of_speech == "number"
    assert percentage is not None
    assert percentage.part_of_speech == "number"
    assert "percentage" in percentage.grammemes


def test_auxiliary_selects_infinitive_and_its_accusative_object() -> None:
    morphology = RussianMorphology()

    sentence = morphology.analyze_sentence("Буду печь пирог")

    assert sentence.readings[1] is not None
    assert sentence.readings[1].part_of_speech == "verb"
    assert "INFN" in sentence.readings[1].grammemes
    assert sentence.readings[2] is not None
    assert sentence.readings[2].case == "accusative"


def test_numeral_subject_selects_plural_past_instead_of_imperative() -> None:
    morphology = RussianMorphology()

    sentence = morphology.analyze_sentence("Два кота пришли")
    predicate = sentence.readings[2]

    assert predicate is not None
    assert predicate.lemma == "прийти"
    assert predicate.number == "plural"
    assert predicate.tense == "past"
    assert "impr" not in predicate.grammemes


def test_nominal_clause_prefers_nominative_agreement() -> None:
    morphology = RussianMorphology()

    sentence = morphology.analyze_sentence("Этот замок старый")

    cases = tuple(
        reading.case if reading is not None else None
        for reading in sentence.readings
    )

    assert cases == (
        "nominative",
        "nominative",
        "nominative",
    )


def test_unknown_suffix_guess_is_marked_and_cannot_claim_full_confidence() -> None:
    morphology = RussianMorphology()

    guessed = morphology.best("мурмякает")

    assert guessed is not None
    assert guessed.part_of_speech == "verb"
    assert not guessed.is_known
    assert guessed.analysis_kind == "guessed"
    assert guessed.score <= 0.38


def test_colloquial_ka_derivative_is_not_corrected_to_base_dative() -> None:
    morphology = RussianMorphology()

    reading = morphology.best("мусорку")
    sentence = morphology.analyze_sentence("Мусорку выкинь")

    assert reading is not None
    assert reading.lemma == "мусорка"
    assert reading.case == "accusative"
    assert reading.corrected_form is None
    assert sentence.readings[0] is not None
    assert sentence.readings[0].lemma == "мусорка"
    assert sentence.readings[0].case == "accusative"


def test_exposes_derivational_verb_features_and_preserves_full_readings() -> None:
    morphology = RussianMorphology()

    verb = morphology.best("прочитала")
    productive_readings = morphology.parse_word("супер-пупер")
    genitive_genders = {
        reading.gender
        for reading in productive_readings
        if reading.case == "genitive"
        and reading.part_of_speech == "adjective"
    }

    assert verb is not None
    assert verb.features().items() >= {
        "aspect": "perfective",
        "mood": "indicative",
        "transitivity": "transitive",
    }.items()
    assert genitive_genders >= {"masculine", "feminine", "neuter"}
