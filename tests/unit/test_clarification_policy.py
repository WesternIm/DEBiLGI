from debilgi.language.clarification_policy import (
    clarification_rejection,
    lexical_question_target,
    requests_word_definition,
)
from debilgi.language.lexicon import extract_explicit_definition


def test_word_definition_questions_are_rejected_for_known_and_unknown_terms():
    for text in ("Что ты подразумеваешь под «салфеткой»?", "Что значит кваргония?",
                 "Что такое стромба?", "Объясни значение слова кваргония."):
        assert requests_word_definition(text)
        assert clarification_rejection(text, ()) == "autonomous_dictionary_lookup_required"
    assert not requests_word_definition("Что ты хочешь узнать?")
    assert not requests_word_definition("Салфетка является бумагой.")


def test_question_target_preserves_lemma_without_treating_statements_as_questions():
    assert lexical_question_target("Что ты подразумеваешь под «салфеткой»?") == "салфетка"
    assert lexical_question_target("Я подразумеваю под этим салфетку.") is None
    assert lexical_question_target("Ты подразумеваешь кошку или собаку?") is None


def test_discourse_particles_are_not_learned_as_definitions():
    assert extract_explicit_definition("Ну это пока что ты не считаешь себя тупым") is None
    assert extract_explicit_definition("Ты это взял?") is None
    assert extract_explicit_definition("Так это не вопрос") is None
    assert extract_explicit_definition("Салфетка это бумага для гигиены") == (
        "салфетка", "бумага для гигиены",
    )
