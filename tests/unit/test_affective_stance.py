from dataclasses import replace

from debilgi.language.affective_stance import affective_stances
from debilgi.language.copying import is_input_copy
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.personality.expression import PersonalityExpression

BASE = PersonalityExpression(
    genome_signature="test",
    chaos=0.5,
    curiosity=0.5,
    impulsivity=0.5,
    expressiveness=0.5,
    memory_attachment=0.5,
    novelty_drive=0.5,
    coherence_drive=0.8,
    target_tokens=16,
    candidate_count=32,
    warmth=0.5,
    suspicion=0.5,
    assertiveness=0.5,
)


def test_dative_infinitive_is_read_as_a_complete_attitude_proposition():
    structure = analyze_structured_utterance("Мне плевать на твои идеологии")
    clause = structure.clauses[0]

    assert structure.tokens[clause.predicate_index].lemma == "плевать"
    assert {structure.tokens[index].lemma for index in clause.indirect_object_indices} == {"я"}
    assert structure.tokens[clause.complements[0].object_indices[0]].lemma == "идеология"


def test_stance_revoices_roles_and_changes_meaning_instead_of_echoing_the_turn():
    prompt = "Мне плевать на твои идеологии"
    candidates = affective_stances(prompt, personality=BASE)

    assert {candidate.operation for candidate in candidates} == {
        "align_attitude",
        "resist_attitude",
    }
    assert all("мои идеологии" in candidate.text.casefold() for candidate in candidates)
    assert any("не плевать" in candidate.text.casefold() for candidate in candidates)
    assert any("тоже плевать" in candidate.text.casefold() for candidate in candidates)
    resistance = next(candidate for candidate in candidates if not candidate.aligned)
    assert not is_input_copy(resistance.text, prompt)


def test_relationship_emotion_changes_which_stance_has_more_weight():
    warm = affective_stances(
        "Мне плевать на твои идеологии",
        affect_features=(("emotion_affiliation", 1.0), ("emotion_valence", 1.0)),
        personality=replace(BASE, warmth=0.8, suspicion=0.2),
    )
    hostile = affective_stances(
        "Мне плевать на твои идеологии",
        affect_features=(("emotion_affiliation", -1.0), ("emotion_valence", -1.0)),
        personality=replace(BASE, warmth=0.2, suspicion=0.8),
    )

    assert warm[0].aligned is True
    assert hostile[0].aligned is False


def test_stance_mechanism_does_not_answer_questions_or_unrelated_statements():
    assert not affective_stances("Тебе решать?", personality=BASE)
    assert not affective_stances("Кот любит рыбу.", personality=BASE)
