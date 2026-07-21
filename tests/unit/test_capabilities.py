from debilgi.language.capabilities import (
    assess_capability_question,
    realize_capability_assessment,
)
from debilgi.language.entities import analyze_entities
from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.structured_utterance import analyze_structured_utterance


def structure(text: str):
    return analyze_structured_utterance(
        text,
        entity_analysis=analyze_entities(text, {}),
    )


def test_bot_bodily_capability_is_grounded_in_non_physical_agent_type() -> None:
    assessment = assess_capability_question(structure("А ты умеешь летать?"))

    assert assessment is not None
    assert assessment.subject_kind == "bot"
    assert assessment.action_lemma == "летать"
    assert not assessment.supported
    assert realize_capability_assessment(assessment) == "Нет, я не умею летать."


def test_bot_external_action_is_inferred_from_agent_boundary_not_one_action() -> None:
    assessment = assess_capability_question(structure("А ты умеешь копать землю?"))

    assert assessment is not None
    assert not assessment.supported
    assert assessment.reason == "action_requires_unavailable_external_agency"
    assert realize_capability_assessment(assessment) == (
        "Нет, я не умею копать землю."
    )


def test_bot_internal_action_is_supported_by_the_same_capability_model() -> None:
    assessment = assess_capability_question(structure("А ты умеешь анализировать текст?"))

    assert assessment is not None
    assert assessment.supported
    assert assessment.reason == "internal_computational_action"
    assert realize_capability_assessment(assessment) == (
        "Да, я умею анализировать текст."
    )


def test_external_directive_is_grounded_in_agent_boundary() -> None:
    assessment = assess_capability_question(structure("Выкинь штурвал"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert realize_capability_assessment(assessment) == (
        "Я не могу выкинуть штурвал."
    )


def test_future_tense_action_request_is_not_retrieved_as_an_old_fact() -> None:
    assessment = assess_capability_question(structure("Откроешь мне бутылку?"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert assessment.action_lemma == "открыть"
    assert assessment.action_object is not None
    assert realize_capability_assessment(assessment) == (
        "Я не могу открыть бутылку."
    )


def test_past_bodily_experience_is_not_treated_as_missing_topic_data() -> None:
    assessment = assess_capability_question(structure("А яблоко ты ел?"))

    assert assessment is not None
    assert assessment.mode == "experience"
    assert realize_capability_assessment(assessment) == "Нет, я не ел яблоко."


def _word(lemma: str, meaning: str) -> WordAnalysis:
    return WordAnalysis(
        surface_form=lemma,
        normalized_form=lemma,
        lemma=lemma,
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=(meaning,),
        style_labels=(),
        source="dictionary_cache",
        confidence=ConfidenceProfile(0.9, 0.9, 0.9, 0.9, 0.9),
    )


def test_polysemous_past_action_uses_the_objects_semantic_class() -> None:
    physical = assess_capability_question(
        structure("Ты терял когда-то ключи?"),
        analyses=(_word("ключ", "предмет для отпирания замка"),),
    )
    digital = assess_capability_question(
        structure("Ты терял когда-то данные?"),
        analyses=(_word("данные", "цифровая информация и запись"),),
    )

    assert physical is not None
    assert physical.mode == "experience"
    assert realize_capability_assessment(physical) == "Нет, я не терял ключи."
    assert digital is None


def test_human_profession_cannot_fly_without_technology() -> None:
    assessment = assess_capability_question(structure("Как думаешь? Астрономы умеют летать?"))

    assert assessment is not None
    assert assessment.subject_lemma == "астроном"
    assert assessment.reason == "human_requires_technology"
    assert realize_capability_assessment(assessment) == (
        "Нет, астрономы не могут летать."
    )


def test_unknown_or_naturally_flying_subject_is_not_forced_to_no() -> None:
    assert assess_capability_question(structure("Птицы умеют летать?")) is None
    assert assess_capability_question(structure("Люди умеют плавать?")) is None


def test_abstract_evaluative_state_does_not_require_a_body() -> None:
    assessment = assess_capability_question(
        structure("Ты можешь без тела иметь мнение и позицию?")
    )

    assert assessment is not None
    assert assessment.supported
    assert assessment.action_lemma == "иметь"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "мнение"
    assert assessment.reason == "internal_cognitive_or_evaluative_state"
    assert realize_capability_assessment(assessment) == (
        "Да, я умею иметь мнение."
    )
