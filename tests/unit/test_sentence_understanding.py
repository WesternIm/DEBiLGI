from debilgi.language.lexicon import ConfidenceProfile, UnknownKind, WordAnalysis
from debilgi.language.sentence_understanding import understand_utterance


def analysis(
    word: str,
    *,
    lemma: str,
    part: str,
    unknown: bool = False,
) -> WordAnalysis:
    return WordAnalysis(
        surface_form=word,
        normalized_form=word,
        lemma=lemma,
        part_of_speech=part,
        grammatical_features={},
        dictionary_meanings=() if unknown else ("известное значение",),
        style_labels=(),
        source="unknown" if unknown else "test_dictionary",
        confidence=ConfidenceProfile(0.8, 0.0 if unknown else 0.8, 0.7, 0.5, 0.2),
        unknown_kind=UnknownKind.UNKNOWN_FORM if unknown else None,
    )


def test_one_unknown_word_does_not_destroy_a_connected_sentence() -> None:
    understanding = understand_utterance(
        "Я закинул трошку в сумку",
        (
            analysis("закинул", lemma="закинуть", part="VERB"),
            analysis("трошку", lemma="трошка", part="NOUN", unknown=True),
            analysis("сумку", lemma="сумка", part="NOUN"),
        ),
    )

    assert understanding.unknown_terms == ("трошка",)
    assert {"закинуть", "сумка"} <= set(understanding.known_terms)
    assert any(relation.relation == "закинуть" for relation in understanding.relations)
    assert understanding.sufficient is True


def test_wholly_unknown_fragment_remains_a_phrase_level_gap() -> None:
    understanding = understand_utterance(
        "фырба трошка",
        (
            analysis("фырба", lemma="фырба", part="NOUN", unknown=True),
            analysis("трошка", lemma="трошка", part="NOUN", unknown=True),
        ),
    )

    assert understanding.sufficient is False
    assert understanding.subject_phrase == "фырба трошка"
    assert understanding.unknown_terms == ("фырба", "трошка")


def test_service_words_are_logged_and_never_become_relation_endpoints(caplog) -> None:
    with caplog.at_level("INFO", logger="debilgi.relation_filter"):
        understanding = understand_utterance(
            "типо кто самый добрый человек?",
            (),
        )

    endpoints = {
        endpoint
        for relation in understanding.relations
        for endpoint in (relation.source, relation.target)
    }
    assert not endpoints & {"типо", "кто", "самый"}
    assert any(
        {relation.source, relation.target} == {"добрый", "человек"}
        for relation in understanding.relations
    )
    assert any("source='типо'" in record.message for record in caplog.records)


def test_question_pronoun_can_shape_a_question_without_becoming_a_concept() -> None:
    understanding = understand_utterance("какой человек написал книгу?", ())

    assert "какой" not in understanding.content_terms
    assert all(
        "какой" not in {relation.source, relation.target}
        for relation in understanding.relations
    )


def test_resolved_dialogue_object_reaches_semantic_relations_not_only_retrieval() -> None:
    understanding = understand_utterance(
        "А почему не любишь?",
        (analysis("любишь", lemma="любить", part="VERB"),),
        ("Ты рассказывал про небо.",),
        resolved_bindings=(("direct_object", "небо"),),
    )

    assert "небо" in understanding.known_terms
    assert "небо" in understanding.context_links
    assert any(
        relation.source == "ты"
        and relation.relation == "любить"
        and relation.target == "небо"
        and relation.kind == "resolved_action_target"
        for relation in understanding.relations
    )


def test_dependency_roles_drive_experiencer_and_control_relations() -> None:
    preference = understand_utterance("Тебе нравится это имя?", ())
    capability = understand_utterance(
        "Ты можешь без тела иметь мнение и позицию?",
        (),
    )

    assert any(
        relation.source == "ты"
        and relation.relation == "нравиться"
        and relation.target == "имя"
        and relation.kind == "experiencer_state"
        for relation in preference.relations
    )
    controlled = {
        (relation.source, relation.relation, relation.target, relation.kind)
        for relation in capability.relations
    }
    assert ("ты", "иметь", "мнение", "controlled_action") in controlled
    assert ("ты", "иметь", "позиция", "controlled_action") in controlled
    assert ("ты", "мочь", "иметь", "action_complement") in controlled
