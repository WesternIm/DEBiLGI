from debilgi.language.construction_nlg import (
    ConstructionSpec,
    ResponseContentPlan,
    SemanticClause,
    SemanticTerm,
    realize_content_plan,
)


def _possession(object_lemma: str, object_surface: str, *, truth: bool, number=None):
    return ResponseContentPlan(
        response_act="relation_answer",
        answer_polarity=truth,
        clauses=(
            SemanticClause(
                "possession",
                SemanticTerm("я", "я", part_of_speech="pronoun"),
                SemanticTerm(object_lemma, object_surface, number=number),
                truth,
            ),
        ),
        confidence=0.9,
    )


def test_same_construction_realizes_arbitrary_lexemes_with_morphology() -> None:
    nails = realize_content_plan(_possession("ноготь", "ногти", truth=False, number="plural"))
    keys = realize_content_plan(_possession("ключ", "ключи", truth=False, number="plural"))
    memory = realize_content_plan(_possession("память", "память", truth=True))

    assert nails[0].text == "Нет, у меня нет ногтей."
    assert keys[0].text == "Нет, у меня нет ключей."
    assert memory[0].text == "Да, у меня есть память."


def test_negated_question_separates_answer_polarity_from_world_state() -> None:
    plan = ResponseContentPlan(
        response_act="relation_answer",
        answer_polarity=False,
        clauses=(
            SemanticClause(
                "possession",
                SemanticTerm("я", "я", part_of_speech="pronoun"),
                SemanticTerm("память", "память"),
                True,
            ),
        ),
        confidence=0.9,
    )

    assert realize_content_plan(plan)[0].text == "Нет, у меня есть память."


def test_validated_learned_schema_adds_a_variant_without_authored_words() -> None:
    learned = ConstructionSpec.from_mapping(
        {
            "key": "object_first_possession",
            "response_act": "relation_answer",
            "relation": "possession",
            "truth_state": "known",
            "sequence": [
                "polar_marker",
                "comma",
                "object:truth_case",
                "existence_predicate",
                "possession_preposition",
                "subject:genitive",
            ],
        }
    )

    assert learned is not None
    surfaces = realize_content_plan(
        _possession("носок", "носки", truth=False, number="plural"),
        (learned,),
    )

    assert any(item.text == "Нет, носков нет у меня." for item in surfaces)
    assert any(
        item.construction == "semantic_consultant:object_first_possession"
        for item in surfaces
    )


def test_schema_rejects_raw_words_and_incomplete_role_layout() -> None:
    assert ConstructionSpec.from_mapping(
        {
            "key": "hidden_answer",
            "response_act": "relation_answer",
            "relation": "possession",
            "truth_state": "known",
            "sequence": [
                "polar_marker",
                "готовая реплика пользователю",
                "subject:genitive",
                "object:genitive",
            ],
        }
    ) is None
