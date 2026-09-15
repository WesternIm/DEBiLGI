from debilgi.language.productive_nlg import (
    ClauseThought,
    ConceptArgument,
    ConceptLexeme,
    DependencyPattern,
    SemanticDependency,
    observe_dependency_patterns,
    realize_thought,
    unresolved_answer_thought,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def _thought(object_lemma: str, object_surface: str) -> ClauseThought:
    return ClauseThought(
        predicate=ConceptLexeme(
            "понять",
            "понял",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(ConceptLexeme(object_lemma, object_surface, "noun")),
            ),
        ),
        response_polarity=True,
        confidence=0.92,
    )


def test_one_semantic_graph_productively_realizes_different_lexemes() -> None:
    idea = realize_thought(_thought("идея", "идею"))[0]
    principle = realize_thought(_thought("принцип", "принцип"))[0]

    assert idea.text == "Да, я понял идею."
    assert principle.text == "Да, я понял принцип."
    assert idea.role_order == ("subject", "predicate", "direct_object")


def test_plural_complement_does_not_inherit_dictionary_lemma_gender() -> None:
    thought = ClauseThought(
        predicate=ConceptLexeme("считать", "считать", "verb"),
        dependencies=(
            SemanticDependency("subject", ConceptArgument(ConceptLexeme("я", "я", "pronoun"))),
            SemanticDependency("direct_object", ConceptArgument(
                ConceptLexeme("результат", "результаты", "noun", (("number", "plural"),)),
            )),
            SemanticDependency("predicate_complement", ConceptArgument(
                ConceptLexeme("хороший", "хороший", "adjective", (("number", "plural"),)),
            )),
        ),
        confidence=0.9,
    )
    assert realize_thought(thought)[0].text == "Я считаю результаты хорошими."


def test_observed_dependency_order_contains_no_authored_words() -> None:
    observed = DependencyPattern(
        ("direct_object", "subject", "predicate"),
        confidence=0.95,
        evidence_count=8,
    )

    surfaces = realize_thought(_thought("идея", "идею"), (observed,))

    assert surfaces[0].text == "Да, идею я понял."
    assert all(role in {"subject", "predicate", "direct_object"} for role in observed.roles)


def test_invalid_lexicalized_pattern_is_rejected() -> None:
    assert not DependencyPattern(("готовый ответ", "predicate")).valid


def test_local_sentence_is_learned_as_roles_without_its_words() -> None:
    patterns = observe_dependency_patterns(
        analyze_structured_utterance("Ты понял сложную идею?")
    )

    assert patterns
    assert patterns[0].roles == ("subject", "predicate", "direct_object")
    assert all(
        word not in patterns[0].roles
        for word in ("ты", "понял", "сложную", "идею")
    )


def test_observation_keeps_contextual_constraints_without_sentence_text() -> None:
    pattern = observe_dependency_patterns(
        analyze_structured_utterance("Я помню наш разговор.")
    )[0]

    assert pattern.construction_id is not None
    assert pattern.clause_kind == "statement"
    assert pattern.predicate_class == "lexical"
    assert pattern.required_roles == ("subject", "predicate")
    assert pattern.optional_roles == ("direct_object",)
    assert pattern.negation_mode == "forbidden"
    assert all(
        word not in pattern.roles
        for word in ("я", "помню", "наш", "разговор")
    )


def test_unresolved_answer_is_also_composed_from_semantic_roles() -> None:
    surfaces = realize_thought(unresolved_answer_thought())

    assert surfaces[0].text == "Я не знаю ответа на этот вопрос."
    assert surfaces[0].role_order == (
        "subject",
        "negation",
        "predicate",
        "direct_object",
        "topic",
    )


def test_past_tense_does_not_receive_incompatible_person_grammeme() -> None:
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "есть",
            "есть",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(
                    ConceptLexeme(
                        "я",
                        "я",
                        "pronoun",
                        (("number", "singular"), ("person", "first")),
                    )
                ),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("яблоко", "яблоко", "noun"),
                    grammatical_case="accusative",
                ),
            ),
        ),
        negated=True,
        response_polarity=False,
        confidence=0.95,
    )

    assert realize_thought(thought)[0].text == "Нет, я не ел яблоко."


def test_reflexive_predicate_cannot_gain_an_accusative_object_by_guessing() -> None:
    malformed = ClauseThought(
        predicate=ConceptLexeme(
            "поперхнуться",
            "поперхнулся",
            "verb",
            (("tense", "past"), ("number", "singular"), ("gender", "masculine")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(ConceptLexeme("я", "я", "pronoun")),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(ConceptLexeme("мозг", "мозг", "noun")),
            ),
        ),
        confidence=0.9,
    )

    assert realize_thought(malformed) == ()


def test_positive_learned_order_cannot_detach_negation_in_a_negative_thought() -> None:
    unsafe_for_negation = DependencyPattern(
        ("direct_object", "predicate", "subject"),
        confidence=0.99,
        evidence_count=100,
    )
    thought = _thought("зеркало", "зеркало")
    negative = ClauseThought(
        predicate=thought.predicate,
        dependencies=thought.dependencies,
        negated=True,
        response_polarity=False,
        confidence=thought.confidence,
    )

    surfaces = realize_thought(negative, (unsafe_for_negation,))

    assert surfaces[0].text == "Нет, я не понял зеркало."
    assert all(
        surface.role_order.index("negation") + 1
        == surface.role_order.index("predicate")
        for surface in surfaces
    )


def test_explicit_wh_focus_cannot_be_demoted_by_a_learned_order() -> None:
    thought = ClauseThought(
        predicate=ConceptLexeme(
            "думать",
            "думаешь",
            "verb",
            (("tense", "present"), ("number", "singular"), ("person", "second")),
        ),
        dependencies=(
            SemanticDependency(
                "subject",
                ConceptArgument(ConceptLexeme("ты", "ты", "pronoun")),
            ),
            SemanticDependency(
                "direct_object",
                ConceptArgument(
                    ConceptLexeme("что", "что", "pronoun"),
                    grammatical_case="accusative",
                ),
            ),
            SemanticDependency(
                "topic",
                ConceptArgument(
                    ConceptLexeme("марс", "Марс", "noun"),
                    preposition="о",
                    grammatical_case="prepositional",
                ),
            ),
        ),
        clause_kind="question",
        focus_role="direct_object",
        confidence=0.9,
    )
    misplaced = DependencyPattern(
        ("predicate", "subject", "direct_object", "topic"),
        clause_kind="question",
        required_roles=("subject", "predicate", "direct_object", "topic"),
        focus_role="direct_object",
        confidence=0.99,
        evidence_count=100,
    )

    surfaces = realize_thought(thought, (misplaced,))

    assert len(surfaces) == 1
    assert surfaces[0].text.casefold() == "что ты думаешь о марсе?"
