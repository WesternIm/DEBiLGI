import pytest

from debilgi.knowledge.service import _title_matches_query
from debilgi.language.definition_relations import definition_answer_thought, definition_relations
from debilgi.language.grounded_repair import grounded_repair_thought
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.structured_utterance import analyze_structured_utterance


@pytest.mark.parametrize(
    "subject,definition,expected",
    [
        (
            "ручка",
            "Ру́чка — письменный инструмент для письма.",
            {("is_a", "письменный инструмент"), ("used_for", "письмо")},
        ),
        ("крыло", "Крыло — часть самолёта.", {("part_of", "самолёт")}),
        ("бронза", "Бронза состоит из металлов.", {("consists_of", "металл")}),
        (
            "ручка",
            "инструмент, предназначенный для письма",
            {("is_a", "инструмент"), ("used_for", "письмо")},
        ),
    ],
)
def test_explicit_relation_keeps_its_definition_evidence(subject, definition, expected):
    edges = definition_relations(subject, definition)
    assert {(edge.predicate, edge.target) for edge in edges} == expected
    assert all(edge.evidence == definition and edge.sense_key for edge in edges)


@pytest.mark.parametrize(
    "definition",
    [
        "Ручка — не инструмент.",
        "Ручка может быть инструментом.",
        "Ручка — инструмент или часть двери.",
        "Ручка — инструмент без чернил.",
        "Ручка, Надежда Александровна — певица.",
        "Стол — предмет мебели. Ручка — инструмент.",
    ],
)
def test_qualified_or_unrelated_prose_is_not_promoted_to_a_fact(definition):
    assert not definition_relations("ручка", definition, require_subject=True)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Что такое ручка?", "Ручка является письменным инструментом."),
        ("Для чего ручка?", "Ручка используется для письма."),
    ],
)
def test_selected_relation_is_realized_instead_of_copying_source(prompt, expected):
    definition = "Ручка — письменный инструмент для письма."
    edges = tuple((e.predicate, e.target) for e in definition_relations("ручка", definition))
    thought = definition_answer_thought(prompt, (("ручка", edges),))
    assert thought is not None
    text = realize_thought(thought, limit=1)[0].text
    assert text == expected and text != definition
    assert definition_answer_thought("У тебя есть ручка?", (("ручка", edges),)) is None


def test_no_arbitrary_choice_between_competing_definitions():
    assert (
        definition_answer_thought(
            "Что такое ручка?", (("ручка", (("is_a", "инструмент"), ("is_a", "деталь"))),)
        )
        is None
    )


@pytest.mark.parametrize(
    "word,title",
    [
        ("ручка", "Ручка, Надежда Александровна"),
        ("считать", "Считать погибшей"),
        ("ножницы", "Ножницы (Windows)"),
    ],
)
def test_partial_titles_cannot_define_a_common_word(word, title):
    assert not _title_matches_query(word, title)
    assert _title_matches_query(word, word.capitalize())
    assert _title_matches_query("гта 5", "Grand Theft Auto V")


@pytest.mark.parametrize("prompt,verb", [("Ищи", "искать"), ("Открой", "открыть")])
def test_bare_command_requests_missing_object_not_verb_meaning(prompt, verb):
    thought = grounded_repair_thought(analyze_structured_utterance(prompt))
    assert thought is not None
    assert any(dep.argument.head.lemma == verb for dep in thought.dependencies)
    assert "подразумеваешь" not in realize_thought(thought, limit=1)[0].text
