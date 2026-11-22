import pytest

from debilgi.language.grounded_repair import grounded_repair_thought
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.self_state_reasoning import analyze_self_state
from debilgi.language.structured_utterance import analyze_structured_utterance


@pytest.mark.parametrize("prompt", ["лошадка", "пиздээ", "не знаю"])
def test_no_lexical_gap_does_not_invent_a_replacement_thought(prompt):
    thought = grounded_repair_thought(
        analyze_structured_utterance(prompt), clarifiable_terms=(),
    )
    assert thought is None


def test_missing_fact_still_allows_honest_uncertainty():
    thought = grounded_repair_thought(
        analyze_structured_utterance("Сколько стоит гречка?"), clarifiable_terms=(),
    )
    assert thought is not None
    assert "answer_not_derived" in thought.evidence


@pytest.mark.parametrize(
    "prompt,subject",
    [
        ("ну всё, время настало", "время"),
        ("ночь наступила", "ночь"),
        ("письмо пришло", "письмо"),
    ],
)
def test_question_is_composed_from_current_subject_and_predicate(prompt, subject):
    structure = analyze_structured_utterance(prompt)
    thought = grounded_repair_thought(structure)
    assert thought is not None
    assert thought.clause_kind == "question"
    assert thought.focus_role == "subject"
    clause = structure.clauses[0]
    assert thought.predicate.lemma == structure.tokens[clause.predicate_index].lemma
    assert thought.dependencies[0].argument.head.lemma == subject
    surfaces = realize_thought(thought, limit=1)
    assert len(surfaces) == 1
    assert subject in surfaces[0].text.casefold()
    assert surfaces[0].text.endswith("?")


@pytest.mark.parametrize("term", ["порог", "кваргония", "телескоп"])
def test_unknown_reference_remains_bound_to_input(term):
    thought = grounded_repair_thought(analyze_structured_utterance(term))
    assert thought is not None
    text = realize_thought(thought, limit=1)[0].text
    assert term in text.casefold()
    assert text.endswith("?")
    assert "сформулировать" not in text.casefold()
    assert "завышенн" not in text.casefold()


def test_contact_is_a_response_to_contact_not_generation_failure():
    thought = grounded_repair_thought(analyze_structured_utterance("але"))
    assert thought is not None
    assert thought.evidence == ("current_message:contact_request",)
    text = realize_thought(thought, limit=1)[0].text
    assert "не могу" not in text.casefold()
    assert not text.endswith("?")


@pytest.mark.parametrize("prompt", ["почему?", "ну?"])
def test_missing_question_target_requests_information_not_word_definition(prompt):
    thought = grounded_repair_thought(analyze_structured_utterance(prompt))
    assert thought is not None
    text = realize_thought(thought, limit=1)[0].text
    assert text.endswith("?")
    assert "подразумеваешь" not in text
    assert "current_message:missing_explanation_target" in thought.evidence


def test_temporal_connector_does_not_turn_negative_statement_into_question():
    structure = analyze_structured_utterance("ну это пока что ты не считаешь себя тупым")
    assert structure.clauses[0].clause_kind == "statement"
    assert structure.clauses[0].negated




@pytest.mark.parametrize('prompt,gender', [('Ты тупой?', 'masculine'),
                                         ('Ты тупая?', 'feminine')])
def test_nominal_predicate_gender_uses_clause_role(prompt, gender):
    frame = analyze_self_state(analyze_structured_utterance(prompt))
    assert frame is not None and frame.target is not None
    assert dict(frame.target.head.features)['gender'] == gender
