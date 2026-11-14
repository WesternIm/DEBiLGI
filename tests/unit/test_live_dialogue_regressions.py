import pytest

from debilgi.language.content import content_lemmas
from debilgi.language.dialogue_acts import analyze_dialogue_act
from debilgi.language.dialogue_move_planner import plan_dialogue_moves
from debilgi.language.morphology import RussianMorphology
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.recovery import is_clarification_request
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.language.tokenizer import tokenize


@pytest.mark.parametrize("prompt", ["Всмысле?", "В смысле?", "а всмысле??"])
def test_clarification_is_not_spelling_corrected_to_fiction(prompt):
    assert is_clarification_request(prompt)
    assert "вымысел" not in content_lemmas(prompt)
    assert "всмысле" not in tokenize(prompt)


def plans(prompt, context=()):
    structure = analyze_structured_utterance(prompt)
    return plan_dialogue_moves(analyze_dialogue_act(structure), structure=structure,
                              prompt=prompt, context_messages=context)


def test_definition_reply_binds_previous_word_and_preserves_purpose():
    result = plans("Это бумага такая для гигиены", ("Что ты подразумеваешь под «салфетка»?",))
    assert len(result) == 1
    assert result[0].strategy == "resolve_dialogue_obligation"
    text = realize_thought(result[0].thought, limit=1)[0].text
    assert "Салфетка" in text and "бумагой" in text and "для гигиены" in text
    assert text.endswith("?")


@pytest.mark.parametrize("prompt,context", [
    ("Это не бумага", ("Что ты подразумеваешь под «салфетка»?",)),
    ("Это бумага или ткань", ("Что ты подразумеваешь под «салфетка»?",)),
    ("Это бумага", ("Что ты подразумеваешь под «салфетка»?", "Поговорим о другом")),
    ("Это бумага", ()),
])
def test_definition_binding_does_not_invent_missing_or_qualified_reference(prompt, context):
    assert all(p.strategy != "resolve_dialogue_obligation" for p in plans(prompt, context))


def test_pronoun_is_not_a_word_meaning_gap():
    assert "этот" not in content_lemmas("Это бумага такая для гигиены")
    result = plans("Это бумага такая для гигиены")
    assert not any(p.strategy == "ask_semantic_scope" and "этот" in p.key_points for p in result)


def test_repeated_sentence_does_not_rescore_all_tokens(monkeypatch):
    morphology = RussianMorphology()
    first = morphology.analyze_sentence("Это бумага такая для гигиены")

    def fail(*args, **kwargs):
        raise AssertionError("identical sentence was recomputed")

    monkeypatch.setattr(morphology, "_contextual_readings", fail)
    assert morphology.analyze_sentence(first.text) is first
