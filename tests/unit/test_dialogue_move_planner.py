from dataclasses import fields

from debilgi.language.dialogue_acts import analyze_dialogue_act
from debilgi.language.dialogue_move_planner import (
    DialogueMovePlan,
    plan_dialogue_moves,
)
from debilgi.language.productive_nlg import realize_thought
from debilgi.language.self_state_reasoning import analyze_self_state
from debilgi.language.structured_utterance import analyze_structured_utterance


def _plans(prompt: str) -> tuple[DialogueMovePlan, ...]:
    structure = analyze_structured_utterance(prompt)
    self_state = analyze_self_state(structure)
    frame = analyze_dialogue_act(structure, self_state=self_state)
    return plan_dialogue_moves(
        frame,
        self_state=self_state,
        structure=structure,
        prompt=prompt,
    )


def test_expressive_fragment_produces_competing_non_lexical_moves() -> None:
    plans = _plans("ебануться")

    assert len(plans) >= 2
    assert {plan.strategy for plan in plans} >= {
        "ask_what_happened",
        "ask_what_triggered_reaction",
    }
    assert "text" not in {field.name for field in fields(DialogueMovePlan)}
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }
    assert "Что ты подразумеваешь под ебануться?" not in surfaces
    assert all(surface.endswith("?") for surface in surfaces)
    assert {plan.semantic_origin for plan in plans} == {"scaffold"}


def test_self_label_stance_changes_with_dialogue_markers() -> None:
    confirmation = _plans("ты же круто, да?")
    challenge = _plans("ты чё герой что-ли?")

    assert len(confirmation) >= 2
    assert len(challenge) >= 2
    assert {plan.stance for plan in confirmation} == {"support"}
    assert {plan.stance for plan in challenge} == {"oppose"}
    assert {plan.strategy for plan in confirmation} >= {
        "contextual_self_assessment",
        "evaluation_stance",
    }


def test_known_nominal_relation_is_digested_instead_of_defined() -> None:
    plans = _plans("островок на башке")

    assert {plan.strategy for plan in plans} == {
        "integrate_nominal_relation",
        "verify_nominal_relation",
    }
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }
    assert "Я представляю островок на башке." in surfaces
    assert all("подразумева" not in surface.casefold() for surface in surfaces)
    assert {plan.semantic_origin for plan in plans} == {"derived"}


def test_complete_statement_continues_its_content_instead_of_dialogue_meta_talk() -> None:
    plans = _plans("завтра будет четверг")
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }

    assert {plan.strategy for plan in plans} == {
        "ask_owned_topic_view",
        "ask_temporal_intention",
    }
    assert "Что ты планируешь на четверг?" in surfaces
    assert all("продолжаю мысль" not in surface.casefold() for surface in surfaces)
    assert all("держу тему" not in surface.casefold() for surface in surfaces)
    assert {plan.semantic_origin for plan in plans} == {"scaffold"}


def test_non_temporal_statement_uses_its_salient_concept() -> None:
    plans = _plans("кот сидит на шкафу")
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }

    assert {plan.strategy for plan in plans} == {"ask_owned_topic_view"}
    assert any("шкаф" in surface.casefold() for surface in surfaces)
    assert all("тема" not in surface.casefold() for surface in surfaces)


def test_evaluative_complement_does_not_replace_its_entity_as_topic() -> None:
    plans = _plans("Марс хорошая планета")
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }

    assert {plan.key_points for plan in plans} == {("марс",)}
    assert {surface.casefold() for surface in surfaces} == {
        "что ты думаешь о марсе?"
    }


def test_underspecified_self_reflection_question_describes_active_process() -> None:
    plans = _plans("думаешь типо?")
    surfaces = {
        surface.text
        for plan in plans
        for surface in realize_thought(plan.thought)
    }

    assert {plan.strategy for plan in plans} == {
        "ground_self_reflection_in_active_process"
    }
    assert surfaces == {"Я анализирую контекст."}
    assert {plan.semantic_origin for plan in plans} == {"derived"}


def test_attention_call_is_a_grounded_local_acknowledgement() -> None:
    plans = _plans("ауу")

    assert {plan.strategy for plan in plans} == {"active_listening", "hold_attention"}
    assert {plan.semantic_origin for plan in plans} == {"derived"}
