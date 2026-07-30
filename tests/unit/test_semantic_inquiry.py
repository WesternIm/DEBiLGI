from debilgi.language.semantic_inquiry import (
    analyze_semantic_inquiry,
    inquiry_trace,
    realize_semantic_inquiry,
)
from debilgi.language.structured_utterance import analyze_structured_utterance


def test_scene_is_decomposed_before_inference_and_hypothesis() -> None:
    inquiry = analyze_semantic_inquiry(
        analyze_structured_utterance("Волк бежит за зайцем"),
        exploration=0.7,
    )

    steps = {step.question_kind: step for step in inquiry.steps}
    assert steps["actor"].answer_lemma == "волк"
    assert steps["action"].answer_lemma == "бежать"
    assert steps["relation"].answer_role == "pursuit_target"
    assert steps["relation"].answer_lemma == "заяц"
    assert steps["relation"].status == "inferred"
    assert steps["participant_action"].answer_lemma == "убегать"
    assert steps["participant_action"].status == "hypothesis"
    assert steps["purpose"].status == "unresolved"
    assert steps["cause"].status == "unresolved"
    assert inquiry.study_terms == ("волк", "заяц", "преследование")


def test_scene_response_is_composed_from_calibrated_conclusions() -> None:
    inquiry = analyze_semantic_inquiry(
        analyze_structured_utterance("Волк бежит за зайцем"),
        exploration=0.7,
    )

    surfaces = realize_semantic_inquiry(inquiry)

    assert surfaces[0].text == ("Похоже, волк преследует зайца. Вероятно, заяц убегает от волка.")
    assert surfaces[0].source == "semantic_inquiry_composition"


def test_inanimate_target_does_not_trigger_an_escape_hypothesis() -> None:
    inquiry = analyze_semantic_inquiry(
        analyze_structured_utterance("Волк бежит за машиной"),
        exploration=0.9,
    )

    assert not inquiry.renderable
    assert realize_semantic_inquiry(inquiry) == ()


def test_relevant_knowledge_strengthens_but_does_not_turn_hypothesis_into_fact() -> None:
    unsupported = analyze_semantic_inquiry(
        analyze_structured_utterance("Волк бежит за зайцем"),
        exploration=0.7,
    )
    supported = analyze_semantic_inquiry(
        analyze_structured_utterance("Волк бежит за зайцем"),
        knowledge_facts=(
            "Волк — хищник.",
            "Заяц может быть добычей и убегать от хищника.",
        ),
        exploration=0.7,
    )

    assert supported.confidence > unsupported.confidence
    assert supported.conclusions[-1].epistemic_status == "hypothesis"


def test_trace_exposes_slots_sources_and_unresolved_gaps() -> None:
    inquiry = analyze_semantic_inquiry(analyze_structured_utterance("Волк бежит за зайцем"))

    trace = inquiry_trace(inquiry)

    assert trace["renderable"] is True
    assert trace["study_terms"] == ("волк", "заяц", "преследование")
    assert any(
        step["answer_role"] == "goal" and step["status"] == "unresolved" for step in trace["steps"]
    )
