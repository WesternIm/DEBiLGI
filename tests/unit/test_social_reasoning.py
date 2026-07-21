from debilgi.language.dialogue_acts import DialogueActFrame
from debilgi.language.social_reasoning import (
    SOCIAL_POLICY_CURRICULUM,
    candidate_social_features,
    choose_communicative_intention,
)


def _frame(kind: str, obligation: str, *, register: str = "neutral") -> DialogueActFrame:
    return DialogueActFrame(
        kind=kind,
        response_obligation=obligation,
        subject="self",
        predicate="быть",
        objects=("носки",),
        polarity=0.0,
        register=register,
        complete=True,
        confidence=0.92,
        evidence=("test",),
    )


def test_personal_question_selects_answer_before_explanation() -> None:
    intention = choose_communicative_intention(
        _frame("information_question", "answer_question"),
        SOCIAL_POLICY_CURRICULUM,
        grounding_available=True,
    )

    assert intention.primary_move == "answer"
    assert intention.answer_first
    assert "define_isolated_word" in intention.forbidden_moves
    assert intention.epistemic.status == "grounded"


def test_desire_to_agree_never_turns_unknown_into_truth() -> None:
    intention = choose_communicative_intention(
        _frame("relation_question", "resolve_relation"),
        SOCIAL_POLICY_CURRICULUM,
        grounding_available=False,
    )

    assert intention.epistemic.status == "unknown"
    assert intention.truth_priority == 1.0

    unsupported = candidate_social_features(
        intention=intention,
        source="graph",
        text="Да, это точно так.",
        metadata={"response_move": "answer"},
        required_facts=(),
        source_confidence=0.9,
    )
    calibrated = candidate_social_features(
        intention=intention,
        source="semantic_thought",
        text="Я не знаю.",
        metadata={"response_move": "qualify"},
        required_facts=(),
        source_confidence=0.5,
    )

    assert calibrated["truth_grounding"] > unsupported["truth_grounding"]
    assert calibrated["stance_consistency"] > unsupported["stance_consistency"]


def test_social_reference_scores_moves_without_reference_wording() -> None:
    intention = choose_communicative_intention(
        _frame("positive_evaluation", "acknowledge_evaluation", register="coarse"),
        SOCIAL_POLICY_CURRICULUM,
        grounding_available=False,
    )
    features = candidate_social_features(
        intention=intention,
        source="dialogue_act",
        text="Да, я стараюсь.",
        metadata={"response_move": "acknowledge"},
        required_facts=(),
        source_confidence=0.9,
    )

    assert intention.epistemic.status == "not_applicable"
    assert features["semantic_frame_fit"] == 1.0
    assert features["reference_copy_penalty"] == 0.0
    assert all(not hasattr(policy, "text") for policy in SOCIAL_POLICY_CURRICULUM)
