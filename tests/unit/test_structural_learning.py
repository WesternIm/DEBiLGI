from debilgi.language.structural_learning import (
    learned_structural_bonus,
    structural_response_features,
)


def test_structural_features_transfer_beyond_exact_words() -> None:
    first = structural_response_features(
        prompt="Тебе нравятся эмодзи?",
        candidate_text="Да, мне нравятся эмодзи.",
        source="semantic_thought",
        speech_act="question",
        metadata={"response_obligation": "express_position"},
    )
    second = structural_response_features(
        prompt="Тебе нравится музыка?",
        candidate_text="Да, мне нравится музыка.",
        source="semantic_thought",
        speech_act="question",
        metadata={"response_obligation": "express_position"},
    )

    shared = set(first) & set(second)
    assert "prompt:clause:question" in shared
    assert "candidate:clause:statement" in shared
    assert "cross:obligation_source:express_position>semantic_thought" in shared
    assert not any("эмодзи" in key or "музыка" in key for key in first | second)


def test_learned_structural_bonus_is_bounded() -> None:
    features = {"candidate:clause:statement": 1.0, "source:semantic_thought": 1.0}
    weights = {key: 0.75 for key in features}

    assert 0.0 < learned_structural_bonus(features, weights) <= 0.12
