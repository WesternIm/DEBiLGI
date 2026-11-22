from debilgi.language.language_layer_curriculum import SCENES, total_turns


def test_language_layer_curriculum_is_linked_and_covers_new_capabilities() -> None:
    skills = {skill for scene in SCENES for skill in scene.skills}

    assert len(SCENES) >= 11
    assert total_turns() >= 66
    assert {
        "reference_dictionary",
        "personal_interpretation",
        "unknown_form",
        "polysemy",
        "curiosity_queue",
        "response_plan",
        "causal_summary",
        "idle_timeout",
        "neologism_reason",
        "human_feedback",
        "expression_gap",
    } <= skills


def test_each_scene_is_a_real_alternating_dialogue() -> None:
    for scene in SCENES:
        assert len(scene.turns) >= 4
        assert scene.turns[0].speaker == "human"
        assert all(turn.text.strip() for turn in scene.turns)
        assert all(
            left.speaker != right.speaker
            for left, right in zip(scene.turns, scene.turns[1:], strict=False)
        )
