from debilgi.language.dialogue_curriculum import SCENES, total_turns


def test_curriculum_is_varied_and_multi_turn() -> None:
    required = {
        "agreement",
        "disagreement",
        "refusal",
        "approval",
        "curiosity",
        "clarification",
        "repair",
        "addressee",
        "topic_change",
        "mixed_reaction",
    }
    skills = {skill for scene in SCENES for skill in scene.skills}

    assert len(SCENES) >= 16
    assert total_turns() >= 90
    assert required <= skills
    assert all(len(scene.turns) >= 4 for scene in SCENES)
    assert all(scene.turns[0].speaker != "bot" for scene in SCENES)


def test_curriculum_has_unique_ids_and_natural_turn_taking() -> None:
    assert len({scene.scene_id for scene in SCENES}) == len(SCENES)
    assert len({turn.text for scene in SCENES for turn in scene.turns}) == total_turns()
    assert any(
        {turn.speaker for turn in scene.turns} == {"bot", "human_a", "human_b"} for scene in SCENES
    )
    assert all(
        any(turn.speaker == "bot" for turn in scene.turns)
        and any(turn.speaker != "bot" for turn in scene.turns)
        for scene in SCENES
    )
