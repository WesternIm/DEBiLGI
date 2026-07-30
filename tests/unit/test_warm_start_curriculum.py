from debilgi.language.warm_start_curriculum import WARM_START_PAIRS, WARM_START_SKILLS


def test_warm_start_contains_no_authored_prompt_reply_pairs() -> None:
    assert WARM_START_PAIRS == ()


def test_warm_start_keeps_only_structural_skill_labels() -> None:
    assert len(WARM_START_SKILLS) >= 10
    assert {
        "dialogue_act",
        "lexical_lookup",
        "contextual_interpretation",
        "explanation_learning",
        "source_conflict_tracking",
    } <= set(WARM_START_SKILLS)
    assert all(" " not in skill and skill == skill.casefold() for skill in WARM_START_SKILLS)
