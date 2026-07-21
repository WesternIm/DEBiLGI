from debilgi.language.grammar_semantics_curriculum import SCENES, covered_skills, total_turns


def test_curriculum_covers_grammar_and_semantics() -> None:
    skills = set(covered_skills())

    assert len(SCENES) >= 16
    assert total_turns() >= 64
    assert {
        "nominative",
        "genitive",
        "dative",
        "accusative",
        "instrumental",
        "prepositional",
        "agreement",
        "synonyms",
        "homonyms",
        "reference_dictionary",
        "context_resolution",
        "particles",
        "interjections",
        "pragmatics",
        "online_reference",
    } <= skills
