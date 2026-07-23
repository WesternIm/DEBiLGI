from debilgi.language.valency import (
    assess_predicate_valency,
    governed_object_case,
    predicate_valency,
    resolve_complement_role,
)


def test_lexical_profile_requires_core_object_in_isolated_fact() -> None:
    assessment = assess_predicate_valency(
        "взять",
        present_roles=frozenset({"subject"}),
    )

    assert not assessment.complete
    assert assessment.missing_roles == ("direct_object",)
    assert assessment.profile_source == "lexical_profile"


def test_generic_profile_does_not_invent_an_object_requirement() -> None:
    assessment = assess_predicate_valency(
        "улыбнуться",
        present_roles=frozenset({"subject"}),
    )

    assert assessment.complete
    assert assessment.profile_source == "generic"


def test_predicate_government_resolves_ambiguous_complement_role() -> None:
    assert resolve_complement_role("взять", "со", "source_or_companion") == "source"
    assert resolve_complement_role("говорить", "с", "source_or_companion") == "companion"
    assert resolve_complement_role("жить", "в", "location_or_destination") == "location"
    assert resolve_complement_role("положить", "на", "location_or_destination") == "destination"


def test_valency_profile_exposes_cases_without_reply_text() -> None:
    profile = predicate_valency("дать")

    assert profile is not None
    slots = {item.role: item.grammatical_case for item in profile.slots}
    assert slots == {
        "subject": "nominative",
        "direct_object": "accusative",
        "indirect_object": "dative",
    }


def test_lexical_predicate_can_govern_non_accusative_object() -> None:
    assert governed_object_case("бояться", "direct_object", "accusative") == "genitive"
    assert governed_object_case("управлять", "direct_object", "accusative") == "instrumental"
    assert governed_object_case("пользоваться", "direct_object", "accusative") == (
        "instrumental"
    )
