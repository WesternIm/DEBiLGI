from debilgi.language.causal_reasoning import (
    CausalCue,
    analyze_causal_query,
    extract_causal_observations,
    infer_compositional_consequence,
)

CUES = (
    CausalCue("потому что", "marker", "causes", "effect_before_cause", 0.94),
    CausalCue("поэтому", "marker", "causes", "cause_before_effect", 0.94),
    CausalCue("если", "marker", "enables", "condition_then_effect", 0.86),
    CausalCue("потом", "marker", "precedes", "cause_before_effect", 0.74),
    CausalCue("вызывать", "predicate", "causes", "subject_then_object", 0.90),
)


def test_because_orients_reason_toward_effect() -> None:
    observations = extract_causal_observations(
        "Свет погас, потому что лампа перегорела.",
        CUES,
    )

    assert len(observations) == 1
    assert observations[0].cause.terms == ("лампа", "перегореть")
    assert observations[0].effect.terms == ("свет", "погаснуть")
    assert observations[0].relation == "causes"


def test_therefore_and_condition_preserve_different_modal_strength() -> None:
    consequence = extract_causal_observations(
        "Лампа перегорела, поэтому свет погас.", CUES
    )[0]
    condition = extract_causal_observations(
        "Если лампа перегорит, то свет погаснет.", CUES
    )[0]

    assert consequence.relation == "causes"
    assert condition.relation == "enables"
    assert consequence.cause.predicate_lemma == condition.cause.predicate_lemma


def test_leading_because_marker_keeps_cause_before_effect() -> None:
    cues = (
        *CUES,
        CausalCue("поскольку", "marker", "causes", "effect_before_cause", 0.90),
    )
    observation = extract_causal_observations(
        "Поскольку прошёл дождь, дорога намокла.",
        cues,
    )[0]

    assert observation.cause.terms == ("пройти", "дождь")
    assert observation.effect.terms == ("дорога", "намокнуть")


def test_causal_predicate_builds_nominal_cause_and_effect() -> None:
    observation = extract_causal_observations("Дождь вызывает наводнение.", CUES)[0]

    assert observation.cause.terms == ("дождь",)
    assert observation.effect.terms == ("наводнение",)
    assert observation.supports_link


def test_temporal_order_is_not_mislabeled_as_causation() -> None:
    observation = extract_causal_observations(
        "Лампа мигнула, потом свет погас.", CUES
    )[0]

    assert observation.relation == "precedes"


def test_questions_choose_the_requested_side_of_the_link() -> None:
    cause = analyze_causal_query("Почему свет погас?")
    effect = analyze_causal_query("Что будет если лампа перегорит?")

    assert cause is not None and cause.direction == "cause"
    assert set(cause.focus_terms) == {"свет", "погаснуть"}
    assert effect is not None and effect.direction == "effect"
    assert set(effect.focus_terms) == {"лампа", "перегореть"}


def test_short_cause_follow_up_restores_effect_from_dialogue() -> None:
    query = analyze_causal_query(
        "Почему?",
        ("Лампа перегорела.", "Свет погас."),
    )

    assert query is not None
    assert query.direction == "cause"
    assert set(query.focus_terms) == {"свет", "погаснуть"}


def test_effect_query_keeps_action_object_and_source_roles() -> None:
    query = analyze_causal_query("Что будет, если толкнуть книгу со стола?")

    assert query is not None
    assert query.direction == "effect"
    assert query.predicate_lemma == "толкнуть"
    assert query.object_lemma == "книга"
    assert query.source_lemma == "стол"


def test_physical_consequence_is_composed_from_roles_not_a_stored_phrase() -> None:
    query = analyze_causal_query("Что будет, если толкнуть книгу со стола?")

    assert query is not None
    inference = infer_compositional_consequence(query)

    assert inference is not None
    assert inference.text == "Книга, скорее всего, упадёт."
    assert inference.evidence == ("schema:force_displacement_removes_support",)
    assert inference.path == ()


def test_topical_overlap_without_a_causal_schema_does_not_invent_an_effect() -> None:
    query = analyze_causal_query("Что будет, если прочитать книгу со стола?")

    assert query is not None
    assert infer_compositional_consequence(query) is None
