from debilgi.language.abstract_reasoning import (
    build_mental_model,
    infer_relations,
    resolve_abstract_question,
)


def test_hypothetical_space_is_answered_inside_the_model_not_as_a_fact() -> None:
    model = build_mental_model(
        "Представь: шар внутри куба. Где шар?",
        source_event_id="event:space",
    )
    resolution = resolve_abstract_question(model)

    assert model is not None
    assert model.mode == "hypothetical"
    assert model.imaginative is True
    assert resolution is not None
    assert resolution.text == (
        "В этой воображаемой ситуации шар находится внутри куба."
    )
    assert "mental_mode:hypothetical" in resolution.evidence


def test_spatial_transitivity_supports_reasoning_beyond_a_copied_premise() -> None:
    model = build_mental_model(
        "Допустим, башня слева от реки, а дом слева от башни. Где дом?"
    )
    resolution = resolve_abstract_question(model)

    assert model is not None
    assert any(
        relation.subject == "дом"
        and relation.relation == "left_of"
        and relation.object == "река"
        and relation.inferred
        for relation in model.relations
    )
    assert resolution is not None
    assert resolution.text == (
        "В этой воображаемой ситуации дом находится слева от башни."
    )


def test_context_can_supply_the_premise_for_a_follow_up_question() -> None:
    model = build_mental_model(
        "А где сфера?",
        context_messages=("Представь, что сфера находится над башней.",),
    )

    assert model is not None
    assert model.mode == "hypothetical"
    assert resolve_abstract_question(model).text == (
        "В этой воображаемой ситуации сфера находится над башней."
    )


def test_analogy_is_kept_as_a_mapping_instead_of_factual_identity() -> None:
    model = build_mental_model("Память — это как библиотека.")

    assert model is not None
    assert model.mode == "metaphorical"
    assert any(
        relation.subject == "память"
        and relation.relation == "maps_to"
        and relation.object == "библиотека"
        for relation in model.relations
    )


def test_inference_never_creates_a_reflexive_spatial_relation() -> None:
    model = build_mental_model(
        "Башня слева от реки, а река слева от дома."
    )

    assert model is not None
    inferred = infer_relations(model.relations)
    assert all(relation.subject != relation.object for relation in inferred)


def test_single_letter_symbols_are_valid_abstract_spatial_objects() -> None:
    model = build_mental_model("Пусть A слева от B. Где A?")

    assert model is not None
    resolution = resolve_abstract_question(model)
    assert resolution is not None
    assert resolution.text == (
        "В этой воображаемой ситуации a находится слева от b."
    )


def test_follow_up_combines_only_the_connected_context_component() -> None:
    model = build_mental_model(
        "Где дом?",
        context_messages=(
            "Остров находится рядом с морем.",
            "Дом слева от башни.",
            "Башня слева от реки.",
        ),
    )

    assert model is not None
    assert any(
        relation.subject == "дом"
        and relation.relation == "left_of"
        and relation.object == "река"
        and relation.inferred
        for relation in model.relations
    )
    assert all(
        relation.subject not in {"остров", "море"}
        and relation.object not in {"остров", "море"}
        for relation in model.relations
    )


def test_unrelated_relation_in_question_does_not_hide_relevant_context() -> None:
    model = build_mental_model(
        "Где дом, если река слева от сада?",
        context_messages=("Дом находится внутри башни.",),
    )

    resolution = resolve_abstract_question(model)
    assert resolution is not None
    assert resolution.text == (
        "В этой воображаемой ситуации дом находится внутри башни."
    )


def test_fictional_mode_is_preserved_from_follow_up_context() -> None:
    model = build_mental_model(
        "Где сфера?",
        context_messages=("В вымышленном мире сфера находится над башней.",),
    )

    assert model is not None
    assert model.mode == "fictional"
    assert "в вымышленном мире сфера находится над башней." in model.assumptions
