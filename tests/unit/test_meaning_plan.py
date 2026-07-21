from debilgi.cognition.models import ActionType
from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.meaning_plan import (
    build_response_meaning_plan,
    realize_response_meaning,
)
from debilgi.language.planning import ReplyContext, StructuredThought, utterance_plan
from debilgi.language.sense_resolution import resolve_contextual_senses
from debilgi.language.structured_utterance import analyze_structured_utterance


def plan_for(prompt: str, *facts: str):
    thought = StructuredThought(ActionType.SHORT_REPLY, None, "grounded answer", 0.84, ())
    plan = utterance_plan(
        thought,
        context=ReplyContext(
            prompt=prompt,
            required_facts=tuple(facts),
            seed="meaning-plan-test",
        ),
    )
    assert plan is not None
    structure = analyze_structured_utterance(prompt)
    resolution = resolve_contextual_senses(structure)
    return build_response_meaning_plan(plan, structure, resolution)


def test_realizes_grounded_subject_action_object_and_complement() -> None:
    plan = plan_for(
        "Что сделала лиса?",
        "Лиса взяла красный ключ со стола.",
    )

    response = realize_response_meaning(plan)

    assert plan.renderable
    assert response == "Лиса взяла красный ключ со стола."
    assert plan.propositions[0].source_fact == "Лиса взяла красный ключ со стола."


def test_realizes_nominal_definition_with_implicit_copula() -> None:
    plan = plan_for(
        "Что такое дракон?",
        "Дракон — мифическое существо с крыльями.",
    )

    response = realize_response_meaning(plan)

    assert response == "Дракон — мифическое существо с крыльями."
    assert plan.propositions[0].implicit_predicate == "быть"


def test_incomplete_or_ungrounded_input_does_not_create_a_candidate() -> None:
    no_facts = plan_for("продолжай разговор")
    fragment = plan_for("где это?", "карантине.")

    assert not no_facts.propositions
    assert realize_response_meaning(no_facts) is None
    assert not fragment.propositions
    assert realize_response_meaning(fragment) is None


def test_selected_contextual_sense_is_carried_into_meaning_plan() -> None:
    slang = WordAnalysis(
        surface_form="имба",
        normalized_form="имба",
        lemma="имба",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=("устаревшее имя",),
        style_labels=("server_slang",),
        source="server_slang",
        confidence=ConfidenceProfile(0.95, 0.82, 0.91, 0.82, 0.82),
        selected_meaning="чрезмерно сильный игровой объект",
        local_meaning="чрезмерно сильный игровой объект",
    )
    prompt = "этот меч имба"
    structure = analyze_structured_utterance(prompt, (slang,))
    resolution = resolve_contextual_senses(structure)
    thought = StructuredThought(ActionType.SHORT_REPLY, None, "answer", 0.8, ())
    utterance = utterance_plan(
        thought,
        context=ReplyContext(prompt=prompt, required_facts=("Меч наносит высокий урон.",)),
    )
    assert utterance is not None

    plan = build_response_meaning_plan(utterance, structure, resolution)

    selected = next(item for item in plan.selected_senses if item.lemma == "имба")
    assert selected.meaning == "чрезмерно сильный игровой объект"
    assert selected.source == "server_slang"


def test_links_grounded_facts_with_an_unambiguous_subject_pronoun() -> None:
    plan = plan_for(
        "Расскажи о драконе",
        "Дракон — мифическое существо с крыльями.",
        "Дракон живёт в горах.",
    )

    response = realize_response_meaning(plan)

    assert response == "Дракон — мифическое существо с крыльями. Он живёт в горах."
    assert plan.discourse_links[0].relation == "elaboration"
    assert plan.discourse_links[0].reference_strategy == "pronoun"
    assert plan.discourse_links[0].subject_surface == "он"
    assert tuple(item.kind for item in plan.response_moves) == (
        "main_claim",
        "supporting_detail",
    )


def test_keeps_explicit_subject_when_reference_is_not_proven() -> None:
    plan = plan_for(
        "Что произошло?",
        "Лиса взяла ключ.",
        "Ворон сел на дерево.",
    )

    response = realize_response_meaning(plan)

    assert response == "Лиса взяла ключ. Ворон сел на дерево."
    assert plan.discourse_links[0].reference_strategy == "explicit"


def test_rejects_isolated_fact_missing_a_required_valency_slot() -> None:
    plan = plan_for("Что сделала лиса?", "Лиса взяла.")

    assert not plan.propositions
    assert plan.rejections[0].reasons == ("missing_valency:direct_object",)
    assert realize_response_meaning(plan) is None


def test_resolves_ambiguous_preposition_from_predicate_valency() -> None:
    plan = plan_for("Куда она убрала ключ?", "Лиса спрятала ключ в норе.")

    assert realize_response_meaning(plan) == "Лиса спрятала ключ в норе."
    assert plan.propositions[0].complements[0].role == "destination"


def test_deduplicates_equivalent_grounded_propositions() -> None:
    plan = plan_for(
        "Где ключ?",
        "Ключ лежит на столе.",
        "Ключ лежит на столе!",
    )

    assert realize_response_meaning(plan) == "Ключ лежит на столе."
    assert len(plan.propositions) == 1
    assert plan.rejections[-1].reasons == ("duplicate_proposition",)


def test_preserves_coordinated_actions_and_inherits_their_subject() -> None:
    plan = plan_for(
        "Что сделала лиса?",
        "Лиса взяла ключ и спрятала его в норе.",
    )

    response = realize_response_meaning(plan)

    assert response == "Лиса взяла ключ. Она спрятала его в норе."
    assert len(plan.propositions) == 2
    assert plan.propositions[1].implicit_subject == "previous_clause_subject"
    assert plan.discourse_links[0].reference_strategy == "pronoun"


def test_preserves_irregular_locative_surface_from_grounded_fact() -> None:
    plan = plan_for("Где живёт волк?", "Волк живёт в лесу.")

    assert realize_response_meaning(plan) == "Волк живёт в лесу."


def test_predicate_valency_preserves_non_accusative_government() -> None:
    plan = plan_for("Чего боится дракон?", "Дракон боится холода.")

    assert realize_response_meaning(plan) == "Дракон боится холода."


def test_response_moves_keep_explicit_contrast_as_a_qualification() -> None:
    plan = plan_for(
        "Что известно о драконе?",
        "Дракон — сильное существо.",
        "Однако дракон боится холода.",
    )

    assert realize_response_meaning(plan) == ("Дракон — сильное существо. Однако он боится холода.")
    assert tuple(item.kind for item in plan.response_moves) == (
        "main_claim",
        "qualification",
    )
