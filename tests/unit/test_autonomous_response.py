from dataclasses import replace

from debilgi.language.autonomous_response import realize_autonomous_responses
from debilgi.language.semantic_consultant import ConsultationResult


def _result(*, evidence_id: str) -> ConsultationResult:
    return ConsultationResult.from_payload(
        {
            "interpretation": "Семантические альтернативы без текста ответа.",
            "intent": "answer",
            "response_text": "ГОТОВЫЙ ВНЕШНИЙ ОТВЕТ НЕ ДОЛЖЕН ПОПАСТЬ В РЕЗУЛЬТАТ.",
            "response_plans": [
                {
                    "move": "answer",
                    "stance": "direct",
                    "propositions": [
                        {
                            "predicate": "наблюдать",
                            "predicate_features": {"tense": "present"},
                            "arguments": [
                                {
                                    "role": "subject",
                                    "lemma": "я",
                                    "part_of_speech": "pronoun",
                                },
                                {
                                    "role": "direct_object",
                                    "lemma": "раскладка",
                                    "part_of_speech": "noun",
                                    "modifiers": ["системный"],
                                },
                            ],
                            "clause_kind": "statement",
                            "negated": True,
                            "evidence_ids": [evidence_id],
                        }
                    ],
                    "confidence": 0.9,
                }
            ],
            "confidence": 0.9,
        },
        provider="test-planner",
        model="semantic-only",
    )


def test_external_text_is_discarded_and_local_morphology_authors_surface() -> None:
    consultation = _result(evidence_id="runtime.layout.observable")

    candidates = realize_autonomous_responses(
        consultation,
        perspective=None,
        allowed_evidence_ids=("runtime.layout.observable",),
    )

    assert consultation.response_text is None
    assert [candidate.text for candidate in candidates] == [
        "Я не наблюдаю системную раскладку."
    ]
    assert "ВНЕШНИЙ" not in candidates[0].text


def test_plan_with_invented_evidence_is_rejected_before_realization() -> None:
    candidates = realize_autonomous_responses(
        _result(evidence_id="invented.fact"),
        perspective=None,
        allowed_evidence_ids=("runtime.layout.observable",),
    )

    assert candidates == ()


def test_provider_evidence_wrapper_is_normalized_only_to_an_allowed_id() -> None:
    candidates = realize_autonomous_responses(
        _result(evidence_id="grounding_facts.id=runtime.layout.observable"),
        perspective=None,
        allowed_evidence_ids=("runtime.layout.observable",),
    )

    assert [candidate.text for candidate in candidates] == [
        "Я не наблюдаю системную раскладку."
    ]


def test_structured_fact_overrides_distorted_roles_and_polarity() -> None:
    consultation = ConsultationResult.from_payload(
        {
            "interpretation": "План с формальной ссылкой, но искажёнными ролями.",
            "intent": "answer",
            "response_plans": [
                {
                    "move": "answer",
                    "stance": "direct",
                    "propositions": [
                        {
                            "predicate": "иметь",
                            "predicate_features": {"tense": "present"},
                            "arguments": [
                                {
                                    "role": "subject",
                                    "lemma": "раскладка",
                                    "part_of_speech": "noun",
                                }
                            ],
                            "clause_kind": "statement",
                            "negated": False,
                            "evidence_ids": ["runtime.layout.observable"],
                        }
                    ],
                    "confidence": 0.8,
                }
            ],
            "confidence": 0.8,
        },
        provider="weak-semantic-planner",
        model="semantic-only",
    )
    facts = (
        (
            "runtime.layout.observable",
            (
                "predicate=наблюдать;subject=agent;"
                "direct_object=раскладка;"
                "direct_object_modifiers=системный;"
                "tense=present;truth=false. "
                "Агент не наблюдает системную раскладку."
            ),
        ),
    )

    candidates = realize_autonomous_responses(
        consultation,
        perspective=None,
        allowed_evidence_ids=("runtime.layout.observable",),
        grounding_facts=facts,
    )

    assert [candidate.text for candidate in candidates] == [
        "Я не наблюдаю системную раскладку."
    ]


def test_participle_cannot_masquerade_as_nominal_argument() -> None:
    consultation = ConsultationResult.from_payload(
        {
            "interpretation": "Ошибочный план.",
            "intent": "answer",
            "response_plans": [
                {
                    "move": "answer",
                    "stance": "neutral",
                    "propositions": [
                        {
                            "predicate": "отвечать",
                            "predicate_features": {"tense": "present"},
                            "arguments": [
                                {
                                    "role": "subject",
                                    "lemma": "я",
                                    "part_of_speech": "pronoun",
                                },
                                {
                                    "role": "direct_object",
                                    "lemma": "отвечающий",
                                    "part_of_speech": "noun",
                                },
                            ],
                            "clause_kind": "statement",
                            "negated": False,
                            "evidence_ids": ["knowledge.answer"],
                        }
                    ],
                    "confidence": 0.8,
                }
            ],
            "confidence": 0.8,
        },
        provider="weak-semantic-planner",
        model="semantic-only",
    )

    candidates = realize_autonomous_responses(
        consultation,
        perspective=None,
        allowed_evidence_ids=("knowledge.answer",),
        grounding_facts=(("knowledge.answer", "Я отвечаю на вопрос."),),
    )

    assert candidates == ()


def test_duplicate_propositions_are_realized_once() -> None:
    consultation = _result(evidence_id="runtime.layout.observable")
    proposition = consultation.response_plans[0].propositions[0]
    plan = consultation.response_plans[0]
    consultation = replace(
        consultation,
        response_plans=(
            replace(
                plan,
                propositions=(proposition, proposition),
            ),
        ),
    )

    candidates = realize_autonomous_responses(
        consultation,
        perspective=None,
        allowed_evidence_ids=("runtime.layout.observable",),
    )

    assert [candidate.text for candidate in candidates] == [
        "Я не наблюдаю системную раскладку."
    ]


def test_ambiguous_infinitive_is_selected_as_verb_in_predicate_slot() -> None:
    consultation = ConsultationResult.from_payload(
        {
            "interpretation": "Глагольный план.",
            "intent": "answer",
            "response_plans": [
                {
                    "move": "answer",
                    "stance": "direct",
                    "propositions": [
                        {
                            "predicate": "мочь",
                            "predicate_features": {"tense": "present"},
                            "arguments": [
                                {
                                    "role": "subject",
                                    "lemma": "я",
                                    "part_of_speech": "pronoun",
                                },
                                {
                                    "role": "action_complement",
                                    "lemma": "выкинуть",
                                    "part_of_speech": "verb",
                                },
                                {
                                    "role": "direct_object",
                                    "lemma": "мусор",
                                    "part_of_speech": "noun",
                                },
                            ],
                            "clause_kind": "statement",
                            "negated": True,
                            "evidence_ids": ["agent.capability"],
                        }
                    ],
                    "confidence": 0.86,
                }
            ],
            "confidence": 0.86,
        },
        provider="weak-semantic-planner",
        model="semantic-only",
    )
    fact = (
        "predicate=мочь;subject=agent;action_complement=выкинуть;"
        "direct_object=мусор;tense=present;truth=false."
    )

    candidates = realize_autonomous_responses(
        consultation,
        perspective=None,
        allowed_evidence_ids=("agent.capability",),
        grounding_facts=(("agent.capability", fact),),
    )

    assert [candidate.text for candidate in candidates] == [
        "Я не могу выкинуть мусор."
    ]
