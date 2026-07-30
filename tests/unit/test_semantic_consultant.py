import pytest

from debilgi.language.semantic_consultant import (
    ConsultationResult,
    SemanticConsultationService,
)


def test_structured_result_caps_external_confidence_and_validates_fields() -> None:
    result = ConsultationResult.from_payload(
        {
            "interpretation": "Пользователь спрашивает значение сленгового выражения.",
            "intent": "definition_question",
            "entities": [
                {"surface": "база", "lemma": "база", "type": "abstract"}
            ],
            "relations": [
                {"source": "пользователь", "relation": "asks_about", "target": "база"}
            ],
            "ambiguities": [],
            "facts": ["В сленге «база» может означать одобрение позиции."],
            "response_text": "В этом контексте «база» означает одобрение сказанного.",
            "confidence": 0.99,
        },
        provider="test-provider",
        model="test-model",
    )

    assert result.intent == "definition_question"
    assert result.entities[0].lemma == "база"
    assert result.confidence == pytest.approx(0.86)
    assert result.response_text is None


def test_structured_result_rejects_empty_output_and_discards_external_wording() -> None:
    with pytest.raises(ValueError, match="no usable interpretation"):
        ConsultationResult.from_payload({}, provider="test-provider", model="test")

    result = ConsultationResult.from_payload(
        {
            "interpretation": "разбор",
            "response_text": "Команды узнай названного объекта.",
            "confidence": 0.8,
        },
        provider="test-provider",
        model="test",
    )

    assert result.response_text is None


def test_response_plan_accepts_only_abstract_lemmas_roles_and_evidence_ids() -> None:
    result = ConsultationResult.from_payload(
        {
            "interpretation": "План ответа без поверхности.",
            "intent": "answer",
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
                                    "lemma": "готовый ответ",
                                    "part_of_speech": "noun",
                                },
                            ],
                            "clause_kind": "statement",
                            "negated": True,
                            "evidence_ids": ["runtime.observable"],
                            "surface": "Этого поля в контракте нет.",
                        }
                    ],
                    "confidence": 0.8,
                    "response_text": "И этого поля тоже нет.",
                }
            ],
            "confidence": 0.8,
        },
        provider="test-provider",
        model="semantic-only",
    )

    assert len(result.response_plans) == 1
    proposition = result.response_plans[0].propositions[0]
    assert proposition.predicate == "наблюдать"
    assert [argument.lemma for argument in proposition.arguments] == ["я"]
    assert proposition.evidence_ids == ("runtime.observable",)
    assert "surface" not in result.as_dict()["response_plans"][0]["propositions"][0]


def test_missing_teacher_arrays_are_recovered_from_abstract_structure() -> None:
    result = ConsultationResult.from_payload(
        {
            "interpretation": "Пользователь просит агента оценить себя.",
            "intent": "self_reflection",
            "tokens": [
                {
                    "surface": "ты",
                    "lemma": "ты",
                    "part_of_speech": "pronoun",
                    "role": "subject",
                },
                {
                    "surface": "думаешь",
                    "lemma": "думать",
                    "part_of_speech": "verb",
                    "role": "predicate",
                },
            ],
            "sentence_pattern": [],
            "expected_response_acts": [],
            "social_frame": {
                "prompt_act": "question",
                "speaker_stance": "neutral",
                "emotion": "curious",
                "response_obligation": "decide",
                "recommended_moves": ["answer_first", "state_self_assessment"],
                "avoid_moves": [],
                "directness": 1.0,
                "warmth": 0.2,
                "playfulness": 0.0,
            },
            "confidence": 0.8,
        },
        provider="cloudflare-workers-ai",
        model="test",
    )

    assert result.sentence_pattern == ("subject", "predicate")
    assert result.expected_response_acts == (
        "decide",
        "answer_first",
        "state_self_assessment",
    )


def test_high_lexical_score_without_reply_bound_answer_still_allows_consultation() -> None:
    service = object.__new__(SemanticConsultationService)
    service.enabled = True
    service.trigger_threshold = 0.72

    assert service.should_consult(
        local_confidence=1.0,
        unknown_terms=(),
        has_bound_answer=False,
    )
    assert not service.should_consult(
        local_confidence=0.2,
        unknown_terms=("слово",),
        has_bound_answer=True,
    )
