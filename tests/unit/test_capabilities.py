from debilgi.language.capabilities import (
    action_requires_external_agency,
    assess_capability_question,
    capability_study_terms,
    realize_capability_assessment,
)
from debilgi.language.entities import analyze_entities
from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.structured_utterance import analyze_structured_utterance
from debilgi.personality.expression import PersonalityExpression
from debilgi.personality.perspective import build_personality_perspective


def structure(text: str):
    return analyze_structured_utterance(
        text,
        entity_analysis=analyze_entities(text, {}),
    )


def playful_perspective():
    return build_personality_perspective(
        PersonalityExpression(
            genome_signature="test:playful",
            chaos=0.7,
            curiosity=0.7,
            impulsivity=0.7,
            expressiveness=0.8,
            memory_attachment=0.7,
            novelty_drive=0.8,
            coherence_drive=0.7,
            target_tokens=30,
            candidate_count=40,
            sociability=0.7,
            suspicion=0.2,
            patience=0.3,
            assertiveness=0.7,
            warmth=0.7,
            playfulness=0.8,
            uncertainty=0.2,
        )
    )


def test_bot_bodily_capability_is_grounded_in_non_physical_agent_type() -> None:
    assessment = assess_capability_question(structure("А ты умеешь летать?"))

    assert assessment is not None
    assert assessment.subject_kind == "bot"
    assert assessment.action_lemma == "летать"
    assert not assessment.supported
    assert realize_capability_assessment(assessment) == "Нет, я не умею летать."


def test_bot_external_action_is_inferred_from_agent_boundary_not_one_action() -> None:
    assessment = assess_capability_question(structure("А ты умеешь копать землю?"))

    assert assessment is not None
    assert not assessment.supported
    assert assessment.reason == "action_requires_unavailable_external_agency"
    assert realize_capability_assessment(assessment) == ("Нет, я не умею копать землю.")


def test_bot_internal_action_is_supported_by_the_same_capability_model() -> None:
    assessment = assess_capability_question(structure("А ты умеешь анализировать текст?"))

    assert assessment is not None
    assert assessment.supported
    assert assessment.reason == "internal_computational_action"
    assert realize_capability_assessment(assessment) == ("Да, я умею анализировать текст.")


def test_modal_action_before_predicate_uses_the_same_capability_model() -> None:
    assessment = assess_capability_question(
        structure("Ты машиной управлять можешь?"),
        study_completed=True,
    )

    assert assessment is not None
    assert assessment.action_lemma == "управлять"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "машина"
    assert not assessment.supported
    surface = realize_capability_assessment(assessment)
    assert surface is not None
    assert "управлять" in surface.casefold()
    assert "машин" in surface.casefold()
    assert assessment.thought.negated


def test_elliptical_modal_question_recovers_action_from_dialogue() -> None:
    assessment = assess_capability_question(
        structure("А ты можешь?"),
        context_messages=("user: Ты машиной управлять можешь?",),
        study_completed=True,
    )

    assert assessment is not None
    assert assessment.action_lemma == "управлять"
    assert not assessment.supported


def test_unknown_capability_exposes_action_and_object_for_foreground_study() -> None:
    parsed = structure("Ты умеешь устраивать рейды?")

    assert assess_capability_question(parsed) is None
    assert capability_study_terms(parsed) == ("устраивать", "рейд")


def test_researched_external_action_becomes_a_grounded_self_answer() -> None:
    assessment = assess_capability_question(
        structure("Ты умеешь устраивать рейды?"),
        knowledge_facts=("Рейд — организованная групповая атака с участниками.",),
        study_completed=True,
    )

    assert assessment is not None
    assert not assessment.supported
    assert assessment.reason == "learned_external_action"
    assert realize_capability_assessment(assessment) == ("Нет, я не умею устраивать рейды.")


def test_nonmodal_self_activity_is_researched_and_answered_as_self_state() -> None:
    parsed = structure("Ты у себя пасёшь овечки?")
    polar_followup = structure("Так пасёшь ты овечек или нет?")

    assert assess_capability_question(parsed) is None
    assert capability_study_terms(parsed) == ("пасти", "овечка")
    assert assess_capability_question(polar_followup) is None
    assert capability_study_terms(polar_followup) == ("пасти", "овечка")

    assessment = assess_capability_question(
        parsed,
        knowledge_facts=("Пасти овец — присматривать за стадом животных во время выпаса.",),
        study_completed=True,
    )

    assert assessment is not None
    assert assessment.mode == "activity"
    assert assessment.reason == "ongoing_action_requires_unavailable_external_agency"
    assert realize_capability_assessment(assessment) == ("Нет, я не пасу овечки.")
    assert (
        assess_capability_question(
            polar_followup,
            knowledge_facts=("Пасти овец — присматривать за стадом животных во время выпаса.",),
            study_completed=True,
        )
        is not None
    )


def test_external_directive_is_grounded_in_agent_boundary() -> None:
    assessment = assess_capability_question(structure("Выкинь штурвал"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert realize_capability_assessment(assessment) == ("Я не могу выкинуть штурвал.")


def test_motion_directive_uses_the_same_agent_boundary() -> None:
    assessment = assess_capability_question(structure("Лети отсюда"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert assessment.action_lemma == "лететь"
    assert realize_capability_assessment(assessment) == ("Я не могу лететь.")


def test_prefixed_motion_verbs_share_an_external_action_family() -> None:
    assert all(
        action_requires_external_agency(action)
        for action in ("бегать", "сбегать", "прибежать", "понабегать")
    )
    assert not action_requires_external_agency("проанализировать")

    assessment = assess_capability_question(structure("Сбегай за сушняком"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert realize_capability_assessment(assessment) == (
        "Я не могу сбегать за сушняком."
    )


def test_future_tense_action_request_is_not_retrieved_as_an_old_fact() -> None:
    assessment = assess_capability_question(structure("Откроешь мне бутылку?"))

    assert assessment is not None
    assert assessment.mode == "directive"
    assert assessment.action_lemma == "открыть"
    assert assessment.action_object is not None
    assert realize_capability_assessment(assessment) == ("Я не могу открыть бутылку.")


def test_analytical_future_about_reply_becomes_local_commitment() -> None:
    assessment = assess_capability_question(structure("Будешь отвечать?"))

    assert assessment is not None
    assert assessment.mode == "commitment"
    assert assessment.action_lemma == "отвечать"
    assert realize_capability_assessment(assessment) == "Да, я буду отвечать."


def test_direct_desire_is_appraised_by_personality_before_realization() -> None:
    assessment = assess_capability_question(
        structure("Ты хочешь побегать?"),
        perspective=playful_perspective(),
    )

    assert assessment is not None
    assert assessment.mode == "personal_desire"
    assert assessment.supported
    assert any(
        item.startswith("desire:drive:playfulness:")
        for item in assessment.evidence
    )
    assert realize_capability_assessment(assessment) == "Да, я хочу побегать."


def test_past_bodily_experience_is_not_treated_as_missing_topic_data() -> None:
    assessment = assess_capability_question(structure("А яблоко ты ел?"))

    assert assessment is not None
    assert assessment.mode == "experience"
    assert realize_capability_assessment(assessment) == "Нет, я не ел яблоко."


def test_unknown_external_instrument_is_a_grounded_negative_self_state() -> None:
    assessment = assess_capability_question(
        structure("Ты кусачками пользуешься?"),
        analyses=(
            _word(
                "кусачки",
                "ручной шарнирно-губцевый инструмент для резки материала",
            ),
        ),
    )

    assert assessment is not None
    assert assessment.mode == "activity"
    assert not assessment.supported
    assert assessment.reason == ("ongoing_action_requires_unavailable_external_agency")
    assert assessment.action_object is not None
    assert assessment.action_object.grammatical_case == "instrumental"
    assert realize_capability_assessment(assessment) == ("Нет, я не пользуюсь кусачками.")


def test_unknown_resource_is_answered_only_after_foreground_study_attempt() -> None:
    before_study = assess_capability_question(structure("Ты кусачками пользуешься?"))
    assessment = assess_capability_question(
        structure("Ты кусачками пользуешься?"),
        study_completed=True,
    )

    assert before_study is None
    assert assessment is not None
    assert assessment.mode == "activity"
    assert not assessment.supported
    assert realize_capability_assessment(assessment) == ("Нет, я не пользуюсь кусачками.")


def _word(lemma: str, meaning: str) -> WordAnalysis:
    return WordAnalysis(
        surface_form=lemma,
        normalized_form=lemma,
        lemma=lemma,
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=(meaning,),
        style_labels=(),
        source="dictionary_cache",
        confidence=ConfidenceProfile(0.9, 0.9, 0.9, 0.9, 0.9),
    )


def test_polysemous_past_action_uses_the_objects_semantic_class() -> None:
    physical = assess_capability_question(
        structure("Ты терял когда-то ключи?"),
        analyses=(_word("ключ", "предмет для отпирания замка"),),
    )
    digital = assess_capability_question(
        structure("Ты терял когда-то данные?"),
        analyses=(_word("данные", "цифровая информация и запись"),),
    )

    assert physical is not None
    assert physical.mode == "experience"
    assert realize_capability_assessment(physical) == "Нет, я не терял ключи."
    assert digital is None


def test_human_profession_cannot_fly_without_technology() -> None:
    assessment = assess_capability_question(structure("Как думаешь? Астрономы умеют летать?"))

    assert assessment is not None
    assert assessment.subject_lemma == "астроном"
    assert assessment.reason == "human_requires_technology"
    assert realize_capability_assessment(assessment) == ("Нет, астрономы не могут летать.")


def test_unknown_or_naturally_flying_subject_is_not_forced_to_no() -> None:
    assert assess_capability_question(structure("Птицы умеют летать?")) is None
    assert assess_capability_question(structure("Люди умеют плавать?")) is None


def test_abstract_evaluative_state_does_not_require_a_body() -> None:
    assessment = assess_capability_question(structure("Ты можешь без тела иметь мнение и позицию?"))

    assert assessment is not None
    assert assessment.supported
    assert assessment.action_lemma == "иметь"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "мнение"
    assert assessment.reason == "internal_cognitive_or_evaluative_state"
    assert realize_capability_assessment(assessment) == ("Да, я умею иметь мнение.")


def test_unknown_named_game_still_forms_a_self_experience_answer() -> None:
    assessment = assess_capability_question(structure("Ты играл в гта 5?"))

    assert assessment is not None
    assert assessment.mode == "experience"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "гта 5"
    assert realize_capability_assessment(assessment) == "Нет, я не играл в гта 5."


def test_hypothetical_desire_uses_the_connected_dialogue_topic() -> None:
    assessment = assess_capability_question(
        structure("Интересно. А хотел бы сыграть?"),
        context_messages=("Играешь в гта 5?",),
    )

    assert assessment is not None
    assert assessment.mode == "hypothetical_desire"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "гта 5"
    assert realize_capability_assessment(assessment) == ("Да, я хочу сыграть в гта 5.")


def test_elliptical_desire_recovers_both_action_and_object_from_dialogue() -> None:
    assessment = assess_capability_question(
        structure("А хотел бы?"),
        context_messages=(
            "Ты занимался кунг-фу?",
            "Нет, я не занимался.",
        ),
    )

    assert assessment is not None
    assert assessment.mode == "hypothetical_desire"
    assert assessment.action_lemma == "заниматься"
    assert assessment.action_object is not None
    assert assessment.action_object.head.lemma == "кунг-фу"
    assert realize_capability_assessment(assessment) == ("Да, я хочу заниматься кунг-фу.")
