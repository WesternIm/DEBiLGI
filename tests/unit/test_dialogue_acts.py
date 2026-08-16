from debilgi.language.dialogue_acts import (
    analyze_dialogue_act,
    candidate_fulfills_obligation,
    realize_dialogue_act,
    refine_dialogue_act,
)
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.question_reasoning import analyze_semantic_question
from debilgi.language.relation_reasoning import analyze_relation_question
from debilgi.language.self_state_reasoning import analyze_self_state
from debilgi.language.structured_utterance import analyze_structured_utterance


def _act(text: str):
    structure = analyze_structured_utterance(text)
    return analyze_dialogue_act(
        structure,
        pragmatics=analyze_pragmatics(text),
        semantic_question=analyze_semantic_question(structure),
        relation_question=analyze_relation_question(structure),
        self_state=analyze_self_state(structure),
    )


def test_positive_evaluation_is_understood_above_its_surface_words() -> None:
    for text in ("Молодец хуле", "Красавчик, блин", "Отлично сделано"):
        act = _act(text)

        assert act.kind == "positive_evaluation"
        assert act.response_obligation == "acknowledge_evaluation"
        assert act.complete is True
        assert realize_dialogue_act(act) is not None


def test_parenthetical_imperative_does_not_hide_answer_evaluation() -> None:
    act = _act("Твой ответ логичный, прикинь")

    assert act.kind == "positive_evaluation"
    assert act.response_obligation == "acknowledge_evaluation"
    assert realize_dialogue_act(act) == "Да, я стараюсь."


def test_negative_evaluation_creates_a_repair_obligation() -> None:
    for text in ("Это плохо", "Ответ — хуйня"):
        act = _act(text)

        assert act.kind == "negative_evaluation"
        assert act.response_obligation == "address_negative_evaluation"
        assert act.complete is True


def test_word_order_does_not_change_an_alternative_question_contract() -> None:
    acts = tuple(
        _act(text)
        for text in (
            "Краба выбираешь или рака?",
            "Выбираешь рака или краба?",
            "Рак или краб?",
        )
    )

    assert all(item.kind == "alternative_choice" for item in acts)
    assert all(item.response_obligation == "select_or_compare" for item in acts)
    assert all(item.complete for item in acts)
    expected_pairs = ({"краба", "рака"}, {"краб", "рак"})
    assert all(set(item.objects) in expected_pairs for item in acts)


def test_complete_question_and_fragment_have_different_obligations() -> None:
    question = _act("Что ты знаешь о книге?")
    fragment = _act("книга")

    assert question.response_obligation == "answer_question"
    assert question.complete is True
    assert fragment.response_obligation == "clarify_fragment"
    assert fragment.complete is False


def test_complete_elliptical_statement_gets_an_acknowledgement() -> None:
    act = _act("Просто ключи взял бы")

    assert act.kind == "statement"
    assert act.response_obligation == "continue_topic"
    assert act.complete is True
    assert realize_dialogue_act(act) == "Я понял тебя."


def test_possessive_dialogue_fragment_is_not_treated_as_a_definition_request() -> None:
    act = _act("Твой ответ")

    assert act.kind == "dialogue_reference_fragment"
    assert act.response_obligation == "acknowledge_reference"
    assert act.objects == ("ответ",)
    assert act.complete is True
    assert realize_dialogue_act(act) == "Я понял тебя."


def test_external_model_can_only_refine_an_act_not_supply_reply_text() -> None:
    fragment = _act("люто")
    refined = refine_dialogue_act(
        fragment,
        external_intent="compliment",
        confidence=0.84,
    )

    assert refined.kind == "positive_evaluation"
    assert refined.response_obligation == "acknowledge_evaluation"
    assert refined.evidence[-1] == "external_intent:praise"
    assert realize_dialogue_act(refined) == "Да, я стараюсь."


def test_teacher_can_reclassify_a_complete_but_generic_question() -> None:
    generic = _act("Что происходит?")
    refined = refine_dialogue_act(
        generic,
        external_intent="self_reflection",
        confidence=0.8,
        external_response_obligation="decide",
        expected_response_acts=("express_position",),
    )

    assert refined.kind == "opinion_question"
    assert refined.response_obligation == "express_position"
    assert "external_intent:position_request" in refined.evidence


def test_teacher_reclassification_normalizes_target_and_quality() -> None:
    generic = _act("Германия крутая страна?")
    refined = refine_dialogue_act(
        generic,
        external_intent="opinion_question",
        confidence=0.8,
        external_response_obligation="answer",
        expected_response_acts=("answer", "answer_first"),
    )

    assert refined.kind == "opinion_question"
    assert refined.response_obligation == "express_position"
    assert refined.objects == ("германия", "крутая")


def test_candidate_meaning_not_generator_name_fulfils_the_obligation() -> None:
    praise = _act("Отлично")
    choice = _act("Чай или кофе?")

    for source in ("dialogue_act", "graph"):
        assert candidate_fulfills_obligation(
            praise,
            source,
            text="Я стараюсь.",
            metadata={"response_move": "acknowledge"},
        )
    for source in ("semantic_reasoning", "semantic_synthesis"):
        assert candidate_fulfills_obligation(
            choice,
            source,
            text="Я выбираю чай.",
        )
        assert not candidate_fulfills_obligation(
            choice,
            source,
            text="Книга лежит на полке.",
        )


def test_relation_question_rejects_definition_and_graph_sources() -> None:
    act = _act("У тебя есть какой-то объект?")

    assert act.kind == "relation_question"
    assert act.response_obligation == "resolve_relation"
    for source in ("relation_reasoning", "lexical_reference", "graph"):
        assert candidate_fulfills_obligation(
            act,
            source,
            text="Нет, у меня нет этого объекта.",
        )
        assert not candidate_fulfills_obligation(
            act,
            source,
            text="Объект — часть философской категории.",
        )


def test_self_state_question_requires_a_generated_semantic_thought() -> None:
    act = _act("А ты понял идею?")

    assert act.kind == "self_state_question"
    assert act.response_obligation == "resolve_self_state"
    for source in ("semantic_thought", "dialogue_act", "graph"):
        assert candidate_fulfills_obligation(
            act,
            source,
            text="Да, я понял идею.",
        )
        assert not candidate_fulfills_obligation(
            act,
            source,
            text="Идея — форма мышления.",
        )


def test_reported_misunderstanding_blocks_fact_parroting() -> None:
    act = _act("Я не понял")

    assert act.kind == "understanding_failure_report"
    assert act.response_obligation == "repair_understanding"
    for source in ("semantic_thought", "grounded_surface"):
        assert candidate_fulfills_obligation(
            act,
            source,
            text="Я неправильно понял предыдущий вопрос.",
            metadata={"response_move": "targeted_repair"},
        )
        assert not candidate_fulfills_obligation(
            act,
            source,
            text="Умник — телесериал.",
        )


def test_self_assessment_is_a_question_about_the_bots_position() -> None:
    act = _act("А ты считаешь себя умным?")

    assert act.kind == "self_state_question"
    assert act.response_obligation == "resolve_self_state"
    assert act.objects == ("умный",)


def test_nominal_self_role_is_conversation_not_an_information_mode() -> None:
    act = _act("Ты чё герой что-ли?")

    assert act.kind == "self_state_question"
    assert act.response_obligation == "resolve_self_state"
    assert act.objects == ("герой",)


def test_colloquial_predicative_confirmation_resolves_to_a_self_trait() -> None:
    act = _act("Ты же круто, да?")

    assert act.kind == "self_state_question"
    assert act.response_obligation == "resolve_self_state"
    assert act.objects == ("крутой",)
    assert act.complete is True


def test_unknown_fragment_does_not_choose_its_reply_inside_act_detection() -> None:
    act = _act("ебануться")

    assert act.kind == "fragment"
    assert act.response_obligation == "clarify_fragment"
    assert act.objects == ("ебануться",)
    assert realize_dialogue_act(act) is None


def test_open_self_reflection_is_not_treated_as_a_dictionary_question() -> None:
    act = _act("Что ты думаешь о себе?")

    assert act.kind == "self_reflection_question"
    assert act.response_obligation == "express_position"


def test_rhetorical_challenge_requests_repair_not_a_literal_fact() -> None:
    act = _act("Ты что мозгом поперхнулся?")

    assert act.kind == "rhetorical_challenge"
    assert act.response_obligation == "repair_previous_answer"
    assert realize_dialogue_act(act) == "Я понял предыдущий вопрос неправильно."


def test_understanding_challenge_is_acknowledged_instead_of_treated_as_question() -> None:
    act = _act("Ой бля, не делай вид, что ты не понял, ахахахах")

    assert act.kind == "understanding_challenge"
    assert act.response_obligation == "acknowledge_understanding"
    assert realize_dialogue_act(act) == "Да, я понял намёк."


def test_elliptical_follow_up_keeps_the_previous_answer_as_its_target() -> None:
    act = _act("А что тогда?")

    assert act.kind == "elliptical_follow_up"
    assert act.response_obligation == "repair_previous_answer"


def test_prior_claim_correction_requires_repair_instead_of_topic_continuation() -> None:
    act = _act("Ты сказал делал")

    assert act.kind == "prior_answer_correction"
    assert act.response_obligation == "repair_previous_answer"


def test_explicit_opinion_request_creates_a_position_obligation() -> None:
    act = _act("Бобун лох или нет? Да или нет? Как думаешь?")

    assert act.kind == "opinion_question"
    assert act.response_obligation == "express_position"
    assert act.objects == ("бобун", "лох")
    # Dialogue-act recognition decides the obligation only.  The stance and
    # its wording belong to the independent position-reasoning layer.
    assert realize_dialogue_act(act) is None


def test_evaluative_zero_copula_question_requests_an_owned_position() -> None:
    act = _act("Лохи хорошие люди?")

    assert act.kind == "opinion_question"
    assert act.response_obligation == "express_position"
    assert act.objects == ("лохи", "хорошие")
    assert "construction:evaluative_copular_question" in act.evidence


def test_wh_question_requires_the_requested_semantic_slot() -> None:
    act = _act("Куда орут?")

    assert "question_target:куда" in act.evidence
    assert not candidate_fulfills_obligation(
        act,
        "grounded_surface",
        text="Люди орут.",
    )
    assert candidate_fulfills_obligation(
        act,
        "semantic_reasoning",
        text="Они орут на улице.",
    )


def test_nominal_cause_question_is_not_reduced_to_its_embedded_preference() -> None:
    act = _act("Какая причина того, что ты не любишь небо?")

    assert "question_target:почему" in act.evidence
    assert not candidate_fulfills_obligation(
        act,
        "semantic_thought",
        text="Нет, я не люблю небо.",
        metadata={"response_move": "answer"},
    )
    assert candidate_fulfills_obligation(
        act,
        "dialogue_context",
        text="У меня нет достаточных оснований.",
        metadata={"response_move": "answer", "answer_slot": "почему"},
    )


def test_compound_study_and_opinion_directive_requires_both_subgoals() -> None:
    act = _act("Изучи, что такое небо, и скажи своё мнение")

    assert act.kind == "compound_directive"
    assert act.response_obligation == "study_and_express_position"
    assert act.objects == ("небо",)
    assert not candidate_fulfills_obligation(
        act,
        "lexical_reference",
        text="Небо — пространство над Землёй.",
        metadata={
            "completed_subgoals": ("study",),
            "knowledge_component": "Небо — пространство над Землёй.",
        },
    )
    assert candidate_fulfills_obligation(
        act,
        "semantic_synthesis",
        text="Небо — пространство над Землёй. Да, я люблю небо.",
        metadata={
            "completed_subgoals": ("study", "express_position"),
            "knowledge_component": "Небо — пространство над Землёй.",
            "position_component": "Да, я люблю небо.",
        },
    )


def test_plain_study_directive_requires_knowledge_and_an_answer() -> None:
    act = _act("Изучи что такое гта 5 и ответь")

    assert act.kind == "study_directive"
    assert act.response_obligation == "study_and_answer"
    assert act.objects == ("гта", "5")
    assert not candidate_fulfills_obligation(
        act,
        "grounded_surface",
        text="Ты изучи такое небо.",
    )
    assert candidate_fulfills_obligation(
        act,
        "knowledge_retrieval",
        text="Grand Theft Auto V — компьютерная игра.",
        metadata={
            "completed_subgoals": ("study", "answer"),
            "knowledge_component": "Grand Theft Auto V — компьютерная игра.",
        },
    )


def test_elliptical_cause_question_keeps_priority_over_following_evidence() -> None:
    act = _act("Почему? Ты сейчас оцениваешь и отвечаешь мне")

    assert act.kind == "information_question"
    assert act.response_obligation == "answer_question"
    assert "question_target:почему" in act.evidence
    assert not candidate_fulfills_obligation(
        act,
        "grounded_surface",
        text="Ты отвечай на вопрос.",
    )
    assert candidate_fulfills_obligation(
        act,
        "semantic_reasoning",
        text="Потому что я уже оцениваю контекст.",
    )


def test_contextual_self_label_becomes_a_complete_evaluation_contract() -> None:
    structure = analyze_structured_utterance("гений")
    self_state = analyze_self_state(
        structure,
        replied_to_agent=True,
        context_messages=("Я ответил бессвязно.",),
    )
    act = analyze_dialogue_act(structure, self_state=self_state)

    assert act.kind == "self_evaluation"
    assert act.response_obligation == "respond_to_evaluation"
    assert act.objects == ("гений",)
    assert act.complete is True


def test_negated_second_person_state_is_not_reduced_to_a_fragment() -> None:
    structure = analyze_structured_utterance("а ты будто не знаешь")
    self_state = analyze_self_state(
        structure,
        resolved_bindings=(("direct_object", "гений"),),
    )
    act = analyze_dialogue_act(structure, self_state=self_state)

    assert act.kind == "self_state_attribution"
    assert act.response_obligation == "resolve_self_state"
    assert act.objects == ("гений",)
