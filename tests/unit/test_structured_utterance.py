from debilgi.language.entities import analyze_entities
from debilgi.language.lexicon import ConfidenceProfile, WordAnalysis
from debilgi.language.structured_utterance import analyze_structured_utterance

PAYLOAD = {
    "author_identity": {
        "id": 100,
        "username": "ferzevina",
        "display_name": "Эвелина",
    },
    "bot_identity": {
        "id": 123,
        "username": "DEBiLGI",
        "display_name": "DEBiLGI",
    },
}


def test_builds_subject_predicate_and_object_frame() -> None:
    analysis = analyze_structured_utterance("Девушка прочитала книгу.")
    frame = analysis.clauses[0]

    assert analysis.tokens[frame.subject_indices[0]].lemma == "девушка"
    assert frame.predicate_index is not None
    assert analysis.tokens[frame.predicate_index].lemma == "прочитать"
    assert analysis.tokens[frame.direct_object_indices[0]].lemma == "книга"
    assert analysis.tokens[frame.subject_indices[0]].semantic_role == "subject"
    assert analysis.tokens[frame.predicate_index].semantic_role == "predicate"


def test_expressive_surface_keeps_canonical_lemma_and_intensity_signal() -> None:
    analysis = analyze_structured_utterance("привеееет")

    assert analysis.tokens[0].surface == "привеееет"
    assert analysis.tokens[0].normalized == "привет"
    assert analysis.tokens[0].lemma == "привет"
    assert "pragmatic:greeting" in analysis.niche_signals
    assert any(signal.startswith("expressive_elongation:") for signal in analysis.niche_signals)


def test_separates_vocative_from_imperative_clause() -> None:
    text = "Маша, открой дверь."
    entities = analyze_entities(text, PAYLOAD)
    analysis = analyze_structured_utterance(text, entity_analysis=entities)
    frame = analysis.clauses[0]

    assert tuple(analysis.tokens[index].lemma for index in frame.vocative_indices) == ("маша",)
    assert frame.implicit_subject == "addressee"
    assert frame.clause_kind == "command"
    assert frame.predicate_index is not None
    assert analysis.tokens[frame.predicate_index].lemma == "открыть"
    assert analysis.tokens[frame.direct_object_indices[0]].lemma == "дверь"
    assert "direct_address" in analysis.niche_signals


def test_preserves_known_nickname_as_entity_and_subject() -> None:
    text = "ferzevina открыла сервер."
    entities = analyze_entities(text, PAYLOAD)
    analysis = analyze_structured_utterance(text, entity_analysis=entities)
    subject = analysis.tokens[analysis.clauses[0].subject_indices[0]]

    assert subject.surface == "ferzevina"
    assert subject.entity_kind == "person"
    assert subject.entity_id == 100
    assert subject.corrected_form is None
    assert subject.semantic_role == "subject"
    assert "entity:person" in analysis.niche_signals


def test_keeps_typo_correction_as_a_separate_token_property() -> None:
    analysis = analyze_structured_utterance("таоелка лежит на столе")
    token = analysis.tokens[0]

    assert token.surface == "таоелка"
    assert token.corrected_form == "тарелка"
    assert token.lemma == "тарелка"
    assert "spelling_correction:таоелка->тарелка" in analysis.niche_signals


def test_marks_server_slang_without_turning_it_into_a_response_template() -> None:
    slang = WordAnalysis(
        surface_form="имба",
        normalized_form="имба",
        lemma="имба",
        part_of_speech="NOUN",
        grammatical_features={},
        dictionary_meanings=(),
        style_labels=("server_slang", "gaming"),
        source="server_slang",
        confidence=ConfidenceProfile(0.95, 0.82, 0.91, 0.82, 0.82),
        selected_meaning="чрезмерно сильный игровой объект",
        local_meaning="чрезмерно сильный игровой объект",
    )

    analysis = analyze_structured_utterance("этот меч имба", (slang,))
    token = next(item for item in analysis.tokens if item.surface == "имба")

    assert token.selected_meaning == "чрезмерно сильный игровой объект"
    assert token.lexical_source == "server_slang"
    assert "contextual_lexicon:server_slang" in analysis.niche_signals


def test_marks_abbreviation_as_a_named_semantic_object() -> None:
    analysis = analyze_structured_utterance("ИИ анализирует текст")
    token = analysis.tokens[0]

    assert token.lemma == "ии"
    assert token.entity_kind == "abbreviation"
    assert "abbreviation:ии" in analysis.niche_signals


def test_detects_attention_ellipsis_and_nominal_copula() -> None:
    attention = analyze_structured_utterance("ауу")
    nominal = analyze_structured_utterance("ну ты гений?")

    assert "attention_call" in attention.niche_signals
    assert attention.clauses[0].elliptical
    assert nominal.clauses[0].implicit_predicate == "быть"
    assert nominal.clauses[0].clause_kind == "question"
    predicate_complement = nominal.clauses[0].predicate_complement_indices[0]
    assert nominal.tokens[predicate_complement].lemma == "гений"


def test_question_word_and_profanity_keep_structural_pragmatic_roles() -> None:
    analysis = analyze_structured_utterance("чего бля")
    frame = analysis.clauses[0]
    intensifier = next(item for item in analysis.tokens if item.normalized == "бля")

    assert frame.clause_kind == "question"
    assert frame.question_target_index is not None
    assert intensifier.semantic_role == "pragmatic_marker"
    assert "emotional_intensifier" in analysis.niche_signals


def test_finds_inverted_subject_after_question_predicate() -> None:
    analysis = analyze_structured_utterance("Насколько считаешь ты себя осознанным?")
    frame = analysis.clauses[0]

    assert tuple(analysis.tokens[index].lemma for index in frame.subject_indices) == ("ты",)


def test_second_person_choice_has_implicit_addressee() -> None:
    analysis = analyze_structured_utterance("Краба выбираешь или рака?")
    frame = analysis.clauses[0]

    assert frame.clause_kind == "question"
    assert frame.implicit_subject == "addressee"
    assert analysis.tokens[frame.predicate_index].lemma == "выбирать"
    assert tuple(analysis.tokens[index].lemma for index in frame.direct_object_indices) == (
        "краб",
        "рак",
    )
    assert len(frame.coordinations) == 1
    assert frame.coordinations[0].relation == "alternative"


def test_first_person_finite_verb_has_implicit_speaker() -> None:
    analysis = analyze_structured_utterance("Просто удивляюсь")
    frame = analysis.clauses[0]

    assert frame.implicit_subject == "speaker"
    assert analysis.tokens[frame.predicate_index].lemma == "удивляться"


def test_service_words_operate_on_content_without_becoming_arguments() -> None:
    analysis = analyze_structured_utterance("Что ты знаешь о книге?")
    frame = analysis.clauses[0]

    assert analysis.tokens[frame.question_target_index].lemma == "что"
    assert tuple(item.role for item in frame.complements) == ("topic",)
    assert tuple(
        analysis.tokens[index].lemma for item in frame.complements for index in item.object_indices
    ) == ("книга",)
    assert all(
        analysis.tokens[index].semantic_role != "function_word"
        for index in (*frame.subject_indices, *frame.direct_object_indices)
    )


def test_discourse_marker_changes_register_but_not_predicate_arguments() -> None:
    analysis = analyze_structured_utterance("Молодец хуле")
    frame = analysis.clauses[0]
    marker = next(item for item in analysis.tokens if item.normalized == "хуле")

    assert marker.semantic_role == "pragmatic_marker"
    assert marker.index not in frame.predicate_complement_indices


def test_preposed_accusative_is_an_object_not_a_second_subject() -> None:
    analysis = analyze_structured_utterance("Я тебя спрашиваю")
    frame = analysis.clauses[0]

    assert tuple(analysis.tokens[index].lemma for index in frame.subject_indices) == ("я",)
    assert tuple(analysis.tokens[index].lemma for index in frame.direct_object_indices) == ("ты",)
    assert analysis.tokens[frame.predicate_index].lemma == "спрашивать"


def test_preposed_instrumental_object_follows_predicate_government() -> None:
    analysis = analyze_structured_utterance("Ты кусачками пользуешься?")
    frame = analysis.clauses[0]

    assert tuple(analysis.tokens[index].lemma for index in frame.direct_object_indices) == (
        "кусачки",
    )
    assert analysis.tokens[frame.predicate_index].lemma == "пользоваться"
    assert any(
        analysis.tokens[arc.head_index].lemma == "пользоваться"
        and analysis.tokens[arc.dependent_index].lemma == "кусачки"
        and arc.relation == "obj"
        for arc in analysis.dependencies
    )


def test_subordinate_question_word_does_not_turn_statement_into_question() -> None:
    analysis = analyze_structured_utterance("Я думаю ты тогда должен узнать что это такое")

    assert analysis.clauses[0].clause_kind == "statement"
    assert analysis.clauses[0].question_target_index is not None


def test_preposed_question_word_still_marks_colloquial_question_without_mark() -> None:
    analysis = analyze_structured_utterance("Что ты делаешь")

    assert analysis.clauses[0].clause_kind == "question"


def test_direct_question_after_comma_keeps_its_own_predicate_and_force() -> None:
    analysis = analyze_structured_utterance(
        "Вагонетка мчится налево, откуда она появится на экране сперва?"
    )

    assert len(analysis.clauses) == 2
    premise, question = analysis.clauses
    assert premise.clause_kind == "statement"
    assert analysis.tokens[premise.predicate_index].lemma == "мчаться"
    assert question.clause_kind == "question"
    assert analysis.tokens[question.predicate_index].lemma == "появиться"
    assert analysis.tokens[question.question_target_index].lemma == "откуда"


def test_embedded_interrogative_after_knowledge_predicate_is_not_a_root_question() -> None:
    analysis = analyze_structured_utterance("Я знаю, где она появится.")

    assert len(analysis.clauses) == 2
    assert analysis.clauses[1].clause_kind != "question"
    assert any(link.relation == "ccomp:interrogative" for link in analysis.clause_links)


def test_preposed_condition_does_not_swallow_the_following_time_question() -> None:
    analysis = analyze_structured_utterance(
        "А если машина едет со скоростью 200 км/ч, когда она доедет 654 километра"
    )

    assert tuple(clause.clause_kind for clause in analysis.clauses) == (
        "statement",
        "question",
    )
    assert tuple(
        (
            link.parent_clause_index,
            link.child_clause_index,
            link.relation,
        )
        for link in analysis.clause_links
    ) == ((1, 0, "advcl:condition"),)
    question = analysis.clauses[1]
    assert analysis.tokens[question.question_target_index].lemma == "когда"


def test_preposed_object_and_imperative_objects_have_dependency_edges() -> None:
    preposed = analyze_structured_utterance("А небо ты любишь?")
    command = analyze_structured_utterance("Изучи, что такое небо, и скажи своё мнение")

    frame = preposed.clauses[0]
    assert tuple(preposed.tokens[index].lemma for index in frame.subject_indices) == ("ты",)
    assert tuple(preposed.tokens[index].lemma for index in frame.direct_object_indices) == ("небо",)
    assert {
        (
            preposed.tokens[arc.head_index].lemma,
            preposed.tokens[arc.dependent_index].lemma,
            arc.relation,
        )
        for arc in preposed.dependencies
    } >= {("любить", "ты", "nsubj"), ("любить", "небо", "obj")}

    assert all(
        not frame.subject_indices for frame in command.clauses if frame.clause_kind == "command"
    )
    assert tuple(
        tuple(command.tokens[index].lemma for index in frame.direct_object_indices)
        for frame in command.clauses
    ) == ((), (), ("мнение",))
    assert tuple(
        (
            link.parent_clause_index,
            link.child_clause_index,
            link.relation,
        )
        for link in command.clause_links
    ) == ((0, 1, "ccomp"), (0, 2, "coord:additive"))
    assert tuple(command.tokens[index].lemma for index in command.clauses[1].subject_indices) == (
        "небо",
    )
    assert any(
        command.tokens[arc.head_index].lemma == "мнение"
        and command.tokens[arc.dependent_index].lemma == "свой"
        and arc.relation == "amod"
        for arc in command.dependencies
    )


def test_control_and_modal_infinitives_keep_their_own_objects() -> None:
    assertion = analyze_structured_utterance("Это не мешает иметь мнение")
    modal = analyze_structured_utterance("Ты можешь без тела иметь мнение и позицию")

    assertion_frame = assertion.clauses[0]
    assert tuple(assertion.tokens[index].lemma for index in assertion_frame.subject_indices) == (
        "это",
    )
    assert tuple(
        assertion.tokens[index].lemma for index in assertion_frame.action_complement_indices
    ) == ("иметь",)
    assert {
        (
            assertion.tokens[arc.head_index].lemma,
            assertion.tokens[arc.dependent_index].lemma,
            arc.relation,
        )
        for arc in assertion.dependencies
    } >= {
        ("мешать", "это", "nsubj"),
        ("мешать", "иметь", "xcomp"),
        ("иметь", "мнение", "obj"),
        ("мешать", "не", "advmod:neg"),
    }

    modal_frame = modal.clauses[0]
    assert tuple(modal.tokens[index].lemma for index in modal_frame.action_complement_indices) == (
        "иметь",
    )
    assert tuple(modal.tokens[index].lemma for index in modal_frame.direct_object_indices) == (
        "мнение",
        "позиция",
    )


def test_subordinate_content_marker_is_not_an_embedded_question_target() -> None:
    structure = analyze_structured_utterance("Ой бля, не делай вид, что ты не понял, ахахахах")

    child = structure.clauses[1]
    laughter = next(token for token in structure.tokens if token.surface == "ахахахах")
    marker = next(token for token in structure.tokens if token.surface == "что")

    assert child.clause_kind == "statement"
    assert child.question_target_index is None
    assert marker.semantic_role == "function_word"
    assert laughter.semantic_role == "pragmatic_marker"
    assert laughter.index not in child.direct_object_indices
    assert "amusement" in laughter.pragmatic_functions


def test_subordinate_clauses_work_before_and_after_the_matrix_clause() -> None:
    preposed = analyze_structured_utterance("Если дождь закончится, мы выйдем.")
    postposed = analyze_structured_utterance("Мы остались, потому что шёл дождь.")

    assert any(
        link.parent_clause_index == 1
        and link.child_clause_index == 0
        and link.relation == "advcl:condition"
        for link in preposed.clause_links
    )
    assert preposed.clauses[0].clause_kind == "statement"
    assert any(
        link.parent_clause_index == 0
        and link.child_clause_index == 1
        and link.relation == "advcl:cause"
        for link in postposed.clause_links
    )
    assert postposed.clauses[1].clause_kind == "statement"
    assert postposed.clauses[1].question_target_index is None


def test_alternative_and_contrastive_conjunctions_link_whole_clauses() -> None:
    alternative = analyze_structured_utterance("Ты остаёшься или мы уходим?")
    contrast = analyze_structured_utterance("Я согласился, однако он отказался.")

    assert any(link.relation == "coord:alternative" for link in alternative.clause_links)
    assert any(link.relation == "coord:contrast" for link in contrast.clause_links)


def test_dative_experiencer_and_nominative_stimulus_are_not_swapped() -> None:
    structure = analyze_structured_utterance("Тебе нравится это имя?")
    frame = structure.clauses[0]

    assert tuple(structure.tokens[index].lemma for index in frame.subject_indices) == ("имя",)
    assert tuple(structure.tokens[index].lemma for index in frame.indirect_object_indices) == (
        "ты",
    )
    assert not frame.direct_object_indices


def test_morphological_guess_is_explicitly_quarantined_from_dictionary_certainty() -> None:
    structure = analyze_structured_utterance("Кот мурмякает громко")
    guessed = structure.tokens[1]

    assert guessed.part_of_speech == "verb"
    assert guessed.morphology_known is False
    assert guessed.morphology_source == "guessed"
    assert guessed.confidence <= 0.38
    assert "мурмякает:morphology_guess" in structure.ambiguities


def test_numeric_literal_participates_in_structure_as_a_number() -> None:
    structure = analyze_structured_utterance("Купи 3 яблока")
    numeral = structure.tokens[1]
    frame = structure.clauses[0]

    assert numeral.part_of_speech == "number"
    assert numeral.morphology_source == "numeric_literal"
    assert numeral.morphology_known is True
    assert tuple(structure.tokens[index].lemma for index in frame.direct_object_indices) == (
        "яблоко",
    )


def test_subject_agreement_beats_a_syncretic_object_before_past_verb() -> None:
    structure = analyze_structured_utterance("Ты ножницы покупал?")
    frame = structure.clauses[0]

    assert tuple(structure.tokens[index].lemma for index in frame.subject_indices) == ("ты",)
    assert tuple(structure.tokens[index].lemma for index in frame.direct_object_indices) == (
        "ножницы",
    )


def test_pragmatic_reading_cannot_erase_the_only_finite_predicate() -> None:
    structure = analyze_structured_utterance("а ты будто не знаешь")
    frame = structure.clauses[0]

    assert frame.clause_kind == "statement"
    assert structure.tokens[frame.predicate_index].lemma == "знать"
    assert tuple(structure.tokens[index].lemma for index in frame.subject_indices) == ("ты",)
    assert frame.negated is True
    assert frame.elliptical is False


def test_real_parenthetical_marker_detaches_when_a_proposition_already_exists() -> None:
    structure = analyze_structured_utterance("Я уйду, знаешь")
    frame = structure.clauses[0]

    assert structure.tokens[frame.predicate_index].lemma == "уйти"
    marker = next(token for token in structure.tokens if token.lemma == "знать")
    assert marker.semantic_role == "pragmatic_marker"
