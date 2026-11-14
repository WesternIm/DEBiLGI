from dataclasses import replace
from types import SimpleNamespace

import pytest

from debilgi.cognition.models import ActionType
from debilgi.language.arbitration import ArbitrationProfile, ResponseArbitrator, ResponseCandidate
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.planning import ReplyContext, StructuredThought, UtterancePlan
from debilgi.language.tokenizer import tokenize
from debilgi.language.word_relations import WordRelation


def plan(prompt, corpus=(), ngrams=()):
    return UtterancePlan(
        "brief_statement",
        72,
        StructuredThought(ActionType.SHORT_REPLY, 1, "respond", 0.3, ("current",)),
        ReplyContext(prompt=prompt, seed="test", corpus_utterances=corpus, memory_ngrams=ngrams),
    )


def test_statistical_search_uses_observed_vocabulary_and_changes_with_database():
    generator = MemoryPhraseGenerator()
    first_corpus = (
        ("one", "Кот любит рыбу."),
        ("two", "Кот видит мышь."),
        ("three", "Пёс любит кость."),
        ("four", "Пёс видит дом."),
    )
    second_corpus = (
        ("one", "Кот любит рыбу."),
        ("two", "Кот видит мышь."),
        ("three", "Пёс любит молоко."),
        ("four", "Пёс видит сад."),
    )
    first = generator.generate_explorations(plan("кот", first_corpus))
    second = generator.generate_explorations(plan("кот", second_corpus))
    assert first and second
    assert {item.text for item in first} != {item.text for item in second}
    first_sources = {text for _, text in first_corpus}
    second_sources = {text for _, text in second_corpus}
    assert all(item.text not in first_sources for item in first)
    assert all(item.text not in second_sources for item in second)
    vocabulary = set(tokenize(" ".join(text for _, text in first_corpus) + " кот"))
    assert all(set(tokenize(item.text)) <= vocabulary for item in first)
    assert first == generator.generate_explorations(plan("кот", first_corpus))


def test_stored_transitions_work_without_raw_messages_or_a_complete_prompt():
    grams = (
        (("\0start", "кот"), 1),
        (("кот", "любит"), 1),
        (("любит", "рыбу"), 1),
        (("рыбу", "\0end"), 1),
    )
    surfaces = MemoryPhraseGenerator().generate_explorations(plan("😂", ngrams=grams))
    assert surfaces and any("рыбу" in item.text for item in surfaces)
    assert all("\0" not in item.text for item in surfaces)


def test_empty_database_constructs_an_inquiry_using_the_observed_symbols():
    surfaces = MemoryPhraseGenerator().generate_explorations(plan("зюзюбрик"))
    assert surfaces[0].text == "?"


def test_unresolved_turn_uses_a_learned_negated_cognitive_relation() -> None:
    original = plan("қотақ жепсің")
    context = replace(
        original.context,
        dialogue_situation=SimpleNamespace(
            active_task=SimpleNamespace(goal="respond_relevantly")
        ),
        word_relations=(WordRelation("знать", "я", "nsubj", "pronoun", "nominative", 4),),
    )

    surfaces = MemoryPhraseGenerator().generate_explorations(
        replace(original, context=context)
    )

    assert surfaces[0].text == "Я не знаю."
    assert surfaces[0].unresolved_reaction is True
    assert surfaces[0].cognitive_predicate == "знать"


def test_unresolved_turn_never_uses_a_generic_communicative_action() -> None:
    original = plan("қотақ жепсің")
    context = replace(
        original.context,
        dialogue_situation=SimpleNamespace(
            active_task=SimpleNamespace(goal="respond_relevantly")
        ),
        word_relations=(
            WordRelation("отвечать", "я", "nsubj", "pronoun", "nominative", 20),
        ),
    )

    surfaces = MemoryPhraseGenerator().generate_explorations(
        replace(original, context=context)
    )

    assert surfaces[0].text == "?"
    assert "отвеч" not in surfaces[0].text.casefold()


@pytest.mark.parametrize(
    "prompt,corpus",
    [
        ("кот", (("old", "Кот."),)),
        ("Кот спит", (("old", "Кот спит."),)),
        ("Я думаю", (("old", "Я думаю."),)),
        ("ножницы", (("old", "Ножницы режут бумагу."),)),
    ],
)
def test_input_and_its_inflected_perspective_are_never_returned_as_new_text(prompt, corpus):
    from debilgi.language.copying import is_input_copy

    generator = MemoryPhraseGenerator()
    assert not any(
        is_input_copy(item.text, prompt)
        for item in generator.generate_explorations(plan(prompt, corpus))
    )


def test_arbitration_cannot_recover_an_echo_by_ignoring_its_rejection():
    candidate = ResponseCandidate.create(
        event_id="test",
        source="learned_distribution",
        renderer="test",
        text="«Кот»?",
        speech_act="brief_statement",
        source_confidence=0.2,
        metadata={"generation_score": 100.0, "exploratory": True},
    )
    decision = ResponseArbitrator().decide_exploration(
        event_id="test",
        prompt="кот",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is None


def test_exploration_cannot_ignore_an_answer_obligation_rejection():
    candidate = ResponseCandidate.create(
        event_id="answer-task",
        source="learned_distribution",
        renderer="observed_word_transitions",
        text="Как это я машиной управлять могу?",
        speech_act="brief_statement",
        source_confidence=0.25,
        metadata={
            "generation_score": 100.0,
            "exploratory": True,
            "required_response_act": "answer",
            "dialogue_task_required": False,
        },
    )

    decision = ResponseArbitrator().decide_exploration(
        event_id="answer-task",
        prompt="ауу",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )

    assert decision.winner is None
    assert "question_does_not_fulfill_answer_or_repair" in (
        decision.scores[0].rejection_reasons
    )


@pytest.mark.parametrize(
    "prompt,reply",
    [
        ("Я думаю", "Ты думаешь?"),
        (
            "Если ты не знаешь, чего хочешь, как решаешь, что делать дальше?",
            "Но ты не знаешь, чего хочешь, как решаешь, что делать дальше?",
        ),
        ("Я ногти грызу", "Ты грызёшь ногти."),
        ("кот", "«кот»?"),
        ("Я? Я не считаю себя тупым", "Ну всё, но ты не считаешь себя тупым."),
        ("Так почему не считаешь себя тупым?", "Ты не считаешь себя тупым."),
        ("Мне плевать на твои идеологии", "Плевать на мои идеологии."),
    ],
)
def test_echo_guard_catches_quotes_reordering_and_a_cosmetic_prefix(prompt, reply):
    from debilgi.language.copying import is_input_copy

    assert is_input_copy(reply, prompt)


def test_new_relation_is_not_confused_with_repetition_of_the_topic():
    from debilgi.language.copying import is_input_copy

    assert not is_input_copy("Почему ты грызёшь ногти?", "Я грызу ногти.")
    assert not is_input_copy("Ножницы режут бумагу.", "ножницы")


def test_short_excerpt_from_a_long_memory_is_still_a_source_copy():
    source = tokenize(
        "Первый том написан в тюрьме Ландсберг после неудачного пивного путча, "
        "а второй том был опубликован позднее."
    )
    excerpt = tokenize(
        "Том написан в тюрьме Ландсберг после неудачного пивного путча."
    )

    assert MemoryPhraseGenerator._is_source_copy(excerpt, source)


def test_generation_rejects_a_new_dependency_role_for_a_known_predicate():
    from debilgi.language.word_relations import observe_word_relations

    original = plan(
        "Почему робот скучает?",
        (
            ("robot", "Робот скучает без дела."),
            ("dog", "Пёс любит кость."),
            ("person", "Человек скучает без разговора."),
        ),
    )
    relations = tuple(
        relation
        for _, text in original.context.corpus_utterances
        for relation in observe_word_relations(text)
    )
    constrained = replace(
        original,
        context=replace(original.context, word_relations=relations),
    )

    surfaces = MemoryPhraseGenerator().generate_explorations(constrained)

    assert all(item.text != "Робот скучает кость." for item in surfaces)
    assert all(item.relation_conflicts == 0 for item in surfaces)


@pytest.mark.parametrize("options", [("да", "нет"), ("чай", "кофе"), ("красный", "синий")])
def test_experimental_choice_stays_in_any_observed_answer_domain(options):
    original = plan("Мне тебя выключить?", (("old", "Что думаю о ножницах?"),))
    constrained = replace(original, context=replace(original.context, response_options=options))
    result = MemoryPhraseGenerator().generate_explorations(constrained)
    assert len(result) == 1
    assert result[0].text.casefold() in options
    assert result == MemoryPhraseGenerator().generate_explorations(constrained)


def test_provisional_option_rank_changes_with_connected_human_data():
    original = plan("Что пьёшь?", (("source", "Пью чай."),))
    context = replace(original.context, response_options=("чай", "кофе"))
    generator = MemoryPhraseGenerator()
    assert generator.generate_explorations(replace(original, context=context))[0].text == "Чай"
    context = replace(context, corpus_utterances=(("other", "Пью кофе."),))
    assert generator.generate_explorations(replace(original, context=context))[0].text == "Кофе"


def test_exploration_keeps_action_anchor_across_aspect_without_preferring_empty_function_words():
    original = plan("Мне тебя выключать?", (("action", "Я выключу тебя."),))
    context = replace(
        original.context,
        generation_ngrams=(
            (("\0start", "как"), 999),
            (("как", "это"), 999),
            (("это", "не"), 999),
            (("не", "?"), 999),
            (("?", "\0end"), 999),
        ),
    )
    result = MemoryPhraseGenerator().generate_explorations(replace(original, context=context))
    assert result and "выключ" in result[0].text.casefold()


def test_generation_does_not_mix_summaries_or_an_unrelated_archive_into_current_speech():
    original = plan("там грязь", (("dirt", "Там грязь."), ("insult", "Ты тупой потому что.")))
    polluted = replace(
        original,
        context=replace(
            original.context,
            memory_messages=("Затем система отвечает на вопрос. Содержание команды: я отвечаю.",),
            knowledge_facts=("Система выбирает содержание команды.",),
            memory_ngrams=((("\0start", "ты", "тупой"), 999), (("тупой", "потому"), 999)),
        ),
    )
    surfaces = MemoryPhraseGenerator().generate_explorations(polluted)
    assert surfaces
    assert all(
        not {"затем", "система", "команды", "тупой", "потому"}.intersection(tokenize(item.text))
        for item in surfaces
    )


def test_short_sentence_stops_at_its_first_boundary_without_appending_another_thought():
    corpus = (
        ("one", "Кот любит рыбу. Затем кот летает."),
        ("two", "Кот видит мышь."),
        ("three", "Пёс любит кость."),
        ("four", "Пёс видит дом."),
    )
    surfaces = MemoryPhraseGenerator().generate_explorations(
        plan("кот", corpus),
    )
    assert surfaces
    source_sentences = {"Кот любит рыбу.", "Затем кот летает.", "Кот видит мышь.",
                        "Пёс любит кость.", "Пёс видит дом."}
    assert all(item.text not in source_sentences for item in surfaces)
    assert all(
        sum(token in {".", "!", "?", "…"} for token in tokenize(item.text)) == 1
        for item in surfaces
    )


def test_one_stored_question_is_not_revoiced_as_a_ready_reply():
    surfaces = MemoryPhraseGenerator().generate_explorations(
        plan("ногти", (("source", "Почему я грызу ногти?"),)),
    )
    assert all("я грызу" not in item.text.casefold() for item in surfaces)
    assert all("ты грызёшь" not in item.text.casefold() for item in surfaces)


def test_scoped_reader_transitions_work_after_raw_text_is_gone():
    grams = (
        (("\0start", "ты"), 1),
        (("ты", "читаешь"), 1),
        (("читаешь", "книгу"), 1),
        (("книгу", "\0end"), 1),
    )
    original = plan("😂")
    original = replace(original, context=replace(original.context, generation_ngrams=grams))
    surfaces = MemoryPhraseGenerator().generate_explorations(original)
    assert surfaces and surfaces[0].text == "Ты читаешь книгу."


def test_redacted_corpus_keeps_the_lexical_connection_to_an_inflected_topic():
    grams = (
        (("\0start", "ты"), 1),
        (("ты", "читаешь"), 1),
        (("читаешь", "книгу"), 1),
        (("книгу", "\0end"), 1),
    )
    original = plan("книга")
    original = replace(original, context=replace(original.context, generation_ngrams=grams))
    surfaces = MemoryPhraseGenerator().generate_explorations(original)
    assert any(item.text == "Ты читаешь книгу." for item in surfaces)


def test_omitted_first_person_is_still_a_participant_shift_when_question_is_about_the_user():
    original = plan(
        "Почему я тупой?",
        (
            ("other", "Не считаешь себя тупым."),
            ("same", "Я не считаю себя тупым."),
        ),
    )
    surfaces = MemoryPhraseGenerator().generate_explorations(original)
    assert surfaces
    assert "считаю" not in surfaces[0].text.casefold()


def test_quality_threshold_blocks_unbound_exploration_instead_of_replaying_it():
    candidate = ResponseCandidate.create(
        event_id="one",
        source="learned_distribution",
        renderer="observed_word_transitions",
        text="Кот рыба нравится.",
        speech_act="brief_statement",
        source_confidence=0.25,
        metadata={"generation_score": 1.0, "exploratory": True},
    )
    arbitrator = ResponseArbitrator(minimum_score=0.99)
    decision = arbitrator.decide_exploration(
        event_id="one",
        prompt="кот",
        focus_concepts=(),
        candidates=(candidate,),
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )
    assert decision.winner is None
    assert decision.selection_reason == "below_minimum_score"
    assert decision.scores[0].admission_score < 0.99


def test_context_bound_exploration_prefers_semantic_score_over_corpus_frequency():
    candidates = tuple(
        ResponseCandidate.create(
            event_id="ranked",
            source="learned_distribution",
            renderer="observed_word_transitions",
            text=text,
            speech_act="brief_statement",
            source_confidence=confidence,
            metadata={
                "generation_score": generation_score,
                "dialogue_task_progress": progress,
                "dialogue_task_required": True,
                "context_relevance": progress,
                "exploratory": True,
            },
        )
        for text, confidence, generation_score, progress in (
            ("Я продолжаю разговор.", 0.80, 1.0, 0.80),
            ("Как меня зовут?", 0.05, 1000.0, 0.45),
        )
    )
    decision = ResponseArbitrator(minimum_score=10.0).decide_exploration(
        event_id="ranked",
        prompt="и че",
        focus_concepts=("разговор",),
        candidates=candidates,
        recent_replies=(),
        profile=ArbitrationProfile.baseline(1),
    )

    assert decision.winner is not None
    assert decision.winner.text == "Я продолжаю разговор."
    assert decision.selection_reason == "context_bound_learned_composition"
