import pytest

from debilgi.language.dialogue_state import (
    DialogueContextWindow,
    analyze_dialogue_state,
    realize_reference_clarification,
)


def test_classifies_continuation_shift_and_return() -> None:
    continuation = analyze_dialogue_state(
        "А какие команды играют?",
        ("Мы обсуждаем футбольные команды.",),
    )
    shift = analyze_dialogue_state(
        "Кстати, теперь о книгах.",
        ("Мы обсуждаем футбольные команды.",),
    )
    returned = analyze_dialogue_state(
        "А футбол всё ещё нравится?",
        (
            "Мы обсуждали футбол.",
            "Потом перешли к книгам.",
        ),
    )

    assert continuation.topic.transition == "continuation"
    assert shift.topic.transition == "shift"
    assert returned.topic.transition == "return"


def test_resolves_pronoun_by_gender_and_recent_grammatical_roles() -> None:
    analysis = analyze_dialogue_state(
        "Где он живёт?",
        ("Лиса увидела волка.",),
    )

    reference = analysis.references[0]
    assert reference.chosen is not None
    assert reference.chosen.lemma == "волк"
    assert "волк" in analysis.added_context_terms
    assert analysis.retrieval_query.endswith("волк")


def test_does_not_guess_between_equally_compatible_antecedents() -> None:
    analysis = analyze_dialogue_state(
        "Он ушёл?",
        ("Дракон встретил рыцаря.",),
    )

    reference = analysis.references[0]
    assert reference.chosen is None
    assert reference.reason == "ambiguous_antecedent"
    assert {item.lemma for item in reference.alternatives[:2]} == {"дракон", "рыцарь"}
    assert realize_reference_clarification(analysis) == (
        "Под «он» имеется в виду «дракон» или «рыцарь»?"
    )


def test_recovers_unique_implicit_object_without_rewriting_prompt() -> None:
    analysis = analyze_dialogue_state(
        "Она взяла?",
        ("Лиса нашла ключ.",),
    )

    chosen = {
        item.kind: item.chosen.lemma if item.chosen is not None else None
        for item in analysis.references
    }
    assert chosen == {
        "explicit_pronoun": "лиса",
        "implicit_direct_object": "ключ",
    }
    assert analysis.retrieval_query == "Она взяла? лиса ключ"
    assert analysis.resolved_bindings == (
        ("subject", "лиса"),
        ("direct_object", "ключ"),
    )


def test_short_elliptical_question_inherits_only_the_active_topic() -> None:
    analysis = analyze_dialogue_state(
        "А почему?",
        ("Дракон покинул замок.",),
    )

    assert analysis.topic.transition == "continuation"
    assert analysis.topic.reason == "short_elliptical_follow_up"
    assert set(analysis.added_context_terms) <= {"дракон", "покинуть", "замок"}


def test_bare_new_noun_is_not_forced_into_previous_topic() -> None:
    analysis = analyze_dialogue_state(
        "А книги?",
        ("Мы обсуждали футбол.",),
    )

    assert analysis.topic.transition == "shift"
    assert not analysis.added_context_terms


def test_answer_failure_returns_to_the_last_real_question() -> None:
    question = "Насколько ты считаешь себя осознанным?"
    analysis = analyze_dialogue_state(
        "Ты не ответил на вопрос",
        (question,),
    )

    assert analysis.repair_target == question
    assert analysis.retrieval_query == question
    assert "осознанный" in analysis.added_context_terms


@pytest.mark.parametrize(
    "repair_message",
    (
        "Я тебя спрашиваю",
        "Блять, але, я тебя спрашиваю",
        "Я уже спросил",
    ),
)
def test_meta_question_reference_reactivates_the_previous_question(
    repair_message: str,
) -> None:
    question = "Так есть ли у тебя ногти?"
    analysis = analyze_dialogue_state(
        repair_message,
        (question, "Ноготь — роговая пластинка."),
    )

    assert analysis.repair_target == question
    assert analysis.retrieval_query == question
    assert "ноготь" in analysis.added_context_terms


def test_retry_with_a_full_rephrased_question_does_not_jump_back() -> None:
    analysis = analyze_dialogue_state(
        "Ответь нормально: насколько ты считаешь себя осознанным?",
        ("Ты умный?",),
    )

    assert analysis.repair_target is None


def test_complete_short_question_does_not_inherit_a_bad_previous_object() -> None:
    analysis = analyze_dialogue_state(
        "А ты умеешь летать?",
        ("Я умею смысл.",),
    )

    assert analysis.topic.transition == "continuation"
    assert "смысл" not in analysis.added_context_terms
    assert analysis.retrieval_query == "А ты умеешь летать?"


def test_repeated_mentions_support_one_referent_instead_of_faking_ambiguity() -> None:
    analysis = analyze_dialogue_state(
        "А почему не любишь?",
        (
            "А небо ты любишь?",
            "Почему ты не любишь небо?",
        ),
    )

    reference = next(
        item for item in analysis.references if item.kind == "implicit_direct_object"
    )
    assert reference.chosen is not None
    assert reference.chosen.lemma == "небо"
    assert tuple(item.lemma for item in reference.alternatives) == ("небо",)
    assert analysis.resolved_bindings == (("direct_object", "небо"),)


def test_reply_chain_anchor_restores_iq_for_short_hypothesis_follow_up() -> None:
    window = DialogueContextWindow.build(
        (
            "Сколько IQ у тебя?",
            "Я не могу назвать точное число.",
        ),
        reply_chain=(
            "Сколько IQ у тебя?",
            "Я не могу назвать точное число.",
        ),
        semantic_anchors=("Сколько IQ у тебя?",),
    )

    analysis = analyze_dialogue_state("А ты предположи", context_window=window)

    assert analysis.topic.transition == "continuation"
    assert analysis.topic.reason == "reply_chain_semantic_anchor"
    assert analysis.topic.anchor_message == "Сколько IQ у тебя?"
    assert analysis.resolved_bindings == (("direct_object", "iq"),)
    assert analysis.added_context_terms == ("iq",)
    assert analysis.retrieval_query == "А ты предположи iq"


def test_context_window_is_bounded_and_prioritizes_the_direct_chain() -> None:
    window = DialogueContextWindow.build(
        tuple(f"обычная реплика {index}" for index in range(20)),
        reply_chain=("Корневой вопрос?", "Ответ бота."),
        semantic_anchors=("Корневой вопрос?",),
        max_turns=6,
    )

    assert len(window.messages) == 6
    assert window.messages[-2:] == ("Корневой вопрос?", "Ответ бота.")
    assert window.semantic_anchors == ("Корневой вопрос?",)
