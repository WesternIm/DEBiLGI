from debilgi.language.addressing import RussianAddressing
from debilgi.language.discourse import classify_utterance, topic_keys
from debilgi.language.semantics import extract_semantic_frames


def test_finds_initial_middle_final_and_unpunctuated_addresses() -> None:
    addressing = RussianAddressing()
    examples = {
        "Маша, открой дверь": ("открой дверь", "initial", "маша"),
        "Скажи, дорогой друг, где ключ": ("скажи, где ключ", "parenthetical", "друг"),
        "Узнай, умник": ("узнай", "final", "умник"),
        "Узнай умник": ("узнай", "final_unpunctuated", "умник"),
        "Умник узнай что такое мороженое": (
            "узнай что такое мороженое",
            "initial_unpunctuated",
            "умник",
        ),
        "Скажи умник где ключ": ("скажи где ключ", "medial_unpunctuated", "умник"),
        "Дорогой друг помоги мне": ("помоги мне", "initial_unpunctuated", "друг"),
        "Маша ты знаешь ответ": ("ты знаешь ответ", "initial_unpunctuated", "маша"),
        "Ну ребята слушайте внимательно": (
            "ну слушайте внимательно",
            "initial_unpunctuated",
            "ребята",
        ),
    }

    for phrase, (content, position, lemma) in examples.items():
        analysis = addressing.analyze(phrase)
        assert analysis.content_text == content
        assert analysis.addresses[0].position == position
        assert lemma in analysis.address_terms


def test_does_not_confuse_address_with_subject_object_or_quoted_term() -> None:
    addressing = RussianAddressing()

    for phrase in (
        "Умник ответил",
        "Ты умник",
        "Найди умника",
        "Позови Машу",
        "Что значит «умник»?",
        "Узнай значение слова умник",
        "Ты умник расскажи",
        "Девушка читает книгу",
    ):
        analysis = addressing.analyze(phrase)
        assert not analysis.addresses
        assert analysis.content_text.casefold() == phrase.casefold()


def test_unpunctuated_address_has_lower_confidence_than_comma_marked_one() -> None:
    addressing = RussianAddressing()

    punctuated = addressing.analyze("Скажи, умник, где ключ").addresses[0]
    unpunctuated = addressing.analyze("Скажи умник где ключ").addresses[0]

    assert punctuated.confidence > unpunctuated.confidence


def test_address_is_not_a_topic_or_semantic_actor() -> None:
    phrase = "Умник, девушка открыла дверь."

    assert "умник" not in topic_keys(phrase)
    frames = extract_semantic_frames(phrase)
    assert frames
    assert frames[0].actor == "девушка"


def test_address_before_question_or_command_does_not_mask_speech_act() -> None:
    assert classify_utterance("Маша, где ключ?") == "question"
    assert classify_utterance("Умник, расскажи про мороженое") == "command"
    assert classify_utterance("ладно, иди отдыхай") == "command"
    assert classify_utterance("ну отдохни") == "command"
    assert classify_utterance("мой компьютер решил") == "assertion"
