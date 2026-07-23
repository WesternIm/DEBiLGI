from debilgi.language.content import content_lemmas, content_tokens
from debilgi.language.discourse import topic_keys


def test_content_terms_remove_pragmatic_and_grammatical_scaffolding() -> None:
    text = "Хмм, ну я же думаю о новом ключе"

    assert content_tokens(text) == ("думаю", "новом", "ключе")
    assert content_lemmas(text) == ("думать", "новый", "ключ")
    assert topic_keys(text) == ("думать", "новый", "ключ")


def test_inflected_pronouns_and_bare_stances_do_not_become_topics() -> None:
    assert content_lemmas("чего ты хочешь от него") == ("хотеть",)
    assert topic_keys("Ну да нет") == ()
    assert topic_keys("Хмм") == ()


def test_question_word_is_not_topic_but_predicate_is() -> None:
    assert topic_keys("Почему ты так думаешь") == ("думать",)


def test_colloquial_particles_do_not_become_graph_topics() -> None:
    assert content_lemmas("типо ну вот самый добрый человек") == (
        "добрый",
        "человек",
    )


def test_content_lemmas_correct_safe_typo_but_keep_paronyms_separate() -> None:
    assert content_lemmas("положи таоелку на стол") == (
        "положить",
        "тарелка",
        "стол",
    )
    assert content_lemmas("варенье и варение") == ("варенье", "варение")


def test_expressive_chat_forms_do_not_pollute_topics() -> None:
    assert content_lemmas("бла-бла-бла про крууутое зеркало") == (
        "крутой",
        "зеркало",
    )


def test_short_abbreviation_remains_a_content_concept() -> None:
    assert content_tokens("Что такое ИИ?") == ("ии",)
    assert content_lemmas("Что такое ИИ?") == ("ии",)
    assert content_tokens("т.е. объясни иначе") == ("т.е.", "объясни", "иначе")
    assert content_lemmas("т.е. объясни иначе") == ("те", "объяснить", "иначе")
