import pytest

from debilgi.language.word_relations import (
    grammatical_tokens,
    observe_word_relations,
    relation_coherence,
    relation_fit,
)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("Я читаю книгу.", ("ты", "читаешь", "книгу", ".")),
        ("Я написала письмо.", ("ты", "написала", "письмо", ".")),
        ("Ты слушаешь меня?", ("я", "слушаю", "тебя", "?")),
        ("Моя машина работает.", ("твоя", "машина", "работает", ".")),
        ("Ты не считаю себя тупым.", ("я", "не", "считаю", "себя", "тупым", ".")),
        ("ногти потеряешь", ("ногти", "потеряю")),
    ],
)
def test_participants_and_agreement_are_transferred_without_replacing_words(source, expected):
    assert grammatical_tokens(source, human_perspective=True) == expected


def test_quoted_speaker_and_reported_clause_are_not_reassigned():
    quoted = grammatical_tokens("Я написал «я грызу ногти».", human_perspective=True)
    assert quoted == ("ты", "написал", "«", "я", "грызу", "ногти", "»", ".")
    reported = grammatical_tokens("Я сказал, что я читаю книгу.", human_perspective=True)
    assert reported == ("ты", "сказал", ",", "что", "я", "читаю", "книгу", ".")


def test_noun_dependency_guess_cannot_replace_an_omitted_addressee(monkeypatch):
    from debilgi.language import syntax_backend
    from debilgi.language.syntax_backend import SyntaxEdge

    monkeypatch.setattr(
        syntax_backend,
        "syntax_edges",
        lambda _: (
            SyntaxEdge(1, 0, "nsubj"),
            SyntaxEdge(-1, 1, "root"),
        ),
    )
    grammatical_tokens.cache_clear()
    try:
        assert grammatical_tokens("ногти потеряешь") == ("ногти", "потеряешь")
    finally:
        grammatical_tokens.cache_clear()


def test_learned_government_prefers_observed_case_and_generalizes_to_another_noun():
    links = observe_word_relations("Ты пишешь другу.")
    assert any(
        link.governor == "писать"
        and link.dependent == "друг"
        and link.relation == "iobj"
        and link.case == "dative"
        for link in links
    )
    exact = relation_fit("Ты пишешь другу.", links)
    generalized = relation_fit("Ты пишешь сестре.", links)
    wrong = relation_fit("Ты пишешь книгу.", links)
    assert exact > generalized > wrong


def test_generated_pronoun_without_a_role_does_not_become_a_second_argument():
    tokens = grammatical_tokens("Ты это не считаешь себя умным.", prune_unbound=True)
    assert tokens == ("ты", "не", "считаешь", "себя", "умным", ".")
    bound = grammatical_tokens("Ты это не считаешь.", prune_unbound=True)
    assert bound == ("ты", "это", "не", "считаешь", ".")


def test_secondary_predicate_is_a_learnable_link_with_its_actual_case():
    links = observe_word_relations("Ты считаешь себя умным.")
    assert any(
        link.governor == "считать"
        and link.dependent == "умный"
        and link.relation == "xcomp"
        and link.case == "instrumental"
        for link in links
    )


def test_known_predicate_cannot_gain_an_unobserved_object_during_recombination():
    learned = (
        *observe_word_relations("Робот скучает без дела."),
        *observe_word_relations("Пёс любит кость."),
    )

    coherent = relation_coherence("Робот скучает без разговора.", learned)
    spliced = relation_coherence("Робот скучает кость.", learned)

    assert coherent.conflicts == 0
    assert coherent.score > 0
    assert spliced.conflicts >= 1
