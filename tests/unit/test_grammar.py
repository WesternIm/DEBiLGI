from debilgi.language.grammar import SoftGrammar, role


def test_soft_grammar_learns_roles_and_prefers_complete_sequences() -> None:
    grammar = SoftGrammar.learn(
        (
            ("я", "смотрит", "на", "дракона", "."),
            ("он", "смотрит", "на", "город", "."),
            ("она", "видит", "странные", "огни", "."),
        )
    )

    assert role("он") == "pronoun"
    assert role("смотрит") == "verb"
    assert role("на") == "preposition"
    assert role("ответ") == "content"
    assert role("отвечающую") == "participle"
    assert grammar.score(("он", "смотрит", "на", "город", ".")) > grammar.score(
        ("на", "и", "потому")
    )
