from debilgi.language.lexical_choice import LexicalOption, choose_lexeme


def test_choice_requires_grounded_meaning():
    assert (
        choose_lexeme(
            "инструмент",
            (LexicalOption("орудие", "орудие", "инструмент"),),
        )
        is None
    )


def test_choice_uses_register_and_avoids_recent_surface():
    choice = choose_lexeme(
        "инструмент",
        (
            LexicalOption(
                "инструмент", "инструмент", "инструмент", evidence=("sense:a",), salience=0.7
            ),
            LexicalOption(
                "орудие",
                "орудие",
                "инструмент",
                evidence=("sense:b",),
                register="rough",
                salience=0.6,
            ),
        ),
        recently_used=("орудие",),
    )
    assert choice is not None
    assert choice.option.lemma == "инструмент"
    assert "shared_grounded_meaning" in choice.reason


def test_unknown_concept_does_not_invent_a_word():
    assert choose_lexeme("квазар", ()) is None
