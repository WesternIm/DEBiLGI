from debilgi.language.character_markov import (
    WORD_END,
    CharacterMarkovConfig,
    CharacterMarkovModel,
)


def test_character_model_trains_words_with_boundaries_not_joined_text() -> None:
    model = CharacterMarkovModel(CharacterMarkovConfig(min_order=2, max_order=4))

    assert model.train(("верёвка", "discord-слово")) == 2

    assert model.plausibility("верёвка") > 0.5
    assert model.plausibility("верёвкаdiscord-слово") < model.plausibility("верёвка")
    assert any(WORD_END in options for options in model.transitions.values())


def test_character_model_uses_backoff_and_has_bounded_deterministic_generation() -> None:
    config = CharacterMarkovConfig(
        min_order=2,
        max_order=5,
        max_word_length=12,
        generation_attempts=40,
    )
    model = CharacterMarkovModel(config)
    model.train(("стромба", "стромбит", "бомбаст", "бомбить", "ромбик"))
    model.transitions = {
        prefix: options for prefix, options in model.transitions.items() if len(prefix) <= 3
    }

    first = model.generate(seed="fixed")
    second = model.generate(seed="fixed")

    assert first == second
    assert first is None or 3 <= len(first) <= 12


def test_character_model_rejects_runaway_repetition() -> None:
    model = CharacterMarkovModel(
        CharacterMarkovConfig(
            min_order=2,
            max_order=3,
            max_word_length=8,
            max_repeated_characters=2,
        )
    )
    model.train(("аааа", "бааа", "вааа"))

    candidate = model.generate(seed="repeat")

    assert candidate is None or "ааа" not in candidate
