import pytest

from debilgi.language.ngrams import ngram_frequencies, token_frequencies
from debilgi.language.reply_patterns import build_reply_pattern
from debilgi.language.tokenizer import (
    canonical_chat_tokens,
    normalize_chat_token,
    normalize_discord_markup,
    tokenize,
)


def test_tokenizer_anonymizes_discord_ids_and_urls() -> None:
    text = "Привет <@123> <@&456> <#789> https://example.com <:LOL:999>"

    tokens = tokenize(text)

    assert tokens == (
        "привет",
        "<user>",
        "<role>",
        "<channel>",
        "<url>",
        "<emoji:lol>",
    )
    assert "123" not in " ".join(tokens)
    assert "999" not in " ".join(tokens)


def test_tokenizer_preserves_local_words_and_punctuation() -> None:
    assert tokenize("Бля, я подумал 🤔") == ("бля", ",", "я", "подумал", "🤔")


def test_normalization_does_not_touch_plain_text() -> None:
    assert normalize_discord_markup("обычный текст") == "обычный текст"


def test_chat_normalization_separates_word_from_expressive_spelling() -> None:
    greeting = normalize_chat_token("привеееет")
    repetition = normalize_chat_token("бла-бла-бла")
    legitimate_double = normalize_chat_token(
        "класссс",
        is_known=lambda word: word in {"клас", "класс"},
    )

    assert (greeting.canonical, greeting.kind) == ("привет", "elongation")
    assert (repetition.canonical, repetition.kind, repetition.repetitions) == (
        "бла",
        "reduplication",
        3,
    )
    assert legitimate_double.canonical == "класс"
    assert canonical_chat_tokens("дааа, нееет") == ("да", ",", "нет")


def test_token_and_ngram_frequencies_are_deterministic() -> None:
    tokens = ("а", "б", "а")

    assert token_frequencies(tokens) == {"а": 2, "б": 1}
    assert ngram_frequencies(tokens, 2) == {("а", "б"): 1, ("б", "а"): 1}
    assert ngram_frequencies(tokens, 3) == {("а", "б", "а"): 1}


def test_only_bigrams_and_trigrams_are_allowed() -> None:
    with pytest.raises(ValueError, match="bigrams"):
        ngram_frequencies(("a", "b"), 1)


def test_reply_pattern_uses_prompt_tail_and_response_head() -> None:
    pattern = build_reply_pattern(
        ("раз", "два", "три", "четыре"),
        ("ответ", "сюда", "пять", "шесть"),
    )

    assert pattern == (("два", "три", "четыре"), ("ответ", "сюда", "пять"))
