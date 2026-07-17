import pytest

from debilgi.cognition.models import ActionType
from debilgi.language.memory_based import MemoryPhraseGenerator
from debilgi.language.ngrams import ngram_frequencies, token_frequencies
from debilgi.language.planning import ReplyContext, StructuredThought, utterance_plan
from debilgi.language.reply_patterns import build_reply_pattern
from debilgi.language.tokenizer import normalize_discord_markup, tokenize


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


def test_semantic_fact_can_ground_generated_language() -> None:
    thought = StructuredThought(ActionType.SHORT_REPLY, None, "answer", 0.8, ())
    context = ReplyContext(
        prompt="что такое дракон",
        knowledge_facts=("Дракон — мифическое существо с крыльями",),
        seed="knowledge-test",
    )
    plan = utterance_plan(thought, context=context)

    assert plan is not None
    response = MemoryPhraseGenerator().generate(plan)
    assert "мифическое существо" in response


def test_recalled_fact_beats_unrelated_markov_continuation() -> None:
    thought = StructuredThought(ActionType.SHORT_REPLY, None, "recall", 0.8, ())
    context = ReplyContext(
        prompt="где находится ключ",
        memory_messages=(
            "сервер шумит и кнопка ломает старую программу",
            "дракон летит над городом и смеётся",
        ),
        knowledge_facts=("ворон достал ключ из ящика",),
        seed="semantic-recall-test",
    )
    plan = utterance_plan(thought, context=context)

    assert plan is not None
    assert MemoryPhraseGenerator().generate(plan) == "ворон достал ключ из ящика"


def test_question_intent_prefers_learned_question_openings() -> None:
    thought = StructuredThought(ActionType.ASK_QUESTION, None, "clarify", 0.7, ())
    context = ReplyContext(
        prompt="медный ключ",
        memory_messages=(
            "где сейчас лежит медный ключ?",
            "почему ворон забрал ключ?",
            "обычный сервер работает спокойно.",
        ),
        seed="question-grammar-test",
    )
    plan = utterance_plan(thought, context=context)

    assert plan is not None
    response = MemoryPhraseGenerator().generate(plan)
    assert response.endswith("?")
    assert response.split()[0] in {"где", "почему"}
