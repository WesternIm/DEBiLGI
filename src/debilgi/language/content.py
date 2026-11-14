"""Shared extraction of topical surface words and normalized content lemmas."""

from __future__ import annotations

from collections.abc import Iterable

from debilgi.language.abbreviations import (
    abbreviation_mentions,
    is_abbreviation_candidate,
    normalize_abbreviation,
)
from debilgi.language.function_words import is_service_word
from debilgi.language.grammar import role
from debilgi.language.morphology import DEFAULT_MORPHOLOGY
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.tokenizer import normalize_chat_token, tokenize

_CONTENT_ROLES = frozenset(
    {
        "content",
        "adjective",
        "verb",
        "number",
        "adverb",
        "predicate",
        "participle",
        "gerund",
    }
)
_NON_TOPICAL_LEMMAS = frozenset(
    {
        "да",
        "есть",
        "нет",
        "весь",
        "быть",
        "мочь",
        "мой",
        "наш",
        "очень",
        "где",
        "зачем",
        "как",
        "какой",
        "когда",
        "кто",
        "куда",
        "откуда",
        "почему",
        "просто",
        "свой",
        "так",
        "такой",
        "этот",
        "тот",
        "твой",
        "там",
        "тогда",
        "тут",
        "уже",
        "что",
        "ваш",
        "ещё",
    }
)
_GRAMMATICAL_PARTS = frozenset(
    {"pronoun", "preposition", "conjunction", "particle", "interjection"}
)


def content_tokens(value: str | Iterable[str]) -> tuple[str, ...]:
    """Return topical surface words, excluding pragmatic and grammatical scaffolding."""

    return tuple(item[0] for item in _content_items(value))


def content_lemmas(value: str | Iterable[str]) -> tuple[str, ...]:
    """Return unique contextual lemmas used consistently by every memory layer."""

    return tuple(dict.fromkeys(item[1] for item in _content_items(value)))


def _content_items(value: str | Iterable[str]) -> tuple[tuple[str, str], ...]:
    """Keep surface forms and context-selected lemmas aligned in one parse."""

    if isinstance(value, str):
        tokens = tokenize(value)
        pragmatic_text = value
    else:
        tokens = tuple(value)
        pragmatic_text = " ".join(tokens)
    cue_words = {
        normalize_chat_token(word.casefold().replace("ё", "е")).canonical
        for cue in analyze_pragmatics(pragmatic_text).cues
        for word in tokenize(cue.surface)
        if word[:1].isalnum()
    }
    abbreviations = frozenset(abbreviation_mentions(pragmatic_text))
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(pragmatic_text)
    readings = sentence.readings
    if sentence.tokens != tokens:
        readings = tuple(DEFAULT_MORPHOLOGY.best(token) for token in tokens)
    selected: list[tuple[str, str]] = []
    for token, reading in zip(tokens, readings, strict=True):
        raw_normalized = token.casefold().replace("ё", "е")
        abbreviation = is_abbreviation_candidate(token)
        normalized = (
            normalize_abbreviation(token)
            if abbreviation
            else reading.corrected_form
            if reading is not None and reading.corrected_form
            else normalize_chat_token(raw_normalized).canonical
        )
        if (
            not normalized[:1].isalnum()
            or len(normalized) <= 2
            and normalized not in abbreviations
        ):
            continue
        if normalized in cue_words:
            continue
        if not abbreviation and is_service_word(
            reading.lemma if reading is not None else normalized,
            part_of_speech=reading.part_of_speech if reading is not None else None,
            question_context="?" in pragmatic_text,
        ):
            continue
        if (
            not abbreviation
            and reading is not None
            and reading.part_of_speech in _GRAMMATICAL_PARTS
        ):
            continue
        if (
            not abbreviation
            and reading is None
            and role(normalized) not in _CONTENT_ROLES
        ):
            continue
        lemma = (
            normalized
            if abbreviation
            else reading.lemma
            if reading is not None
            else DEFAULT_MORPHOLOGY.lemma(normalized)
        )
        if lemma in _NON_TOPICAL_LEMMAS:
            continue
        selected.append((token if abbreviation and "." in token else normalized, lemma))
    return tuple(dict.fromkeys(selected))
