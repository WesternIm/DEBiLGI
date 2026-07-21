"""Shared extraction of topical surface words and normalized content lemmas."""

from __future__ import annotations

import re
from collections.abc import Iterable

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
        "так",
        "такой",
        "там",
        "тогда",
        "тут",
        "уже",
        "что",
        "ещё",
    }
)
_GRAMMATICAL_PARTS = frozenset(
    {"pronoun", "preposition", "conjunction", "particle", "interjection"}
)
_ACRONYM_RE = re.compile(r"(?<![\w-])[A-ZА-ЯЁ]{2,10}(?![\w-])")


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
        acronym_lemmas = {
            match.group(0).casefold().replace("ё", "е")
            for match in _ACRONYM_RE.finditer(value)
        }
    else:
        tokens = tuple(value)
        pragmatic_text = " ".join(tokens)
        acronym_lemmas = set()
    cue_words = {
        normalize_chat_token(word.casefold().replace("ё", "е")).canonical
        for cue in analyze_pragmatics(pragmatic_text).cues
        for word in tokenize(cue.surface)
        if word[:1].isalnum()
    }
    sentence = DEFAULT_MORPHOLOGY.analyze_sentence(pragmatic_text)
    readings = sentence.readings
    if sentence.tokens != tokens:
        readings = tuple(DEFAULT_MORPHOLOGY.best(token) for token in tokens)
    selected: list[tuple[str, str]] = []
    for token, reading in zip(tokens, readings, strict=True):
        raw_normalized = token.casefold().replace("ё", "е")
        is_acronym = raw_normalized in acronym_lemmas or (
            token.isascii()
            and token.isalpha()
            and len(token) == 2
            and reading is not None
            and not reading.is_known
        )
        normalized = (
            reading.corrected_form
            if reading is not None and reading.corrected_form
            else normalize_chat_token(raw_normalized).canonical
        )
        if not normalized[:1].isalnum() or (len(normalized) <= 2 and not is_acronym):
            continue
        if normalized in cue_words:
            continue
        if is_service_word(
            reading.lemma if reading is not None else normalized,
            part_of_speech=reading.part_of_speech if reading is not None else None,
            question_context="?" in pragmatic_text,
        ):
            continue
        if reading is not None and reading.part_of_speech in _GRAMMATICAL_PARTS:
            continue
        if reading is None and role(normalized) not in _CONTENT_ROLES and not is_acronym:
            continue
        lemma = reading.lemma if reading is not None else DEFAULT_MORPHOLOGY.lemma(normalized)
        if lemma in _NON_TOPICAL_LEMMAS:
            continue
        selected.append((normalized, lemma))
    return tuple(dict.fromkeys(selected))
