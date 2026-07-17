"""Text normalization for concept lookup without a heavyweight NLP dependency."""

from __future__ import annotations

import re

_MENTION_RE = re.compile(r"<@!?\d+>")
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_-]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+\]")


def normalize_key(value: str) -> str:
    words = _WORD_RE.findall(_MENTION_RE.sub(" ", value).casefold().replace("ё", "е"))
    return " ".join(words)


def words(value: str) -> tuple[str, ...]:
    return tuple(_WORD_RE.findall(_MENTION_RE.sub(" ", value).casefold().replace("ё", "е")))


def alias_keys(value: str, *, maximum_words: int = 3) -> tuple[str, ...]:
    tokens = words(value)
    keys: list[str] = []
    for size in range(min(maximum_words, len(tokens)), 0, -1):
        for index in range(len(tokens) - size + 1):
            key = " ".join(tokens[index : index + size])
            if key not in keys:
                keys.append(key)
    return tuple(keys)


def clean_extract(value: str, *, limit: int = 1500) -> str:
    without_citations = _CITATION_RE.sub("", value)
    return _WHITESPACE_RE.sub(" ", without_citations).strip()[:limit]
