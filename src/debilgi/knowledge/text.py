"""Text normalization for concept lookup without a heavyweight NLP dependency."""

from __future__ import annotations

import re

from debilgi.language.abbreviations import is_abbreviation_candidate, normalize_abbreviation

_MENTION_RE = re.compile(r"<@!?\d+>")
# Knowledge names frequently contain a numeric discriminator (``GTA 5``,
# ``Python 3``).  Letter tokens keep the old three-character floor so common
# one-letter function words do not pollute aliases, while standalone numbers
# are retained instead of silently changing the queried entity.
_WORD_RE = re.compile(
    r"(?:[A-Za-zА-Яа-яЁё]\.){2,}[A-Za-zА-Яа-яЁё]?\.?"
    r"|[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_-]*"
    r"|[0-9]{1,8}"
)
_WHITESPACE_RE = re.compile(r"\s+")
_CITATION_RE = re.compile(r"\[\d+\]")


def normalize_key(value: str) -> str:
    return " ".join(_knowledge_words(value))


def words(value: str) -> tuple[str, ...]:
    return _knowledge_words(value)


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


def _knowledge_words(value: str) -> tuple[str, ...]:
    raw_words = _WORD_RE.findall(_MENTION_RE.sub(" ", value).replace("ё", "е"))
    selected: list[str] = []
    for raw in raw_words:
        normalized = raw.casefold()
        if raw.isdigit():
            selected.append(normalized)
        elif len(normalize_abbreviation(raw)) >= 3:
            selected.append(normalize_abbreviation(raw) if "." in raw else normalized)
        elif is_abbreviation_candidate(raw):
            selected.append(normalize_abbreviation(raw))
    return tuple(selected)
