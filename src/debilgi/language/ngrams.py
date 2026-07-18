"""Pure frequency helpers for token and n-gram extraction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence


def token_frequencies(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def ngram_frequencies(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if n not in {2, 3}:
        raise ValueError("only bigrams and trigrams are supported")
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
