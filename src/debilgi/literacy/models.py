"""Models for deterministic symbolic literacy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SymbolAnalysis:
    letters: int
    digits: int
    punctuation: int
    symbols: int
    whitespace: int
    numbers: tuple[str, ...]
    scripts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiteracyFact:
    kind: str
    text: str
    exact: bool = True


@dataclass(frozen=True, slots=True)
class SymbolObservation:
    event_id: str
    guild_id: int
    actor_id: int | None
    analysis: SymbolAnalysis
    query_kinds: tuple[str, ...]
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class LiteracyStats:
    messages_observed: int
    letters_observed: int
    digits_observed: int
    punctuation_observed: int
    symbols_observed: int
    arithmetic_queries: int
    alphabet_queries: int
    counting_queries: int
    comparison_queries: int
    symbol_queries: int
