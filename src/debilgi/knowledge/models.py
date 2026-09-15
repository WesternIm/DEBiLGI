"""Models shared by Wikipedia acquisition, storage, and language grounding."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class WikipediaResult:
    page_id: int
    title: str
    extract: str
    canonical_url: str
    wikidata_id: str | None
    revision_id: int | None
    related_titles: tuple[str, ...]
    categories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeConcept:
    concept_id: str
    source: str
    language: str
    external_id: str
    title: str
    definition: str
    canonical_url: str
    wikidata_id: str | None
    revision_id: int | None
    confidence: float
    fetched_at: datetime
    use_count: int


@dataclass(frozen=True, slots=True)
class KnowledgeGrounding:
    """Keep reference definitions identifiable when they enter dialogue memory."""

    facts: tuple[str, ...] = ()
    exact_answer: str | None = None
    definitions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class KnowledgeDecision:
    should_query: bool
    query: str | None
    reason: str | None
    uncertainty: float


@dataclass(frozen=True, slots=True)
class KnowledgeState:
    guild_id: int
    channel_id: int
    evidence_balance: float
    observed_human_events: int
    queries_started: int


@dataclass(frozen=True, slots=True)
class KnowledgeStats:
    concepts: int
    aliases: int
    event_links: int
    context_associations: int
    pending: int
    resolved: int
    cached: int
    no_result: int
    failed: int
