"""Observe symbolic usage and expose exact facts to the reply layer."""

from __future__ import annotations

from debilgi.events.models import Event, EventType
from debilgi.literacy.models import LiteracyFact, SymbolObservation
from debilgi.literacy.reasoner import analyze_symbols, facts_for, query_kinds
from debilgi.literacy.repository import LiteracyRepository

_MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


class LiteracyEngine:
    def __init__(self, repository: LiteracyRepository) -> None:
        self.repository = repository

    async def observe(self, event: Event) -> None:
        if event.event_type not in _MESSAGE_TYPES or event.payload.get("author_is_bot") is True:
            return
        content = event.payload.get("content")
        if not isinstance(content, str) or not content.strip():
            return
        await self.repository.save(
            SymbolObservation(
                event_id=event.event_id,
                guild_id=event.guild_id,
                actor_id=event.actor_id,
                analysis=analyze_symbols(content),
                query_kinds=query_kinds(content),
                observed_at=event.utc_timestamp,
            )
        )

    def context_for_text(self, text: str) -> tuple[LiteracyFact, ...]:
        return facts_for(text)
