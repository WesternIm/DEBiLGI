"""SQLite concept graph, acquisition ledger, and event-based query balance."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from debilgi.events.models import Event, EventType
from debilgi.knowledge.models import (
    KnowledgeConcept,
    KnowledgeState,
    KnowledgeStats,
    WikipediaResult,
)
from debilgi.knowledge.text import alias_keys, clean_extract, normalize_key
from debilgi.language.content import content_lemmas
from debilgi.storage.database import Database


class KnowledgeRepository:
    def __init__(
        self,
        database: Database,
        *,
        evidence_gain: float = 0.10,
        autonomous_query_cost: float = 0.72,
        explicit_query_cost: float = 0.18,
        minimum_autonomous_balance: float = 0.65,
    ) -> None:
        self._database = database
        self._evidence_gain = evidence_gain
        self._autonomous_query_cost = autonomous_query_cost
        self._explicit_query_cost = explicit_query_cost
        self._minimum_autonomous_balance = minimum_autonomous_balance

    async def observe(self, event: Event) -> None:
        if (
            event.channel_id is None
            or event.actor_id is None
            or event.payload.get("author_is_bot") is True
            or event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
        ):
            return
        await self._database.execute(
            """
            INSERT INTO knowledge_acquisition_states (
                guild_id, channel_id, evidence_balance,
                observed_human_events, queries_started, updated_at
            ) VALUES (?, ?, 1.0, 1, 0, ?)
            ON CONFLICT (guild_id, channel_id) DO UPDATE SET
                evidence_balance = MIN(1.0,
                    knowledge_acquisition_states.evidence_balance + ?),
                observed_human_events = knowledge_acquisition_states.observed_human_events + 1,
                updated_at = excluded.updated_at
            """,
            (
                event.guild_id,
                event.channel_id,
                event.utc_timestamp.isoformat(),
                self._evidence_gain,
            ),
        )

    async def concept_for_alias(
        self,
        value: str,
        *,
        preferred_sources: tuple[str, ...] = (),
    ) -> KnowledgeConcept | None:
        key = normalize_key(value)
        if not key:
            return None
        keys = tuple(
            dict.fromkeys(
                (key, *(content_lemmas(value) if len(key.split()) == 1 else ()))
            )
        )
        placeholders = ",".join("?" for _ in keys)
        rows = await self._database.fetchall(
            f"""
            SELECT concept.*, alias.confidence AS alias_confidence, alias.alias_key
            FROM knowledge_aliases AS alias
            JOIN knowledge_concepts AS concept ON concept.concept_id = alias.concept_id
            WHERE alias.alias_key IN ({placeholders})
            ORDER BY alias.confidence DESC, concept.confidence DESC LIMIT 32
            """,
            keys,
        )
        if not rows:
            return None
        if preferred_sources:
            source_rank = {source: index for index, source in enumerate(preferred_sources)}
            rows.sort(
                key=lambda row: (
                    source_rank.get(str(row["source"]), len(source_rank)),
                    str(row["alias_key"]) != key,
                    -float(row["alias_confidence"]),
                    -float(row["confidence"]),
                )
            )
        else:
            rows.sort(
                key=lambda row: (
                    str(row["alias_key"]) != key,
                    -float(row["alias_confidence"]),
                    -float(row["confidence"]),
                )
            )
        return self._concept_from_row(rows[0])

    async def resolved_concept_for_event(self, event_id: str) -> KnowledgeConcept | None:
        """Return the concept deliberately looked up for this exact event."""

        row = await self._database.fetchone(
            """
            SELECT concept.* FROM knowledge_queries AS query
            JOIN knowledge_concepts AS concept ON concept.concept_id = query.concept_id
            WHERE query.event_id = ? AND query.status IN ('resolved', 'cached')
              AND query.reason IN ('explicit', 'followup')
            ORDER BY query.completed_at DESC LIMIT 1
            """,
            (event_id,),
        )
        return self._concept_from_row(row) if row else None

    async def any_resolved_concept_for_event(self, event_id: str) -> KnowledgeConcept | None:
        """Return knowledge learned for an event, including an autonomous uncertainty lookup."""

        row = await self._database.fetchone(
            """SELECT concept.* FROM knowledge_queries AS query
               JOIN knowledge_concepts AS concept ON concept.concept_id = query.concept_id
               WHERE query.event_id = ? AND query.status IN ('resolved', 'cached')
               ORDER BY query.completed_at DESC LIMIT 1""",
            (event_id,),
        )
        return self._concept_from_row(row) if row else None

    async def recent_explicit_query(
        self,
        guild_id: int,
        channel_id: int,
        actor_id: int | None,
        *,
        before: datetime,
    ) -> str | None:
        """Recover the latest explicit subject for commands such as ``узнай``."""

        row = await self._database.fetchone(
            """
            SELECT query_text FROM knowledge_queries
            WHERE guild_id = ? AND channel_id = ?
              AND (? IS NULL OR actor_id = ?)
              AND reason IN ('explicit', 'followup')
              AND status IN ('resolved', 'cached')
              AND created_at < ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (guild_id, channel_id, actor_id, actor_id, before.isoformat()),
        )
        return str(row["query_text"]) if row else None

    async def record_cache_hit(
        self,
        event: Event,
        *,
        query: str,
        reason: str,
        uncertainty: float,
        concept_id: str,
        source: str,
        language: str,
    ) -> None:
        if event.channel_id is None:
            return
        now = event.utc_timestamp.isoformat()
        query_id = f"query:{uuid4()}"
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO knowledge_queries (
                    query_id, guild_id, channel_id, actor_id, event_id, source, language,
                    query_text, query_key, reason, uncertainty, status,
                    concept_id, created_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'cached', ?, ?, ?)
                """,
                (
                    query_id,
                    event.guild_id,
                    event.channel_id,
                    event.actor_id,
                    event.event_id,
                    source,
                    language,
                    query,
                    normalize_key(query),
                    reason,
                    uncertainty,
                    concept_id,
                    now,
                    now,
                ),
            )
            if cursor.rowcount == 1:
                await self._link_concept(
                    connection,
                    event_id=event.event_id,
                    guild_id=event.guild_id,
                    channel_id=event.channel_id,
                    actor_id=event.actor_id,
                    concept_id=concept_id,
                    source_query_id=query_id,
                    explicit=reason in {"explicit", "followup"},
                    observed_at=now,
                )

    async def reserve_query(
        self,
        event: Event,
        *,
        query: str,
        reason: str,
        uncertainty: float,
        source: str,
        language: str,
    ) -> str | None:
        if event.channel_id is None:
            return None
        query_id = f"query:{uuid4()}"
        query_key = normalize_key(query)
        if not query_key:
            return None
        cost = (
            self._explicit_query_cost
            if reason in {"explicit", "followup"}
            else self._autonomous_query_cost
        )
        async with self._database.transaction() as connection:
            state = await (
                await connection.execute(
                    """
                    SELECT evidence_balance FROM knowledge_acquisition_states
                    WHERE guild_id = ? AND channel_id = ?
                    """,
                    (event.guild_id, event.channel_id),
                )
            ).fetchone()
            balance = float(state["evidence_balance"]) if state else 1.0
            if (
                reason not in {"explicit", "followup"}
                and balance < self._minimum_autonomous_balance
            ):
                return None
            try:
                await connection.execute(
                    """
                    INSERT INTO knowledge_queries (
                        query_id, guild_id, channel_id, actor_id, event_id, source, language,
                        query_text, query_key, reason, uncertainty, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        event.guild_id,
                        event.channel_id,
                        event.actor_id,
                        event.event_id,
                        source,
                        language,
                        query,
                        query_key,
                        reason,
                        uncertainty,
                        event.utc_timestamp.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError:
                return None
            await connection.execute(
                """
                INSERT INTO knowledge_acquisition_states (
                    guild_id, channel_id, evidence_balance,
                    observed_human_events, queries_started, updated_at
                ) VALUES (?, ?, ?, 0, 1, ?)
                ON CONFLICT (guild_id, channel_id) DO UPDATE SET
                    evidence_balance = MAX(0.0,
                        knowledge_acquisition_states.evidence_balance - ?),
                    queries_started = knowledge_acquisition_states.queries_started + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    event.guild_id,
                    event.channel_id,
                    max(0.0, 1.0 - cost),
                    event.utc_timestamp.isoformat(),
                    cost,
                ),
            )
        return query_id

    async def complete_query(
        self,
        query_id: str,
        result: WikipediaResult,
        *,
        source: str,
        language: str,
        completed_at: datetime | None = None,
    ) -> KnowledgeConcept:
        now = (completed_at or datetime.now(UTC)).isoformat()
        concept_id = f"{source}:{language}:{result.page_id}"
        definition = clean_extract(result.extract)
        async with self._database.transaction() as connection:
            query_row = await (
                await connection.execute(
                    """
                    SELECT query_text, guild_id, channel_id, actor_id, event_id, reason
                    FROM knowledge_queries WHERE query_id = ?
                    """,
                    (query_id,),
                )
            ).fetchone()
            if query_row is None:
                raise ValueError(f"Unknown knowledge query: {query_id}")
            await connection.execute(
                """
                INSERT INTO knowledge_concepts (
                    concept_id, source, language, external_id, title, normalized_title,
                    definition, canonical_url, wikidata_id, revision_id, confidence,
                    fetched_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.88, ?, ?)
                ON CONFLICT (concept_id) DO UPDATE SET
                    title = excluded.title,
                    normalized_title = excluded.normalized_title,
                    definition = excluded.definition,
                    canonical_url = excluded.canonical_url,
                    wikidata_id = excluded.wikidata_id,
                    revision_id = excluded.revision_id,
                    confidence = excluded.confidence,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    concept_id,
                    source,
                    language,
                    str(result.page_id),
                    result.title,
                    normalize_key(result.title),
                    definition,
                    result.canonical_url,
                    result.wikidata_id,
                    result.revision_id,
                    now,
                    now,
                ),
            )
            aliases = (
                (str(query_row["query_text"]), 0.82),
                (result.title, 0.98),
            )
            for alias, confidence in aliases:
                alias_key = normalize_key(alias)
                if alias_key:
                    await connection.execute(
                        """
                        INSERT INTO knowledge_aliases (
                            alias_key, alias_text, concept_id, confidence, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (alias_key, concept_id) DO UPDATE SET
                            confidence = MAX(knowledge_aliases.confidence, excluded.confidence)
                        """,
                        (alias_key, alias, concept_id, confidence, now),
                    )
            for title in result.related_titles:
                normalized = normalize_key(title)
                if not normalized:
                    continue
                linked_row = await (
                    await connection.execute(
                        """
                        SELECT concept_id FROM knowledge_aliases
                        WHERE alias_key = ? ORDER BY confidence DESC LIMIT 1
                        """,
                        (normalized,),
                    )
                ).fetchone()
                linked_concept_id = str(linked_row["concept_id"]) if linked_row else None
                digest = hashlib.sha256(
                    f"{concept_id}\x1frelated_to\x1f{normalized}".encode()
                ).hexdigest()[:24]
                await connection.execute(
                    """
                    INSERT OR IGNORE INTO knowledge_relations (
                        relation_id, subject_concept_id, predicate, object_title,
                        normalized_object_title, object_concept_id, confidence,
                        source_query_id, created_at
                    ) VALUES (?, ?, 'related_to', ?, ?, ?, 0.45, ?, ?)
                    """,
                    (
                        f"relation:{digest}",
                        concept_id,
                        title,
                        normalized,
                        linked_concept_id,
                        query_id,
                        now,
                    ),
                )
            await connection.execute(
                """
                UPDATE knowledge_relations SET object_concept_id = ?
                WHERE normalized_object_title = ? AND object_concept_id IS NULL
                """,
                (concept_id, normalize_key(result.title)),
            )
            await connection.execute(
                """
                UPDATE knowledge_queries
                SET status = 'resolved', concept_id = ?, completed_at = ?, error = NULL
                WHERE query_id = ? AND status = 'pending'
                """,
                (concept_id, now, query_id),
            )
            await self._link_concept(
                connection,
                event_id=str(query_row["event_id"]),
                guild_id=int(query_row["guild_id"]),
                channel_id=int(query_row["channel_id"]),
                actor_id=(
                    int(query_row["actor_id"]) if query_row["actor_id"] is not None else None
                ),
                concept_id=concept_id,
                source_query_id=query_id,
                explicit=str(query_row["reason"]) in {"explicit", "followup"},
                observed_at=now,
            )
        concept = await self.concept(concept_id)
        assert concept is not None
        return concept

    async def store_internalized_concept(
        self,
        event: Event,
        *,
        query: str,
        result: WikipediaResult,
        source: str,
        language: str,
        completed_at: datetime | None = None,
    ) -> KnowledgeConcept:
        """Persist background/reasoning study without creating a ready-made answer.

        Unlike ``complete_query``, this method deliberately creates no
        ``knowledge_queries`` row.  The acquired definition becomes internal
        evidence linked to the triggering event, but cannot be returned by
        ``resolved_concept_for_event`` as an exact retrieval response.
        """

        if event.channel_id is None:
            raise ValueError("Internal study requires a channel-scoped event")
        now = (completed_at or datetime.now(UTC)).isoformat()
        concept_id = f"{source}:{language}:{result.page_id}"
        definition = clean_extract(result.extract)
        async with self._database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO knowledge_concepts (
                    concept_id, source, language, external_id, title, normalized_title,
                    definition, canonical_url, wikidata_id, revision_id, confidence,
                    fetched_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.88, ?, ?)
                ON CONFLICT (concept_id) DO UPDATE SET
                    title = excluded.title,
                    normalized_title = excluded.normalized_title,
                    definition = excluded.definition,
                    canonical_url = excluded.canonical_url,
                    wikidata_id = excluded.wikidata_id,
                    revision_id = excluded.revision_id,
                    confidence = excluded.confidence,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    concept_id,
                    source,
                    language,
                    str(result.page_id),
                    result.title,
                    normalize_key(result.title),
                    definition,
                    result.canonical_url,
                    result.wikidata_id,
                    result.revision_id,
                    now,
                    now,
                ),
            )
            for alias, confidence in ((query, 0.82), (result.title, 0.98)):
                alias_key = normalize_key(alias)
                if alias_key:
                    await connection.execute(
                        """
                        INSERT INTO knowledge_aliases (
                            alias_key, alias_text, concept_id, confidence, created_at
                        ) VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT (alias_key, concept_id) DO UPDATE SET
                            confidence = MAX(knowledge_aliases.confidence, excluded.confidence)
                        """,
                        (alias_key, alias, concept_id, confidence, now),
                    )
            for title in result.related_titles:
                normalized = normalize_key(title)
                if not normalized:
                    continue
                linked_row = await (
                    await connection.execute(
                        """SELECT concept_id FROM knowledge_aliases
                           WHERE alias_key = ? ORDER BY confidence DESC LIMIT 1""",
                        (normalized,),
                    )
                ).fetchone()
                linked_concept_id = str(linked_row["concept_id"]) if linked_row else None
                digest = hashlib.sha256(
                    f"{concept_id}\x1frelated_to\x1f{normalized}".encode()
                ).hexdigest()[:24]
                await connection.execute(
                    """
                    INSERT OR IGNORE INTO knowledge_relations (
                        relation_id, subject_concept_id, predicate, object_title,
                        normalized_object_title, object_concept_id, confidence,
                        source_query_id, created_at
                    ) VALUES (?, ?, 'related_to', ?, ?, ?, 0.45, NULL, ?)
                    """,
                    (
                        f"relation:{digest}",
                        concept_id,
                        title,
                        normalized,
                        linked_concept_id,
                        now,
                    ),
                )
            await connection.execute(
                """UPDATE knowledge_relations SET object_concept_id = ?
                   WHERE normalized_object_title = ? AND object_concept_id IS NULL""",
                (concept_id, normalize_key(result.title)),
            )
            await self._link_concept(
                connection,
                event_id=event.event_id,
                guild_id=event.guild_id,
                channel_id=event.channel_id,
                actor_id=event.actor_id,
                concept_id=concept_id,
                source_query_id=None,
                explicit=False,
                observed_at=now,
            )
        concept = await self.concept(concept_id)
        assert concept is not None
        return concept

    async def finish_without_result(
        self, query_id: str, *, error: str | None = None, completed_at: datetime | None = None
    ) -> None:
        status = "failed" if error else "no_result"
        await self._database.execute(
            """
            UPDATE knowledge_queries
            SET status = ?, error = ?, completed_at = ?
            WHERE query_id = ? AND status = 'pending'
            """,
            (
                status,
                error[:500] if error else None,
                (completed_at or datetime.now(UTC)).isoformat(),
                query_id,
            ),
        )

    async def query_status_for_event(self, event_id: str) -> str | None:
        row = await self._database.fetchone(
            """
            SELECT status FROM knowledge_queries
            WHERE event_id = ? ORDER BY created_at DESC LIMIT 1
            """,
            (event_id,),
        )
        return str(row["status"]) if row else None

    async def remember_event_concepts(self, event: Event) -> int:
        if event.channel_id is None or event.payload.get("author_is_bot") is True:
            return 0
        content = event.payload.get("content")
        if not isinstance(content, str):
            return 0
        concepts = await self.concepts_for_text(content, limit=8, mark_used=False)
        linked = 0
        async with self._database.transaction() as connection:
            for concept in concepts:
                changed = await self._link_concept(
                    connection,
                    event_id=event.event_id,
                    guild_id=event.guild_id,
                    channel_id=event.channel_id,
                    actor_id=event.actor_id,
                    concept_id=concept.concept_id,
                    source_query_id=None,
                    explicit=False,
                    observed_at=event.utc_timestamp.isoformat(),
                )
                linked += int(changed)
        return linked

    async def concepts_for_text(
        self,
        value: str,
        limit: int = 3,
        *,
        mark_used: bool = True,
        preferred_sources: tuple[str, ...] = (),
    ) -> tuple[KnowledgeConcept, ...]:
        keys = tuple(dict.fromkeys((*alias_keys(value), *content_lemmas(value))))
        if not keys:
            return ()
        placeholders = ",".join("?" for _ in keys)
        rows = await self._database.fetchall(
            f"""
            SELECT concept.*, MAX(alias.confidence) AS alias_confidence,
                   MAX(LENGTH(alias.alias_key)) AS alias_length
            FROM knowledge_aliases AS alias
            JOIN knowledge_concepts AS concept ON concept.concept_id = alias.concept_id
            WHERE alias.alias_key IN ({placeholders})
            GROUP BY concept.concept_id
            ORDER BY alias_confidence DESC, alias_length DESC
            LIMIT ?
            """,
            (*keys, max(limit, 32 if preferred_sources else limit)),
        )
        if preferred_sources:
            source_rank = {source: index for index, source in enumerate(preferred_sources)}
            rows.sort(
                key=lambda row: (
                    source_rank.get(str(row["source"]), len(source_rank)),
                    -float(row["alias_confidence"]),
                    -int(row["alias_length"]),
                )
            )
        concepts = tuple(self._concept_from_row(row) for row in rows[:limit])
        if concepts and mark_used:
            now = datetime.now(UTC).isoformat()
            async with self._database.transaction() as connection:
                for concept in concepts:
                    await connection.execute(
                        """
                        UPDATE knowledge_concepts
                        SET use_count = use_count + 1, last_used_at = ? WHERE concept_id = ?
                        """,
                        (now, concept.concept_id),
                    )
        return concepts

    async def concept(self, concept_id: str) -> KnowledgeConcept | None:
        row = await self._database.fetchone(
            "SELECT * FROM knowledge_concepts WHERE concept_id = ?", (concept_id,)
        )
        return self._concept_from_row(row) if row else None

    async def state(self, guild_id: int, channel_id: int) -> KnowledgeState:
        row = await self._database.fetchone(
            """
            SELECT * FROM knowledge_acquisition_states WHERE guild_id = ? AND channel_id = ?
            """,
            (guild_id, channel_id),
        )
        if row is None:
            return KnowledgeState(guild_id, channel_id, 1.0, 0, 0)
        return KnowledgeState(
            guild_id,
            channel_id,
            float(row["evidence_balance"]),
            int(row["observed_human_events"]),
            int(row["queries_started"]),
        )

    async def stats(self, guild_id: int) -> KnowledgeStats:
        concept_row = await self._database.fetchone(
            "SELECT COUNT(*) AS concepts FROM knowledge_concepts"
        )
        alias_row = await self._database.fetchone(
            "SELECT COUNT(*) AS aliases FROM knowledge_aliases"
        )
        event_link_row = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM knowledge_event_concepts WHERE guild_id = ?",
            (guild_id,),
        )
        association_row = await self._database.fetchone(
            """
            SELECT COUNT(*) AS count FROM knowledge_context_associations AS association
            WHERE EXISTS (
                SELECT 1 FROM knowledge_event_concepts AS link
                WHERE link.guild_id = ? AND link.concept_id = association.concept_id
            )
            """,
            (guild_id,),
        )
        rows = await self._database.fetchall(
            """
            SELECT status, COUNT(*) AS count FROM knowledge_queries
            WHERE guild_id = ? GROUP BY status
            """,
            (guild_id,),
        )
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return KnowledgeStats(
            concepts=int(concept_row["concepts"]) if concept_row else 0,
            aliases=int(alias_row["aliases"]) if alias_row else 0,
            event_links=int(event_link_row["count"]) if event_link_row else 0,
            context_associations=int(association_row["count"]) if association_row else 0,
            pending=counts.get("pending", 0),
            resolved=counts.get("resolved", 0),
            cached=counts.get("cached", 0),
            no_result=counts.get("no_result", 0),
            failed=counts.get("failed", 0),
        )

    @staticmethod
    def _concept_from_row(row: Any) -> KnowledgeConcept:
        return KnowledgeConcept(
            concept_id=str(row["concept_id"]),
            source=str(row["source"]),
            language=str(row["language"]),
            external_id=str(row["external_id"]),
            title=str(row["title"]),
            definition=str(row["definition"]),
            canonical_url=str(row["canonical_url"]),
            wikidata_id=str(row["wikidata_id"]) if row["wikidata_id"] else None,
            revision_id=int(row["revision_id"]) if row["revision_id"] is not None else None,
            confidence=float(row["confidence"]),
            fetched_at=datetime.fromisoformat(str(row["fetched_at"])),
            use_count=int(row["use_count"]),
        )

    @staticmethod
    async def _link_concept(
        connection: Any,
        *,
        event_id: str,
        guild_id: int,
        channel_id: int,
        actor_id: int | None,
        concept_id: str,
        source_query_id: str | None,
        explicit: bool,
        observed_at: str,
    ) -> bool:
        existing = await (
            await connection.execute(
                """
                SELECT explicit FROM knowledge_event_concepts
                WHERE event_id = ? AND concept_id = ?
                """,
                (event_id, concept_id),
            )
        ).fetchone()
        mention_delta = 0
        explicit_delta = 0
        if existing is None:
            await connection.execute(
                """
                INSERT INTO knowledge_event_concepts (
                    event_id, concept_id, guild_id, channel_id, actor_id,
                    source_query_id, explicit, observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    concept_id,
                    guild_id,
                    channel_id,
                    actor_id,
                    source_query_id,
                    int(explicit),
                    observed_at,
                ),
            )
            mention_delta = 1
            explicit_delta = int(explicit)
        elif explicit and not bool(existing["explicit"]):
            await connection.execute(
                """
                UPDATE knowledge_event_concepts
                SET explicit = 1, source_query_id = COALESCE(?, source_query_id)
                WHERE event_id = ? AND concept_id = ?
                """,
                (source_query_id, event_id, concept_id),
            )
            explicit_delta = 1
        else:
            return False

        scopes = [("guild", guild_id), ("channel", channel_id)]
        if actor_id is not None:
            scopes.append(("user", actor_id))
        signal = 0.90 if explicit else 0.62
        for scope_type, scope_id in scopes:
            await connection.execute(
                """
                INSERT INTO knowledge_context_associations (
                    scope_type, scope_id, concept_id, mention_count,
                    explicit_query_count, relevance, confidence,
                    first_seen_at, last_seen_at, last_event_id
                ) VALUES (?, ?, ?, ?, ?, ?, 0.55, ?, ?, ?)
                ON CONFLICT (scope_type, scope_id, concept_id) DO UPDATE SET
                    mention_count = knowledge_context_associations.mention_count + ?,
                    explicit_query_count = knowledge_context_associations.explicit_query_count + ?,
                    relevance = MIN(1.0,
                        knowledge_context_associations.relevance * 0.85 + ? * 0.15),
                    confidence = MIN(1.0,
                        knowledge_context_associations.confidence + 0.04),
                    last_seen_at = excluded.last_seen_at,
                    last_event_id = excluded.last_event_id
                """,
                (
                    scope_type,
                    scope_id,
                    concept_id,
                    mention_delta,
                    explicit_delta,
                    signal,
                    observed_at,
                    observed_at,
                    event_id,
                    mention_delta,
                    explicit_delta,
                    signal,
                ),
            )
        return True
