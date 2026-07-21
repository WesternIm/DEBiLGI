"""Selective consolidation and context-driven recall of compressed memories."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from debilgi.events.models import Event
from debilgi.language.content import content_lemmas
from debilgi.memory.short_term import MemoryFragment, ShortTermMemory
from debilgi.storage.database import Database


@dataclass(frozen=True, slots=True)
class MemoryCapsule:
    capsule_id: str
    guild_id: int
    channel_id: int | None
    participant_ids: tuple[int, ...]
    topic_terms: tuple[str, ...]
    summary: str
    importance: float
    confidence: float
    strength: float
    created_at: datetime


@dataclass(frozen=True, slots=True)
class MemoryLifecycleStats:
    pending_fragments: int
    capsules: int
    forgotten_fragments: int
    recalled_capsules: int


class LongTermMemoryRepository:
    """Move salient evictions to durable capsules and forget weak noise."""

    def __init__(
        self,
        database: Database,
        working_memory: ShortTermMemory,
        *,
        min_importance: float = 0.34,
        fragment_batch_size: int = 12,
        max_fragment_age: timedelta = timedelta(minutes=30),
        capsule_limit: int = 5_000,
        recall_limit: int = 6,
    ) -> None:
        if not 0.0 <= min_importance <= 1.0:
            raise ValueError("minimum importance must be between zero and one")
        if fragment_batch_size <= 0 or capsule_limit <= 0 or recall_limit <= 0:
            raise ValueError("memory lifecycle limits must be positive")
        self._database = database
        self._working_memory = working_memory
        self.min_importance = min_importance
        self.fragment_batch_size = fragment_batch_size
        self.max_fragment_age = max_fragment_age
        self.capsule_limit = capsule_limit
        self.recall_limit = recall_limit
        self._forgotten_fragments = 0
        self._recalled_capsules = 0

    async def observe(self, event: Event) -> None:
        fragments = await self._working_memory.drain_evicted()
        if fragments:
            await self.queue_fragments(fragments)
            await self.consolidate_ready(event.utc_timestamp)

    async def queue_fragments(self, fragments: tuple[MemoryFragment, ...]) -> int:
        rows = []
        forgotten = 0
        for item in fragments:
            content = item.event.payload.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if item.importance < self.min_importance:
                forgotten += 1
                continue
            rows.append(
                (
                    f"fragment:{uuid4()}",
                    item.event.event_id,
                    item.event.guild_id,
                    item.event.channel_id,
                    item.event.actor_id,
                    item.event.utc_timestamp.isoformat(),
                    _clean(content),
                    _json(memory_terms(content)),
                    item.importance,
                    item.eviction_reason,
                )
            )
        self._forgotten_fragments += forgotten
        if not rows:
            return 0
        async with self._database.transaction() as connection:
            await connection.executemany(
                """INSERT OR IGNORE INTO memory_pending_fragments (
                    fragment_id, event_id, guild_id, channel_id, actor_id,
                    occurred_at, text, topic_terms_json, importance, eviction_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        return len(rows)

    async def consolidate_ready(self, now: datetime | None = None) -> int:
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        rows = await self._database.fetchall(
            """SELECT * FROM memory_pending_fragments
               ORDER BY guild_id, channel_id, occurred_at LIMIT 2000"""
        )
        grouped: dict[tuple[int, int | None], list[Any]] = defaultdict(list)
        for row in rows:
            grouped[(int(row["guild_id"]), _optional_int(row["channel_id"]))].append(row)
        consolidated = 0
        for (guild_id, channel_id), fragments in grouped.items():
            oldest = datetime.fromisoformat(str(fragments[0]["occurred_at"]))
            ready = (
                len(fragments) >= self.fragment_batch_size
                or moment - oldest >= self.max_fragment_age
                or max(float(row["importance"]) for row in fragments) >= 0.78
            )
            if not ready:
                continue
            for start in range(0, len(fragments), self.fragment_batch_size):
                batch = fragments[start : start + self.fragment_batch_size]
                await self._store_capsule(guild_id, channel_id, batch, moment)
                consolidated += 1
        if consolidated:
            await self.maintain(moment)
        return consolidated

    async def recall(
        self,
        guild_id: int,
        prompt: str,
        *,
        channel_id: int | None = None,
        user_id: int | None = None,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> tuple[MemoryCapsule, ...]:
        terms = set(memory_terms(prompt))
        if not terms:
            return ()
        rows = await self._database.fetchall(
            """SELECT * FROM memory_long_term_capsules
               WHERE guild_id = ? AND strength >= 0.08
               ORDER BY importance DESC, updated_at DESC LIMIT 1000""",
            (guild_id,),
        )
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        ranked: list[tuple[float, Any]] = []
        for row in rows:
            stored_terms = set(_json_tuple(row["topic_terms_json"]))
            overlap = len(terms & stored_terms)
            if overlap == 0:
                continue
            topical = overlap / math.sqrt(max(1, len(terms) * len(stored_terms)))
            participants = {int(item) for item in _json_tuple(row["participant_ids_json"])}
            scope = 0.0
            if channel_id is not None and _optional_int(row["channel_id"]) == channel_id:
                scope += 0.14
            if user_id is not None and user_id in participants:
                scope += 0.10
            age_days = max(
                0.0,
                (moment - datetime.fromisoformat(str(row["updated_at"]))).total_seconds()
                / 86400.0,
            )
            recency = 0.5 ** (age_days / 30.0)
            score = (
                topical * 0.50
                + float(row["importance"]) * 0.16
                + float(row["confidence"]) * 0.08
                + float(row["strength"]) * 0.08
                + recency * 0.08
                + scope
            )
            if score >= 0.24:
                ranked.append((score, row))
        maximum = max(1, min(limit or self.recall_limit, 20))
        selected = sorted(ranked, key=lambda item: item[0], reverse=True)[:maximum]
        if selected:
            accessed = moment.isoformat()
            async with self._database.transaction() as connection:
                await connection.executemany(
                    """UPDATE memory_long_term_capsules SET
                        access_count = access_count + 1,
                        strength = MIN(1.0, strength + 0.025),
                        last_accessed_at = ?
                       WHERE capsule_id = ?""",
                    ((accessed, str(row["capsule_id"])) for _, row in selected),
                )
            self._recalled_capsules += len(selected)
        return tuple(self._capsule_from_row(row) for _, row in selected)

    async def maintain(self, now: datetime | None = None) -> int:
        moment = (now or datetime.now(UTC)).astimezone(UTC)
        rows = await self._database.fetchall(
            """SELECT capsule_id, strength, decay_rate, importance, updated_at
               FROM memory_long_term_capsules"""
        )
        changed = 0
        async with self._database.transaction() as connection:
            for row in rows:
                updated = datetime.fromisoformat(str(row["updated_at"]))
                days = max(0.0, (moment - updated).total_seconds() / 86400.0)
                if days <= 0:
                    continue
                strength = float(row["strength"]) * math.exp(
                    -float(row["decay_rate"]) * days
                )
                if strength < 0.06 and float(row["importance"]) < 0.72:
                    await connection.execute(
                        "DELETE FROM memory_long_term_capsules WHERE capsule_id = ?",
                        (str(row["capsule_id"]),),
                    )
                else:
                    await connection.execute(
                        """UPDATE memory_long_term_capsules
                           SET strength = ?, updated_at = ? WHERE capsule_id = ?""",
                        (max(0.0, min(1.0, strength)), moment.isoformat(), row["capsule_id"]),
                    )
                changed += 1
            await connection.execute(
                """DELETE FROM memory_long_term_capsules
                   WHERE capsule_id NOT IN (
                       SELECT capsule_id FROM memory_long_term_capsules
                       ORDER BY importance * strength DESC, updated_at DESC LIMIT ?
                   )""",
                (self.capsule_limit,),
            )
        return changed

    async def backfill_conversation_episodes(self, guild_id: int) -> int:
        rows = await self._database.fetchall(
            """SELECT * FROM language_conversation_episodes
               WHERE guild_id = ? ORDER BY ended_at ASC""",
            (guild_id,),
        )
        inserted = 0
        for row in rows:
            capsule_id = f"capsule:{row['episode_id']}"
            source_text = " ".join(
                (
                    str(row["main_topic"] or ""),
                    str(row["summary"]),
                    " ".join(_json_tuple(row["development_path_json"])),
                    " ".join(_json_tuple(row["conclusions_json"])),
                )
            )
            topics = memory_terms(source_text)[:32]
            compressed = _compress_existing_summary(
                str(row["summary"]),
                _json_tuple(row["conclusions_json"]),
                topics,
            )
            cursor = await self._database.execute(
                """INSERT INTO memory_long_term_capsules (
                    capsule_id, guild_id, channel_id, participant_ids_json,
                    topic_terms_json, summary, source_event_ids_json, source_kind,
                    importance, confidence, strength, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'conversation_episode', ?, ?, ?, ?, ?)
                ON CONFLICT (capsule_id) DO UPDATE SET
                    participant_ids_json = excluded.participant_ids_json,
                    topic_terms_json = excluded.topic_terms_json,
                    summary = excluded.summary,
                    source_event_ids_json = excluded.source_event_ids_json,
                    importance = excluded.importance,
                    confidence = excluded.confidence,
                    strength = MAX(strength, excluded.strength),
                    updated_at = excluded.updated_at""",
                (
                    capsule_id,
                    guild_id,
                    int(row["channel_id"]),
                    str(row["participants_json"]),
                    _json(topics),
                    compressed,
                    str(row["important_message_ids_json"]),
                    float(row["importance_score"]),
                    float(row["confidence"]),
                    max(0.25, float(row["importance_score"])),
                    str(row["ended_at"]),
                    str(row["ended_at"]),
                ),
            )
            inserted += max(0, cursor.rowcount)
        return inserted

    async def stats(self, guild_id: int) -> MemoryLifecycleStats:
        pending = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM memory_pending_fragments WHERE guild_id = ?",
            (guild_id,),
        )
        capsules = await self._database.fetchone(
            "SELECT COUNT(*) AS count FROM memory_long_term_capsules WHERE guild_id = ?",
            (guild_id,),
        )
        return MemoryLifecycleStats(
            pending_fragments=int(pending["count"]) if pending else 0,
            capsules=int(capsules["count"]) if capsules else 0,
            forgotten_fragments=self._forgotten_fragments,
            recalled_capsules=self._recalled_capsules,
        )

    async def _store_capsule(
        self, guild_id: int, channel_id: int | None, rows: list[Any], now: datetime
    ) -> None:
        ordered = sorted(rows, key=lambda row: str(row["occurred_at"]))
        topics = tuple(
            dict.fromkeys(
                term
                for row in ordered
                for term in _json_tuple(row["topic_terms_json"])
            )
        )[:32]
        participants = tuple(
            dict.fromkeys(
                int(row["actor_id"]) for row in ordered if row["actor_id"] is not None
            )
        )
        importance = max(float(row["importance"]) for row in ordered)
        confidence = min(0.94, 0.46 + len(ordered) * 0.04 + importance * 0.20)
        summary = _compress(ordered, topics)
        capsule_id = f"capsule:{uuid4()}"
        async with self._database.transaction() as connection:
            await connection.execute(
                """INSERT INTO memory_long_term_capsules (
                    capsule_id, guild_id, channel_id, participant_ids_json,
                    topic_terms_json, summary, source_event_ids_json, source_kind,
                    importance, confidence, strength, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'working_memory', ?, ?, ?, ?, ?)""",
                (
                    capsule_id,
                    guild_id,
                    channel_id,
                    _json(participants),
                    _json(topics),
                    summary,
                    _json(tuple(str(row["event_id"]) for row in ordered)),
                    importance,
                    confidence,
                    max(0.25, importance),
                    str(ordered[0]["occurred_at"]),
                    now.isoformat(),
                ),
            )
            await connection.executemany(
                "DELETE FROM memory_pending_fragments WHERE fragment_id = ?",
                ((str(row["fragment_id"]),) for row in ordered),
            )

    @staticmethod
    def _capsule_from_row(row: Any) -> MemoryCapsule:
        return MemoryCapsule(
            capsule_id=str(row["capsule_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=_optional_int(row["channel_id"]),
            participant_ids=tuple(
                int(item) for item in _json_tuple(row["participant_ids_json"])
            ),
            topic_terms=_json_tuple(row["topic_terms_json"]),
            summary=str(row["summary"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            strength=float(row["strength"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )


def _compress(rows: list[Any], topics: tuple[str, ...]) -> str:
    strongest = sorted(
        enumerate(rows),
        key=lambda item: (float(item[1]["importance"]), -item[0]),
        reverse=True,
    )[:3]
    selected = sorted(strongest, key=lambda item: item[0])
    statements: list[str] = []
    seen_terms: list[set[str]] = []
    for _, row in selected:
        text = _clean(str(row["text"]), 280).rstrip()
        terms = set(memory_terms(text))
        if any(terms and len(terms & seen) / max(1, len(terms)) > 0.8 for seen in seen_terms):
            continue
        seen_terms.append(terms)
        statements.append(text.rstrip(".!? ") + ".")
    label = ", ".join(topics[:5])
    prefix = f"Тема: {label}. " if label else ""
    return _clean(prefix + "Суть: " + " ".join(statements), 900)


def _clean(text: str, limit: int = 1500) -> str:
    return " ".join(text.split())[:limit]


def memory_terms(text: str) -> tuple[str, ...]:
    return content_lemmas(text)


def _compress_existing_summary(
    summary: str,
    conclusions: tuple[str, ...],
    topics: tuple[str, ...],
) -> str:
    cleaned = _clean(summary, 10_000).lstrip("…")
    segments = [
        item.strip(" .")
        for item in re.split(r"(?<=[.!?])\s+|\s+(?:Затем|Поэтому)\s+", cleaned)
        if len(item.strip()) >= 12
    ]
    topic_set = set(topics)
    ranked = sorted(
        enumerate(segments),
        key=lambda item: (
            len(set(memory_terms(item[1])) & topic_set),
            int(
                any(
                    marker in item[1].casefold()
                    for marker in ("решили", "вывод", "важно", "потому")
                )
            ),
            item[0],
        ),
        reverse=True,
    )[:3]
    chosen = [text.rstrip(".!? ") + "." for _, text in sorted(ranked)]
    for conclusion in conclusions[-2:]:
        normalized = _clean(conclusion, 260).rstrip(".!? ") + "."
        if normalized not in chosen:
            chosen.append(normalized)
    label = ", ".join(topics[:5])
    prefix = f"Тема: {label}. " if label else ""
    return _clean(prefix + "Суть: " + " ".join(chosen), 700)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _json_tuple(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in json.loads(str(value)))


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
