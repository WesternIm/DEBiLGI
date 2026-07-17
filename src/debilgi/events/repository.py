"""Persistence boundary for normalized events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from debilgi.events.models import Event, EventType
from debilgi.storage.database import Database


class ProcessingStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ProcessingStats:
    pending: int = 0
    processing: int = 0
    processed: int = 0
    failed: int = 0

    @property
    def unfinished(self) -> int:
        return self.pending + self.processing + self.failed


class EventRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def append(self, event: Event) -> bool:
        cursor = await self._database.execute(
            """
            INSERT OR IGNORE INTO events (
                event_id, guild_id, channel_id, actor_id, target_id,
                event_type, occurred_at, payload_json, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.guild_id,
                event.channel_id,
                event.actor_id,
                event.target_id,
                event.event_type.value,
                event.utc_timestamp.isoformat(),
                json.dumps(event.payload, ensure_ascii=False, sort_keys=True),
                event.source,
            ),
        )
        return cursor.rowcount == 1

    async def count(self, guild_id: int | None = None) -> int:
        if guild_id is None:
            row = await self._database.fetchone("SELECT COUNT(*) AS count FROM events")
        else:
            row = await self._database.fetchone(
                "SELECT COUNT(*) AS count FROM events WHERE guild_id = ?", (guild_id,)
            )
        return int(row["count"]) if row else 0

    async def get(self, event_id: str) -> Event | None:
        row = await self._database.fetchone(
            """
            SELECT event_id, guild_id, channel_id, actor_id, target_id,
                   event_type, occurred_at, payload_json, source
            FROM events
            WHERE event_id = ?
            """,
            (event_id,),
        )
        return self._from_row(row) if row else None

    async def counts_by_type(self, guild_id: int) -> dict[str, int]:
        rows = await self._database.fetchall(
            """
            SELECT event_type, COUNT(*) AS count
            FROM events
            WHERE guild_id = ?
            GROUP BY event_type
            ORDER BY count DESC, event_type ASC
            """,
            (guild_id,),
        )
        return {str(row["event_type"]): int(row["count"]) for row in rows}

    async def recent(self, guild_id: int, limit: int = 10) -> list[Event]:
        safe_limit = max(1, min(limit, 100))
        rows = await self._database.fetchall(
            """
            SELECT event_id, guild_id, channel_id, actor_id, target_id,
                   event_type, occurred_at, payload_json, source
            FROM events
            WHERE guild_id = ?
            ORDER BY occurred_at DESC, recorded_at DESC
            LIMIT ?
            """,
            (guild_id, safe_limit),
        )
        return [self._from_row(row) for row in rows]

    async def since(self, cutoff: datetime, limit: int = 5000) -> list[Event]:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        safe_limit = max(1, min(limit, 100_000))
        rows = await self._database.fetchall(
            """
            SELECT * FROM (
                SELECT event_id, guild_id, channel_id, actor_id, target_id,
                       event_type, occurred_at, payload_json, source, recorded_at
                FROM events
                WHERE occurred_at >= ?
                ORDER BY occurred_at DESC, recorded_at DESC
                LIMIT ?
            ) AS recent_events
            ORDER BY occurred_at ASC, recorded_at ASC
            """,
            (cutoff.isoformat(), safe_limit),
        )
        return [self._from_row(row) for row in rows]

    async def claim_for_processing(self, event_id: str) -> bool:
        cursor = await self._database.execute(
            """
            UPDATE event_processing
            SET status = 'processing',
                attempts = attempts + 1,
                started_at = CURRENT_TIMESTAMP,
                completed_at = NULL,
                last_error = NULL
            WHERE event_id = ? AND status = 'pending'
            """,
            (event_id,),
        )
        return cursor.rowcount == 1

    async def mark_processed(self, event_id: str) -> None:
        await self._database.execute(
            """
            UPDATE event_processing
            SET status = 'processed', completed_at = CURRENT_TIMESTAMP, last_error = NULL
            WHERE event_id = ? AND status = 'processing'
            """,
            (event_id,),
        )

    async def mark_failed(self, event_id: str, error: str) -> None:
        await self._database.execute(
            """
            UPDATE event_processing
            SET status = 'failed', completed_at = CURRENT_TIMESTAMP, last_error = ?
            WHERE event_id = ? AND status = 'processing'
            """,
            (error[:1000], event_id),
        )

    async def recover_unfinished(self, max_attempts: int = 3) -> list[str]:
        await self._database.execute(
            """
            UPDATE event_processing
            SET status = 'pending', started_at = NULL, completed_at = NULL
            WHERE status = 'processing'
               OR (status = 'failed' AND attempts < ?)
            """,
            (max_attempts,),
        )
        rows = await self._database.fetchall(
            """
            SELECT ep.event_id
            FROM event_processing AS ep
            JOIN events AS e ON e.event_id = ep.event_id
            WHERE ep.status = 'pending'
            ORDER BY e.occurred_at ASC, ep.queued_at ASC
            """
        )
        return [str(row["event_id"]) for row in rows]

    async def prepare_replay(self, guild_id: int, limit: int = 100) -> list[str]:
        safe_limit = max(1, min(limit, 1000))
        rows = await self._database.fetchall(
            """
            SELECT ep.event_id
            FROM event_processing AS ep
            JOIN events AS e ON e.event_id = ep.event_id
            WHERE e.guild_id = ? AND ep.status != 'processing'
            ORDER BY e.occurred_at DESC, ep.queued_at DESC
            LIMIT ?
            """,
            (guild_id, safe_limit),
        )
        event_ids = [str(row["event_id"]) for row in reversed(rows)]
        for event_id in event_ids:
            await self._database.execute(
                """
                UPDATE event_processing
                SET status = 'pending', attempts = 0, last_error = NULL,
                    started_at = NULL, completed_at = NULL,
                    queued_at = CURRENT_TIMESTAMP
                WHERE event_id = ? AND status != 'processing'
                """,
                (event_id,),
            )
        return event_ids

    async def processing_stats(self, guild_id: int) -> ProcessingStats:
        rows = await self._database.fetchall(
            """
            SELECT ep.status, COUNT(*) AS count
            FROM event_processing AS ep
            JOIN events AS e ON e.event_id = ep.event_id
            WHERE e.guild_id = ?
            GROUP BY ep.status
            """,
            (guild_id,),
        )
        counts = {str(row["status"]): int(row["count"]) for row in rows}
        return ProcessingStats(
            pending=counts.get(ProcessingStatus.PENDING.value, 0),
            processing=counts.get(ProcessingStatus.PROCESSING.value, 0),
            processed=counts.get(ProcessingStatus.PROCESSED.value, 0),
            failed=counts.get(ProcessingStatus.FAILED.value, 0),
        )

    async def processing_status(self, event_id: str) -> ProcessingStatus | None:
        row = await self._database.fetchone(
            "SELECT status FROM event_processing WHERE event_id = ?", (event_id,)
        )
        return ProcessingStatus(str(row["status"])) if row else None

    @staticmethod
    def _from_row(row: Any) -> Event:
        return Event(
            event_id=str(row["event_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]) if row["channel_id"] is not None else None,
            actor_id=int(row["actor_id"]) if row["actor_id"] is not None else None,
            target_id=int(row["target_id"]) if row["target_id"] is not None else None,
            event_type=EventType(str(row["event_type"])),
            timestamp=datetime.fromisoformat(str(row["occurred_at"])),
            payload=json.loads(str(row["payload_json"])),
            source=str(row["source"]),
        )
