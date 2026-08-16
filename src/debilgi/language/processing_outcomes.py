"""Durable diagnostic storage for semantic planning and realization outcomes."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

from debilgi.events.models import Event
from debilgi.language.semantic_reply_plan import ProcessingOutcome
from debilgi.storage.database import Database

logger = logging.getLogger("debilgi.learning_quality")


class ProcessingOutcomeRepository:
    def __init__(self, database: Database) -> None:
        self._database = database
        self._quality_observer: (
            Callable[[Event, ProcessingOutcome, datetime], Awaitable[None]] | None
        ) = None

    def set_quality_observer(
        self,
        observer: Callable[[Event, ProcessingOutcome, datetime], Awaitable[None]] | None,
    ) -> None:
        self._quality_observer = observer

    async def save(
        self,
        event: Event,
        outcome: ProcessingOutcome,
        *,
        response_message_id: int | None,
        created_at: datetime,
    ) -> None:
        await self._database.execute(
            """INSERT OR REPLACE INTO language_processing_outcomes (
                   source_event_id, guild_id, channel_id, response_message_id,
                   intention_id, reply_plan_id, outcome_type, generated_text,
                   failure_stage, failure_reason, was_intentional, diagnostic_json,
                   created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.guild_id,
                event.channel_id,
                response_message_id,
                outcome.intention_id,
                outcome.reply_plan_id,
                outcome.outcome_type.value,
                outcome.generated_text,
                outcome.failure_stage,
                outcome.failure_reason,
                int(outcome.was_intentional),
                json.dumps(
                    dict(outcome.diagnostic_data),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                created_at.isoformat(),
            ),
        )
        if self._quality_observer is not None:
            try:
                await self._quality_observer(event, outcome, created_at)
            except Exception:
                logger.exception(
                    "quality_loop_failed event=%s outcome=%s",
                    event.event_id,
                    outcome.outcome_type.value,
                )
