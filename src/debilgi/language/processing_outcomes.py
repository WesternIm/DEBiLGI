"""Durable diagnostic storage for semantic planning and realization outcomes."""

from __future__ import annotations

import json
from datetime import datetime

from debilgi.events.models import Event
from debilgi.language.semantic_reply_plan import ProcessingOutcome
from debilgi.storage.database import Database


class ProcessingOutcomeRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

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
