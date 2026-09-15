"""Durable delivery state, separate from observation processing and diagnostics."""

from __future__ import annotations

import json
from dataclasses import fields
from datetime import datetime

from debilgi.experience.models import PreparedReply
from debilgi.language.semantic_reply_plan import ProcessingOutcome, ProcessingOutcomeType
from debilgi.storage.database import Database


def encode_reply(reply: PreparedReply) -> str:
    data = {field.name: getattr(reply, field.name) for field in fields(reply)}
    if reply.processing_outcome is not None:
        outcome = reply.processing_outcome
        data['processing_outcome'] = {
            field.name: (dict(outcome.diagnostic_data) if field.name == 'diagnostic_data'
                         else getattr(outcome, field.name))
            for field in fields(outcome)
        }
    return json.dumps(data, ensure_ascii=False)


def decode_reply(payload: str) -> PreparedReply:
    data = json.loads(payload)
    data['tokens'] = tuple(data['tokens'])
    data['initiative_reasoning'] = tuple(data['initiative_reasoning'])
    if data['processing_outcome'] is not None:
        outcome = data['processing_outcome']
        outcome['outcome_type'] = ProcessingOutcomeType(outcome['outcome_type'])
        data['processing_outcome'] = ProcessingOutcome(**outcome)
    return PreparedReply(**data)


class DeliveryRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    async def recover(self) -> None:
        await self.database.execute(
            """UPDATE reply_delivery_jobs SET status = CASE
                   WHEN status = 'sending' THEN 'uncertain'
                   WHEN prepared_json IS NULL THEN 'pending' ELSE 'prepared' END,
                   next_attempt_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE status IN ('preparing', 'sending')"""
        )

    async def next_job(self):
        return await self.database.fetchone(
            """SELECT * FROM reply_delivery_jobs
               WHERE status IN ('pending', 'prepared', 'sent', 'uncertain')
                 AND next_attempt_at <= CURRENT_TIMESTAMP
               ORDER BY CASE WHEN status = 'sent' THEN 0 ELSE 1 END, created_at, event_id
               LIMIT 1"""
        )

    async def get(self, event_id: str):
        return await self.database.fetchone(
            'SELECT * FROM reply_delivery_jobs WHERE event_id = ?', (event_id,)
        )

    async def preparing(self, event_id: str) -> None:
        await self.database.execute(
            """UPDATE reply_delivery_jobs SET status = 'preparing', attempts = attempts + 1,
                   updated_at = CURRENT_TIMESTAMP WHERE event_id = ? AND status = 'pending'""",
            (event_id,),
        )

    async def save_reply(self, event_id: str, reply: PreparedReply) -> None:
        await self.database.execute(
            """UPDATE reply_delivery_jobs SET prepared_json = ?, status = 'prepared',
                   attempts = 0, last_error = NULL, updated_at = CURRENT_TIMESTAMP
               WHERE event_id = ?""", (encode_reply(reply), event_id),
        )

    async def state(self, event_id: str, status: str, error: str | None = None,
                    delay: int = 0) -> None:
        await self.database.execute(
            """UPDATE reply_delivery_jobs SET status = ?, last_error = ?,
                   next_attempt_at = datetime('now', ?), updated_at = CURRENT_TIMESTAMP
               WHERE event_id = ?""",
            (status, error, f'+{delay} seconds', event_id),
        )

    async def sent(self, event_id: str, message_id: int, created_at: datetime) -> None:
        await self.database.execute(
            """UPDATE reply_delivery_jobs SET status = 'sent', response_message_id = ?,
                   sent_at = ?, attempts = 0, last_error = NULL,
                   next_attempt_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE event_id = ?""", (message_id, created_at.isoformat(), event_id),
        )
