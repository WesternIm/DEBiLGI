"""Persistent open goals and event-based initiative balance."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from debilgi.autonomy.models import AutonomyGoal, InitiativeState
from debilgi.events.models import Event, EventType
from debilgi.experience.models import PreparedReply
from debilgi.storage.database import Database


class AutonomyRepository:
    def __init__(
        self,
        database: Database,
        *,
        evidence_gain: float = 0.14,
        initiative_cost: float = 0.72,
        abandonment_events: int = 12,
    ) -> None:
        self._database = database
        self._evidence_gain = evidence_gain
        self._initiative_cost = initiative_cost
        self._abandonment_events = abandonment_events

    async def observe(self, event: Event) -> None:
        if (
            event.channel_id is None
            or event.actor_id is None
            or event.payload.get("author_is_bot") is True
            or event.event_type not in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
        ):
            return
        async with self._database.transaction() as connection:
            await connection.execute(
                """
                INSERT INTO cognition_initiative_states (
                    guild_id, channel_id, evidence_balance,
                    observed_human_events, initiatives_started, updated_at
                ) VALUES (?, ?, 1.0, 1, 0, ?)
                ON CONFLICT (guild_id, channel_id) DO UPDATE SET
                    evidence_balance = MIN(1.0,
                        cognition_initiative_states.evidence_balance + ?),
                    observed_human_events = cognition_initiative_states.observed_human_events + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    event.guild_id,
                    event.channel_id,
                    event.utc_timestamp.isoformat(),
                    self._evidence_gain,
                ),
            )
            row = await (
                await connection.execute(
                    """
                    SELECT * FROM cognition_autonomy_goals
                    WHERE guild_id = ? AND channel_id = ? AND status = 'open'
                    """,
                    (event.guild_id, event.channel_id),
                )
            ).fetchone()
            if row is None:
                return
            reply_to = _payload_int(event, "reply_to_message_id")
            status = None
            if reply_to == int(row["response_message_id"]):
                status = "answered"
            elif row["target_user_id"] is not None and event.actor_id == int(row["target_user_id"]):
                status = "continued"
            events_waited = int(row["events_waited"]) + 1
            if status is None and events_waited >= self._abandonment_events:
                status = "abandoned"
            if status is None:
                await connection.execute(
                    """
                    UPDATE cognition_autonomy_goals SET events_waited = ? WHERE goal_id = ?
                    """,
                    (events_waited, str(row["goal_id"])),
                )
            else:
                await connection.execute(
                    """
                    UPDATE cognition_autonomy_goals
                    SET status = ?, events_waited = ?, resolved_at = ?, outcome_event_id = ?
                    WHERE goal_id = ? AND status = 'open'
                    """,
                    (
                        status,
                        events_waited,
                        event.utc_timestamp.isoformat(),
                        event.event_id,
                        str(row["goal_id"]),
                    ),
                )

    async def state(self, guild_id: int, channel_id: int) -> InitiativeState:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_initiative_states WHERE guild_id = ? AND channel_id = ?
            """,
            (guild_id, channel_id),
        )
        if row is None:
            return InitiativeState(guild_id, channel_id, 1.0, 0, 0, None)
        return InitiativeState(
            guild_id=guild_id,
            channel_id=channel_id,
            evidence_balance=float(row["evidence_balance"]),
            observed_human_events=int(row["observed_human_events"]),
            initiatives_started=int(row["initiatives_started"]),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
        )

    async def open_goal(self, guild_id: int, channel_id: int) -> AutonomyGoal | None:
        row = await self._database.fetchone(
            """
            SELECT * FROM cognition_autonomy_goals
            WHERE guild_id = ? AND channel_id = ? AND status = 'open'
            """,
            (guild_id, channel_id),
        )
        return self._goal_from_row(row) if row else None

    async def start_goal(
        self,
        prompt: Event,
        *,
        response_message_id: int,
        reply: PreparedReply,
        created_at: datetime,
    ) -> bool:
        if prompt.channel_id is None or not reply.initiative or reply.goal_id is None:
            return False
        async with self._database.transaction() as connection:
            cursor = await connection.execute(
                """
                INSERT OR IGNORE INTO cognition_autonomy_goals (
                    goal_id, guild_id, channel_id, target_user_id, trigger_event_id,
                    intent_id, response_message_id, drive, reasoning_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reply.goal_id,
                    prompt.guild_id,
                    prompt.channel_id,
                    prompt.actor_id,
                    prompt.event_id,
                    reply.intent_id,
                    response_message_id,
                    reply.initiative_drive,
                    json.dumps(reply.initiative_reasoning, separators=(",", ":")),
                    created_at.isoformat(),
                ),
            )
            if cursor.rowcount != 1:
                return False
            await connection.execute(
                """
                INSERT INTO cognition_initiative_states (
                    guild_id, channel_id, evidence_balance,
                    observed_human_events, initiatives_started, updated_at
                ) VALUES (?, ?, ?, 0, 1, ?)
                ON CONFLICT (guild_id, channel_id) DO UPDATE SET
                    evidence_balance = MAX(0.0,
                        cognition_initiative_states.evidence_balance - ?),
                    initiatives_started = cognition_initiative_states.initiatives_started + 1,
                    updated_at = excluded.updated_at
                """,
                (
                    prompt.guild_id,
                    prompt.channel_id,
                    max(0.0, 1.0 - self._initiative_cost),
                    created_at.isoformat(),
                    self._initiative_cost,
                ),
            )
        return True

    async def goal_counts(self, guild_id: int) -> dict[str, int]:
        rows = await self._database.fetchall(
            """
            SELECT status, COUNT(*) AS count FROM cognition_autonomy_goals
            WHERE guild_id = ? GROUP BY status
            """,
            (guild_id,),
        )
        return {str(row["status"]): int(row["count"]) for row in rows}

    @staticmethod
    def _goal_from_row(row: Any) -> AutonomyGoal:
        return AutonomyGoal(
            goal_id=str(row["goal_id"]),
            guild_id=int(row["guild_id"]),
            channel_id=int(row["channel_id"]),
            target_user_id=(
                int(row["target_user_id"]) if row["target_user_id"] is not None else None
            ),
            trigger_event_id=str(row["trigger_event_id"]),
            intent_id=str(row["intent_id"]),
            response_message_id=int(row["response_message_id"]),
            drive=float(row["drive"]),
            reasoning=tuple(json.loads(str(row["reasoning_json"]))),
            status=str(row["status"]),
            events_waited=int(row["events_waited"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            resolved_at=(
                datetime.fromisoformat(str(row["resolved_at"])) if row["resolved_at"] else None
            ),
            outcome_event_id=(str(row["outcome_event_id"]) if row["outcome_event_id"] else None),
        )


def _payload_int(event: Event, key: str) -> int | None:
    value = event.payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
