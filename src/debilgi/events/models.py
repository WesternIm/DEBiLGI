"""Discord-independent event types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


class EventType(StrEnum):
    MESSAGE_CREATED = "MESSAGE_CREATED"
    MESSAGE_REPLIED = "MESSAGE_REPLIED"
    REACTION_ADDED = "REACTION_ADDED"
    REACTION_REMOVED = "REACTION_REMOVED"
    MESSAGE_DELETED = "MESSAGE_DELETED"
    BOT_MENTIONED = "BOT_MENTIONED"
    USER_JOINED = "USER_JOINED"
    USER_LEFT = "USER_LEFT"
    COMMAND_USED = "COMMAND_USED"
    BOT_REPLIED = "BOT_REPLIED"
    BOT_REACTION_ADDED = "BOT_REACTION_ADDED"
    ACTION_RESULT = "ACTION_RESULT"
    TIMEOUT_EXPIRED = "TIMEOUT_EXPIRED"


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    guild_id: int
    channel_id: int | None
    actor_id: int | None
    target_id: int | None
    event_type: EventType
    timestamp: datetime
    payload: Mapping[str, Any]
    source: str

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must not be empty")
        if self.guild_id <= 0:
            raise ValueError("guild_id must be positive")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if not self.source:
            raise ValueError("source must not be empty")

    @property
    def utc_timestamp(self) -> datetime:
        return self.timestamp.astimezone(UTC)


def new_event_id(prefix: str = "event") -> str:
    return f"{prefix}:{uuid4()}"
