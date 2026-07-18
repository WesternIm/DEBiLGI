"""Discord-independent archive import models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ImportStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ArchiveImportJob:
    job_id: str
    guild_id: int
    channel_id: int
    requested_by: int
    after_at: datetime
    before_at: datetime
    max_messages: int
    status: ImportStatus
    last_message_id: int | None = None
    messages_seen: int = 0
    messages_processed: int = 0
    delete_raw_after: bool = False
    attempts: int = 0
    last_error: str | None = None

    @property
    def remaining(self) -> int:
        return max(0, self.max_messages - self.messages_seen)

    def __post_init__(self) -> None:
        if self.after_at.tzinfo is None or self.before_at.tzinfo is None:
            raise ValueError("archive time bounds must be timezone-aware")
        if self.after_at.astimezone(UTC) >= self.before_at.astimezone(UTC):
            raise ValueError("after_at must be before before_at")
        if self.max_messages <= 0:
            raise ValueError("max_messages must be positive")


@dataclass(frozen=True, slots=True)
class ArchivedMessage:
    message_id: int
    guild_id: int
    channel_id: int
    author_id: int
    created_at: datetime
    content: str
    reply_to_message_id: int | None
    author_is_bot: bool

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")
