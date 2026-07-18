"""Pure functions that translate observations into normalized events."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from debilgi.events.models import Event, EventType


def normalize_message_created(
    *,
    message_id: int,
    guild_id: int,
    channel_id: int,
    author_id: int,
    created_at: datetime,
    content: str,
    author_is_bot: bool,
    reply_to_message_id: int | None = None,
    reply_to_author_id: int | None = None,
    bot_mentioned: bool = False,
    mentioned_user_ids: Iterable[int] = (),
    attachments: Iterable[Mapping[str, Any]] = (),
) -> Event:
    event_type = EventType.MESSAGE_REPLIED if reply_to_message_id else EventType.MESSAGE_CREATED
    return Event(
        event_id=f"discord:message:{message_id}",
        guild_id=guild_id,
        channel_id=channel_id,
        actor_id=author_id,
        target_id=reply_to_author_id,
        event_type=event_type,
        timestamp=created_at,
        payload={
            "message_id": message_id,
            "content": content,
            "author_is_bot": author_is_bot,
            "reply_to_message_id": reply_to_message_id,
            "bot_mentioned": bot_mentioned,
            "mentioned_user_ids": list(mentioned_user_ids),
            "attachments": [dict(attachment) for attachment in attachments],
        },
        source="discord",
    )


def normalize_reaction(
    *,
    added: bool,
    guild_id: int,
    channel_id: int,
    message_id: int,
    actor_id: int,
    target_id: int | None = None,
    emoji: str,
    emoji_animated: bool = False,
    observed_at: datetime | None = None,
) -> Event:
    event_type = EventType.REACTION_ADDED if added else EventType.REACTION_REMOVED
    operation = "add" if added else "remove"
    identity = "\x1f".join(
        (
            str(guild_id),
            str(channel_id),
            str(message_id),
            str(actor_id),
            emoji,
            operation,
        )
    )
    identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return Event(
        event_id=f"discord:reaction:{operation}:{identity_digest}",
        guild_id=guild_id,
        channel_id=channel_id,
        actor_id=actor_id,
        target_id=target_id,
        event_type=event_type,
        timestamp=observed_at or datetime.now(UTC),
        payload={
            "message_id": message_id,
            "emoji": emoji,
            "emoji_animated": emoji_animated,
        },
        source="discord",
    )


def normalize_command_used(
    *,
    interaction_id: int,
    guild_id: int,
    channel_id: int | None,
    actor_id: int,
    command_name: str,
    observed_at: datetime,
) -> Event:
    return Event(
        event_id=f"discord:interaction:{interaction_id}",
        guild_id=guild_id,
        channel_id=channel_id,
        actor_id=actor_id,
        target_id=None,
        event_type=EventType.COMMAND_USED,
        timestamp=observed_at,
        payload={"interaction_id": interaction_id, "command_name": command_name},
        source="discord",
    )


def normalize_message_deleted(
    *,
    message_id: int,
    guild_id: int,
    channel_id: int,
    author_id: int | None,
    observed_at: datetime | None = None,
) -> Event:
    return Event(
        event_id=f"discord:message-delete:{message_id}",
        guild_id=guild_id,
        channel_id=channel_id,
        actor_id=None,
        target_id=author_id,
        event_type=EventType.MESSAGE_DELETED,
        timestamp=observed_at or datetime.now(UTC),
        payload={
            "message_id": message_id,
            "author_known": author_id is not None,
            "deleted_by_moderator": None,
        },
        source="discord",
    )
