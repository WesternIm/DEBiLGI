"""Discord-independent event types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from debilgi.language.dialogue_acts import TurnForm
    from debilgi.language.structured_utterance import StructuredUtterance


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
    _meaning: StructuredUtterance | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _turn_form: TurnForm | None = field(default=None, init=False, repr=False, compare=False)

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
    def turn_form(self) -> TurnForm:
        if self._turn_form is None:
            from debilgi.language.dialogue_acts import analyze_turn_form

            bot = self.payload.get("bot_user_id")
            reply = (bot is not None and self.payload.get("reply_to_message_id") is not None
                     and bot in {self.target_id, self.payload.get("reply_to_author_id")})
            evidence = tuple(name for name, present in (
                ("mention", self.payload.get("bot_mentioned")), ("reply_to_bot", reply),
                ("target_bot", bot is not None and self.target_id == bot),
            ) if present)
            form = analyze_turn_form(str(self.payload.get("content", "")),
                                     addressed=bool(evidence), address_evidence=evidence,
                                     follows_bot=reply)
            object.__setattr__(self, "_turn_form", form)
        return self._turn_form

    @property
    def meaning(self) -> StructuredUtterance:
        """One versioned analysis shared by observers and reply planning."""
        if self._meaning is None:
            _ = self.turn_form  # Commit form before lexical/semantic parsing.
            from debilgi.language.addressing import DEFAULT_ADDRESSING
            from debilgi.language.entities import analyze_entities, render_message_content
            from debilgi.language.structured_utterance import (
                analyze_structured_utterance,
                bind_utterance,
            )

            content = self.payload.get("content", "")
            text = render_message_content(content if isinstance(content, str) else "", self.payload)
            text = DEFAULT_ADDRESSING.analyze(" ".join(text.split())).content_text
            agent_id = self.payload.get("bot_user_id")
            # Seed dialogues have human addressees; a bot's existence alone is
            # never evidence that every historical 'ты' referred to the agent.
            addressee = self.target_id
            if self.payload.get("bot_mentioned"):
                addressee = agent_id
            previous = self.payload.get("reply_to_message_id")
            structure = bind_utterance(
                analyze_structured_utterance(text),
                source_event_id=self.event_id,
                speaker_id=self.actor_id,
                addressee_id=addressee,
                agent_id=agent_id,
                observed_at=self.timestamp,
                entity_analysis=analyze_entities(text, self.payload),
                previous_event_id=f"discord:message:{previous}" if previous else None,
            )
            object.__setattr__(self, "_meaning", structure)
        return self._meaning

    @property
    def utc_timestamp(self) -> datetime:
        return self.timestamp.astimezone(UTC)


def new_event_id(prefix: str = "event") -> str:
    return f"{prefix}:{uuid4()}"
