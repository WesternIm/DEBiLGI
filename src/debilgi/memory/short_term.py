"""Bounded, decaying, guild-isolated short-term event memory."""

from __future__ import annotations

import asyncio
import math
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from debilgi.events.models import Event, EventType
from debilgi.language.content import content_lemmas
from debilgi.language.pragmatics import analyze_pragmatics
from debilgi.language.tokenizer import tokenize

Clock = Callable[[], datetime]
ChannelKey = tuple[int, int]
UserKey = tuple[int, int]
MessageKey = tuple[int, int, int]
ChainKey = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    retention: timedelta = timedelta(hours=1)
    charge_half_life: timedelta = timedelta(minutes=15)
    max_total_events: int = 5000
    max_channel_events: int = 100
    max_user_events: int = 50
    max_reply_chains: int = 100
    max_chain_events: int = 50
    max_context_events: int = 24
    relevance_half_life: timedelta = timedelta(minutes=20)
    min_context_relevance: float = 0.10
    max_charge: float = 10.0

    def __post_init__(self) -> None:
        numeric_limits = (
            self.max_total_events,
            self.max_channel_events,
            self.max_user_events,
            self.max_reply_chains,
            self.max_chain_events,
            self.max_context_events,
        )
        if (
            self.retention <= timedelta(0)
            or self.charge_half_life <= timedelta(0)
            or self.relevance_half_life <= timedelta(0)
        ):
            raise ValueError("memory durations must be positive")
        if any(value <= 0 for value in numeric_limits) or self.max_charge <= 0:
            raise ValueError("memory size limits must be positive")
        if not 0.0 <= self.min_context_relevance <= 1.0:
            raise ValueError("minimum context relevance must be between zero and one")


@dataclass(frozen=True, slots=True)
class MemoryStats:
    events: int
    channels: int
    users: int
    reply_chains: int
    charged_users: int
    pending_consolidation: int = 0


@dataclass(frozen=True, slots=True)
class MemoryFragment:
    event: Event
    importance: float
    eviction_reason: str


@dataclass(slots=True)
class _Charge:
    value: float
    updated_at: datetime


class ShortTermMemory:
    _CHARGE_WEIGHTS = {
        EventType.MESSAGE_CREATED: 1.0,
        EventType.MESSAGE_REPLIED: 1.25,
        EventType.REACTION_ADDED: 0.5,
        EventType.REACTION_REMOVED: 0.25,
        EventType.BOT_REPLIED: 0.75,
        EventType.BOT_REACTION_ADDED: 0.4,
        EventType.COMMAND_USED: 0.75,
    }

    def __init__(self, limits: MemoryLimits | None = None, *, clock: Clock | None = None) -> None:
        self.limits = limits or MemoryLimits()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = asyncio.Lock()
        self._events: dict[str, Event] = {}
        self._timeline: deque[str] = deque()
        self._channels: dict[ChannelKey, deque[str]] = {}
        self._users: dict[UserKey, deque[str]] = {}
        self._message_roots: dict[MessageKey, int] = {}
        self._message_events: dict[MessageKey, str] = {}
        self._chains: dict[ChainKey, deque[str]] = {}
        self._chain_updated: dict[ChainKey, datetime] = {}
        self._charges: dict[UserKey, _Charge] = {}
        self._importance: dict[str, float] = {}
        self._evicted: deque[MemoryFragment] = deque(maxlen=self.limits.max_total_events)

    async def observe(self, event: Event) -> bool:
        async with self._lock:
            now = self._utc_now()
            self._prune_locked(now)
            if event.event_id in self._events:
                return False
            if event.utc_timestamp < now - self.limits.retention:
                return False

            self._events[event.event_id] = event
            self._importance[event.event_id] = self._event_importance(event)
            self._timeline.append(event.event_id)
            if event.channel_id is not None:
                channel = self._channels.setdefault((event.guild_id, event.channel_id), deque())
                channel.append(event.event_id)
                self._sort_and_trim_locked(channel, self.limits.max_channel_events)
            if event.actor_id is not None:
                user = self._users.setdefault((event.guild_id, event.actor_id), deque())
                user.append(event.event_id)
                self._sort_and_trim_locked(user, self.limits.max_user_events)

            self._observe_reply_chain_locked(event)
            self._observe_charge_locked(event, now)
            self._prune_locked(now)
            return True

    async def rebuild(self, events: Iterable[Event]) -> None:
        await self.clear()
        for event in sorted(events, key=lambda item: (item.utc_timestamp, item.event_id)):
            await self.observe(event)
        async with self._lock:
            self._evicted.clear()

    async def clear(self) -> None:
        async with self._lock:
            self._events.clear()
            self._timeline.clear()
            self._channels.clear()
            self._users.clear()
            self._message_roots.clear()
            self._message_events.clear()
            self._chains.clear()
            self._chain_updated.clear()
            self._charges.clear()
            self._importance.clear()
            self._evicted.clear()

    async def channel_context(self, guild_id: int, channel_id: int) -> tuple[Event, ...]:
        async with self._lock:
            self._prune_locked(self._utc_now())
            return self._resolve(self._channels.get((guild_id, channel_id), ()))

    async def user_context(self, guild_id: int, user_id: int) -> tuple[Event, ...]:
        async with self._lock:
            self._prune_locked(self._utc_now())
            return self._resolve(self._users.get((guild_id, user_id), ()))

    async def guild_context(self, guild_id: int) -> tuple[Event, ...]:
        async with self._lock:
            self._prune_locked(self._utc_now())
            return tuple(
                self._events[event_id]
                for event_id in self._timeline
                if event_id in self._events and self._events[event_id].guild_id == guild_id
            )

    async def relevant_context(
        self,
        guild_id: int,
        *,
        channel_id: int | None,
        user_id: int | None,
        prompt: str,
        limit: int | None = None,
    ) -> tuple[Event, ...]:
        """Select a small working set by topic, recency and salience."""

        async with self._lock:
            now = self._utc_now()
            self._prune_locked(now)
            maximum = max(1, min(limit or self.limits.max_context_events, 100))
            prompt_terms = self._content_terms(prompt)
            candidates = [
                self._events[event_id]
                for event_id in self._timeline
                if event_id in self._events
                and self._events[event_id].guild_id == guild_id
                and (
                    channel_id is None
                    or self._events[event_id].channel_id == channel_id
                    or (user_id is not None and self._events[event_id].actor_id == user_id)
                )
            ]
            recent_channel_ids = [
                event.event_id
                for event in candidates
                if channel_id is not None and event.channel_id == channel_id
            ]
            # Keep an actual conversational window even when a short follow-up has
            # no lexical overlap with the earlier subject ("сверил?", "и что?").
            continuity_width = min(maximum, 12)
            continuity = set(recent_channel_ids[-continuity_width:])
            ranked: list[tuple[float, datetime, Event]] = []
            for event in candidates:
                content = event.payload.get("content")
                event_terms = self._content_terms(content if isinstance(content, str) else "")
                overlap = len(prompt_terms & event_terms)
                topical = overlap / math.sqrt(max(1, len(prompt_terms) * len(event_terms)))
                age = max(0.0, (now - event.utc_timestamp).total_seconds())
                recency = 0.5 ** (age / self.limits.relevance_half_life.total_seconds())
                importance = self._importance.get(event.event_id, 0.2)
                scope = 0.0
                if channel_id is not None and event.channel_id == channel_id:
                    scope += 0.06
                if user_id is not None and event.actor_id == user_id:
                    scope += 0.05
                relevance = recency * 0.38 + topical * 0.38 + importance * 0.13 + scope
                if relevance >= self.limits.min_context_relevance or event.event_id in continuity:
                    ranked.append((relevance, event.utc_timestamp, event))
            selected = sorted(ranked, key=lambda item: (item[0], item[1]), reverse=True)[
                :maximum
            ]
            return tuple(event for _, _, event in sorted(selected, key=lambda item: item[1]))

    async def drain_evicted(self, limit: int = 500) -> tuple[MemoryFragment, ...]:
        """Hand evicted fragments to durable consolidation exactly once."""

        async with self._lock:
            drained: list[MemoryFragment] = []
            for _ in range(min(max(1, limit), len(self._evicted))):
                drained.append(self._evicted.popleft())
            return tuple(drained)

    async def reply_chain(
        self, guild_id: int, channel_id: int, message_id: int
    ) -> tuple[Event, ...]:
        async with self._lock:
            self._prune_locked(self._utc_now())
            message_key = (guild_id, channel_id, message_id)
            root = self._message_roots.get(message_key, message_id)
            chain = self._chains.get((guild_id, channel_id, root))
            if chain is not None:
                return self._resolve(chain)
            event_id = self._message_events.get(message_key)
            event = self._events.get(event_id) if event_id else None
            return (event,) if event else ()

    async def context_charge(self, guild_id: int, user_id: int) -> float:
        async with self._lock:
            now = self._utc_now()
            self._prune_locked(now)
            charge = self._charges.get((guild_id, user_id))
            return self._decayed_value(charge, now) if charge else 0.0

    async def stats(self, guild_id: int) -> MemoryStats:
        async with self._lock:
            self._prune_locked(self._utc_now())
            return MemoryStats(
                events=sum(event.guild_id == guild_id for event in self._events.values()),
                channels=sum(key[0] == guild_id for key in self._channels),
                users=sum(key[0] == guild_id for key in self._users),
                reply_chains=sum(key[0] == guild_id for key in self._chains),
                charged_users=sum(key[0] == guild_id for key in self._charges),
                pending_consolidation=sum(
                    item.event.guild_id == guild_id for item in self._evicted
                ),
            )

    def _observe_reply_chain_locked(self, event: Event) -> None:
        if event.channel_id is None or event.event_type not in {
            EventType.MESSAGE_CREATED,
            EventType.MESSAGE_REPLIED,
        }:
            return
        message_id = self._payload_int(event, "message_id")
        if message_id is None:
            return
        message_key = (event.guild_id, event.channel_id, message_id)
        self._message_events[message_key] = event.event_id
        reply_to = self._payload_int(event, "reply_to_message_id")
        if reply_to is None:
            self._message_roots[message_key] = message_id
            return

        parent_key = (event.guild_id, event.channel_id, reply_to)
        root = self._message_roots.get(parent_key, reply_to)
        chain_key = (event.guild_id, event.channel_id, root)
        chain = self._chains.setdefault(chain_key, deque())
        parent_event_id = self._message_events.get(parent_key)
        if parent_event_id and parent_event_id not in chain:
            chain.append(parent_event_id)
        if event.event_id not in chain:
            chain.append(event.event_id)
        self._sort_and_trim_locked(chain, self.limits.max_chain_events)
        self._message_roots[message_key] = root
        self._chain_updated[chain_key] = event.utc_timestamp
        self._limit_chains_locked(event.guild_id, event.channel_id)

    def _observe_charge_locked(self, event: Event, now: datetime) -> None:
        weight = self._CHARGE_WEIGHTS.get(event.event_type, 0.2)
        observed_at = min(event.utc_timestamp, now)
        if event.actor_id is not None:
            self._add_charge_locked((event.guild_id, event.actor_id), weight, observed_at)
        if event.target_id is not None and event.target_id != event.actor_id:
            self._add_charge_locked((event.guild_id, event.target_id), weight * 0.25, observed_at)
        mentions = event.payload.get("mentioned_user_ids", ())
        if isinstance(mentions, list):
            for user_id in mentions:
                if isinstance(user_id, int) and not isinstance(user_id, bool):
                    self._add_charge_locked((event.guild_id, user_id), weight * 0.25, observed_at)

    def _add_charge_locked(self, key: UserKey, amount: float, at: datetime) -> None:
        existing = self._charges.get(key)
        value = self._decayed_value(existing, at) if existing else 0.0
        self._charges[key] = _Charge(min(self.limits.max_charge, value + amount), at)

    def _prune_locked(self, now: datetime) -> None:
        previous_events = dict(self._events)
        previous_importance = dict(self._importance)
        cutoff = now - self.limits.retention
        retained_ids_in_arrival_order = [
            event_id
            for event_id in self._timeline
            if event_id in self._events and self._events[event_id].utc_timestamp >= cutoff
        ]
        retained_ids_in_arrival_order.sort(
            key=lambda event_id: self._events[event_id].utc_timestamp
        )
        retained = deque(retained_ids_in_arrival_order)
        while len(retained) > self.limits.max_total_events:
            retained.popleft()
        retained_ids = set(retained)
        for event_id, event in previous_events.items():
            if event_id in retained_ids:
                continue
            reason = "expired" if event.utc_timestamp < cutoff else "capacity"
            self._queue_evicted_locked(
                event,
                previous_importance.get(event_id, 0.2),
                reason,
            )
        self._events = {
            event_id: event for event_id, event in self._events.items() if event_id in retained_ids
        }
        self._importance = {
            event_id: value
            for event_id, value in self._importance.items()
            if event_id in retained_ids
        }
        self._timeline = retained
        self._clean_deques_locked(self._channels, self.limits.max_channel_events)
        self._clean_deques_locked(self._users, self.limits.max_user_events)
        self._clean_deques_locked(self._chains, self.limits.max_chain_events)

        for key in list(self._chain_updated):
            if key not in self._chains or self._chain_updated[key] < cutoff:
                self._chains.pop(key, None)
                self._chain_updated.pop(key, None)
        for key, event_id in list(self._message_events.items()):
            if event_id not in self._events:
                self._message_events.pop(key, None)
                self._message_roots.pop(key, None)
        for key, charge in list(self._charges.items()):
            value = self._decayed_value(charge, now)
            if value < 0.01 or charge.updated_at < cutoff:
                self._charges.pop(key, None)

    def _limit_chains_locked(self, guild_id: int, channel_id: int) -> None:
        matching = [key for key in self._chains if key[0] == guild_id and key[1] == channel_id]
        oldest = datetime.min.replace(tzinfo=UTC)
        matching.sort(key=lambda key: self._chain_updated.get(key, oldest))
        for key in matching[: -self.limits.max_reply_chains]:
            self._chains.pop(key, None)
            self._chain_updated.pop(key, None)

    def _clean_deques_locked(self, mapping: dict, maximum: int) -> None:
        for key, values in list(mapping.items()):
            retained = deque(event_id for event_id in values if event_id in self._events)
            self._trim_left(retained, maximum)
            if retained:
                mapping[key] = retained
            else:
                mapping.pop(key, None)

    def _decayed_value(self, charge: _Charge | None, at: datetime) -> float:
        if charge is None:
            return 0.0
        elapsed = max(0.0, (at - charge.updated_at).total_seconds())
        half_life = self.limits.charge_half_life.total_seconds()
        return charge.value * (0.5 ** (elapsed / half_life))

    def _queue_evicted_locked(self, event: Event, importance: float, reason: str) -> None:
        content = event.payload.get("content")
        if (
            event.event_type in {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
            and isinstance(content, str)
            and content.strip()
        ):
            self._evicted.append(MemoryFragment(event, importance, reason))

    @staticmethod
    def _event_importance(event: Event) -> float:
        score = {
            EventType.MESSAGE_CREATED: 0.28,
            EventType.MESSAGE_REPLIED: 0.42,
            EventType.REACTION_ADDED: 0.16,
            EventType.REACTION_REMOVED: 0.12,
            EventType.BOT_REPLIED: 0.34,
            EventType.COMMAND_USED: 0.38,
        }.get(event.event_type, 0.12)
        content = event.payload.get("content")
        if isinstance(content, str):
            normalized = content.casefold()
            pragmatics = analyze_pragmatics(content)
            score += min(0.12, len(tokenize(content)) / 200.0)
            score += 0.10 if content.rstrip().endswith("?") else 0.0
            score += min(0.08, pragmatics.emotional_intensity * 0.08)
            score += 0.04 if pragmatics.question_force >= 0.5 else 0.0
            score += 0.16 if any(
                marker in normalized
                for marker in ("потому", "поэтому", "решили", "важно", "запомни", "вывод")
            ) else 0.0
        score += 0.18 if event.payload.get("bot_mentioned") is True else 0.0
        score += 0.12 if event.payload.get("reply_to_message_id") is not None else 0.0
        return max(0.0, min(1.0, score))

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return set(content_lemmas(text))

    def _resolve(self, event_ids: Iterable[str]) -> tuple[Event, ...]:
        return tuple(self._events[event_id] for event_id in event_ids if event_id in self._events)

    def _sort_and_trim_locked(self, values: deque[str], maximum: int) -> None:
        ordered = sorted(values, key=lambda event_id: self._events[event_id].utc_timestamp)
        values.clear()
        values.extend(ordered[-maximum:])

    def _utc_now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("memory clock must return a timezone-aware datetime")
        return now.astimezone(UTC)

    @staticmethod
    def _trim_left(values: deque[str], maximum: int) -> None:
        while len(values) > maximum:
            values.popleft()

    @staticmethod
    def _payload_int(event: Event, key: str) -> int | None:
        value = event.payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None
