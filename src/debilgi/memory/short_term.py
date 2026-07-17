"""Bounded, decaying, guild-isolated short-term event memory."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from debilgi.events.models import Event, EventType

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
    max_charge: float = 10.0

    def __post_init__(self) -> None:
        numeric_limits = (
            self.max_total_events,
            self.max_channel_events,
            self.max_user_events,
            self.max_reply_chains,
            self.max_chain_events,
        )
        if self.retention <= timedelta(0) or self.charge_half_life <= timedelta(0):
            raise ValueError("memory durations must be positive")
        if any(value <= 0 for value in numeric_limits) or self.max_charge <= 0:
            raise ValueError("memory size limits must be positive")


@dataclass(frozen=True, slots=True)
class MemoryStats:
    events: int
    channels: int
    users: int
    reply_chains: int
    charged_users: int


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

    async def observe(self, event: Event) -> bool:
        async with self._lock:
            now = self._utc_now()
            self._prune_locked(now)
            if event.event_id in self._events:
                return False
            if event.utc_timestamp < now - self.limits.retention:
                return False

            self._events[event.event_id] = event
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
        self._events = {
            event_id: event for event_id, event in self._events.items() if event_id in retained_ids
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
