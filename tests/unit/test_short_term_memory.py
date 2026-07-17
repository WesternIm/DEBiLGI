from datetime import UTC, datetime, timedelta

import pytest

from debilgi.events.models import Event, EventType
from debilgi.memory.short_term import MemoryLimits, ShortTermMemory


class ManualClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


def make_event(
    event_id: str,
    clock: ManualClock,
    *,
    guild_id: int = 1,
    channel_id: int = 10,
    actor_id: int = 100,
    event_type: EventType = EventType.MESSAGE_CREATED,
    message_id: int | None = None,
    reply_to: int | None = None,
    mentions: list[int] | None = None,
) -> Event:
    payload: dict[str, object] = {}
    if message_id is not None:
        payload["message_id"] = message_id
        payload["reply_to_message_id"] = reply_to
    if mentions is not None:
        payload["mentioned_user_ids"] = mentions
    return Event(
        event_id=event_id,
        guild_id=guild_id,
        channel_id=channel_id,
        actor_id=actor_id,
        target_id=None,
        event_type=event_type,
        timestamp=clock.now,
        payload=payload,
        source="test",
    )


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock(datetime(2026, 7, 17, 12, 0, tzinfo=UTC))


async def test_contexts_are_isolated_by_guild_channel_and_user(clock: ManualClock) -> None:
    memory = ShortTermMemory(clock=clock)
    events = [
        make_event("g1-c1", clock),
        make_event("g1-c2", clock, channel_id=20),
        make_event("g2-c1", clock, guild_id=2),
    ]
    for event in events:
        await memory.observe(event)

    assert [event.event_id for event in await memory.channel_context(1, 10)] == ["g1-c1"]
    assert [event.event_id for event in await memory.channel_context(1, 20)] == ["g1-c2"]
    assert [event.event_id for event in await memory.user_context(1, 100)] == [
        "g1-c1",
        "g1-c2",
    ]
    assert [event.event_id for event in await memory.user_context(2, 100)] == ["g2-c1"]


async def test_reply_chain_tracks_root_branches_and_nested_replies(clock: ManualClock) -> None:
    memory = ShortTermMemory(clock=clock)
    messages = [
        make_event("parent", clock, message_id=100),
        make_event(
            "reply-a",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=101,
            reply_to=100,
        ),
        make_event(
            "reply-b",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=102,
            reply_to=100,
        ),
        make_event(
            "nested",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=103,
            reply_to=101,
        ),
    ]
    for event in messages:
        await memory.observe(event)

    chain = await memory.reply_chain(1, 10, 103)
    assert [event.event_id for event in chain] == ["parent", "reply-a", "reply-b", "nested"]


async def test_size_limits_and_duplicate_protection(clock: ManualClock) -> None:
    memory = ShortTermMemory(
        MemoryLimits(
            max_total_events=3,
            max_channel_events=2,
            max_user_events=2,
        ),
        clock=clock,
    )
    events = [make_event(f"event-{index}", clock) for index in range(4)]
    for event in events:
        await memory.observe(event)

    assert not await memory.observe(events[-1])
    assert [event.event_id for event in await memory.channel_context(1, 10)] == [
        "event-2",
        "event-3",
    ]
    assert [event.event_id for event in await memory.user_context(1, 100)] == [
        "event-2",
        "event-3",
    ]
    assert (await memory.stats(1)).events == 3


async def test_charge_decays_by_half_life_and_expires_with_context(clock: ManualClock) -> None:
    memory = ShortTermMemory(
        MemoryLimits(
            retention=timedelta(minutes=10),
            charge_half_life=timedelta(minutes=5),
        ),
        clock=clock,
    )
    await memory.observe(make_event("charged", clock, mentions=[200]))

    assert await memory.context_charge(1, 100) == pytest.approx(1.0)
    assert await memory.context_charge(1, 200) == pytest.approx(0.25)
    clock.advance(timedelta(minutes=5))
    assert await memory.context_charge(1, 100) == pytest.approx(0.5)
    clock.advance(timedelta(minutes=6))
    assert await memory.channel_context(1, 10) == ()
    assert await memory.context_charge(1, 100) == 0.0


async def test_rebuild_is_deterministic(clock: ManualClock) -> None:
    memory = ShortTermMemory(clock=clock)
    later = make_event("later", clock)
    clock.now -= timedelta(seconds=1)
    earlier = make_event("earlier", clock)
    clock.now += timedelta(seconds=1)

    await memory.rebuild([later, earlier, later])

    assert [event.event_id for event in await memory.channel_context(1, 10)] == [
        "earlier",
        "later",
    ]


async def test_reply_chain_count_and_length_are_bounded(clock: ManualClock) -> None:
    memory = ShortTermMemory(MemoryLimits(max_reply_chains=1, max_chain_events=2), clock=clock)
    events = [
        make_event("parent-1", clock, message_id=10),
        make_event(
            "reply-1a",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=11,
            reply_to=10,
        ),
        make_event(
            "reply-1b",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=12,
            reply_to=10,
        ),
        make_event("parent-2", clock, message_id=20),
        make_event(
            "reply-2",
            clock,
            event_type=EventType.MESSAGE_REPLIED,
            message_id=21,
            reply_to=20,
        ),
    ]
    for event in events:
        await memory.observe(event)

    assert (await memory.stats(1)).reply_chains == 1
    assert [event.event_id for event in await memory.reply_chain(1, 10, 21)] == [
        "parent-2",
        "reply-2",
    ]
