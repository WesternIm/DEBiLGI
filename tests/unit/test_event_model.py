from datetime import UTC, datetime

import pytest

from debilgi.events.models import Event, EventType


def make_event(timestamp: datetime) -> Event:
    return Event(
        event_id="test:1",
        guild_id=1,
        channel_id=2,
        actor_id=3,
        target_id=None,
        event_type=EventType.MESSAGE_CREATED,
        timestamp=timestamp,
        payload={"message_id": 4},
        source="test",
    )


def test_event_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        make_event(datetime(2026, 1, 1))


def test_event_exposes_utc_timestamp() -> None:
    event = make_event(datetime(2026, 1, 1, tzinfo=UTC))

    assert event.utc_timestamp == datetime(2026, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize("guild_id", [0, -1])
def test_event_requires_positive_guild_id(guild_id: int) -> None:
    with pytest.raises(ValueError, match="guild_id"):
        Event(
            event_id="test:1",
            guild_id=guild_id,
            channel_id=None,
            actor_id=None,
            target_id=None,
            event_type=EventType.MESSAGE_CREATED,
            timestamp=datetime.now(UTC),
            payload={},
            source="test",
        )
