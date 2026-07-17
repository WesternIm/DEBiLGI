from datetime import UTC, datetime

from debilgi.events.models import EventType
from debilgi.events.normalizer import (
    normalize_command_used,
    normalize_message_created,
    normalize_message_deleted,
    normalize_reaction,
)


def test_message_normalization_is_deterministic_and_discord_independent() -> None:
    created_at = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    event = normalize_message_created(
        message_id=100,
        guild_id=200,
        channel_id=300,
        author_id=400,
        created_at=created_at,
        content="бля а я падумал 🤔",
        author_is_bot=False,
        mentioned_user_ids=[500],
        attachments=[{"filename": "x.png", "size": 10}],
    )

    assert event.event_id == "discord:message:100"
    assert event.event_type is EventType.MESSAGE_CREATED
    assert event.timestamp == created_at
    assert event.payload["content"] == "бля а я падумал 🤔"
    assert event.payload["mentioned_user_ids"] == [500]


def test_reply_becomes_message_replied() -> None:
    event = normalize_message_created(
        message_id=101,
        guild_id=200,
        channel_id=300,
        author_id=400,
        created_at=datetime.now(UTC),
        content="ответ",
        author_is_bot=False,
        reply_to_message_id=99,
        reply_to_author_id=500,
        bot_mentioned=True,
    )

    assert event.event_type is EventType.MESSAGE_REPLIED
    assert event.target_id == 500
    assert event.payload["reply_to_message_id"] == 99
    assert event.payload["bot_mentioned"] is True


def test_reaction_normalization_records_direction_and_message() -> None:
    observed_at = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    added = normalize_reaction(
        added=True,
        guild_id=1,
        channel_id=2,
        message_id=3,
        actor_id=4,
        target_id=5,
        emoji="💩",
        observed_at=observed_at,
    )
    removed = normalize_reaction(
        added=False,
        guild_id=1,
        channel_id=2,
        message_id=3,
        actor_id=4,
        emoji="💩",
        observed_at=observed_at,
    )

    assert added.event_type is EventType.REACTION_ADDED
    assert added.target_id == 5
    assert removed.event_type is EventType.REACTION_REMOVED
    assert added.payload["message_id"] == 3
    assert added.event_id != removed.event_id


def test_reaction_identity_is_reproducible_and_operation_specific() -> None:
    arguments = {
        "guild_id": 10,
        "channel_id": 20,
        "message_id": 30,
        "actor_id": 40,
        "target_id": 50,
        "emoji": "💅",
        "observed_at": datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    }

    first = normalize_reaction(added=True, **arguments)
    duplicate = normalize_reaction(added=True, **arguments)
    removed = normalize_reaction(added=False, **arguments)

    assert first.event_id == duplicate.event_id
    assert removed.event_id != first.event_id
    assert ":add:" in first.event_id
    assert ":remove:" in removed.event_id


def test_command_normalization_records_explicit_command_name() -> None:
    observed_at = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)

    event = normalize_command_used(
        interaction_id=10,
        guild_id=20,
        channel_id=30,
        actor_id=40,
        command_name="debilgi memory",
        observed_at=observed_at,
    )

    assert event.event_type is EventType.COMMAND_USED
    assert event.payload["command_name"] == "debilgi memory"


def test_raw_deletion_does_not_invent_moderator_identity() -> None:
    event = normalize_message_deleted(
        message_id=10,
        guild_id=20,
        channel_id=30,
        author_id=40,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=UTC),
    )

    assert event.event_type is EventType.MESSAGE_DELETED
    assert event.target_id == 40
    assert event.payload["deleted_by_moderator"] is None
