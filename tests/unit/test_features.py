from datetime import UTC, datetime, timedelta

import pytest

from debilgi.cognition.features import BehaviorFeatureExtractor, FeatureConfig
from debilgi.events.models import Event, EventType

BASE = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)
NOW = BASE + timedelta(minutes=5)
SUBJECT = 100
OTHER = 200
BOT = 999


def event(
    name: str,
    offset: float,
    *,
    event_type: EventType = EventType.MESSAGE_CREATED,
    actor_id: int | None = SUBJECT,
    target_id: int | None = None,
    channel_id: int = 10,
    message_id: int | None = None,
    reply_to: int | None = None,
    content: str = "",
    **payload: object,
) -> Event:
    if message_id is not None:
        payload.update(
            message_id=message_id,
            reply_to_message_id=reply_to,
            content=content,
        )
    return Event(
        event_id=name,
        guild_id=1,
        channel_id=channel_id,
        actor_id=actor_id,
        target_id=target_id,
        event_type=event_type,
        timestamp=BASE + timedelta(seconds=offset),
        payload=payload,
        source="test",
    )


def extract(events: list[Event], *, config: FeatureConfig | None = None):
    return BehaviorFeatureExtractor(config).extract(
        events,
        guild_id=1,
        subject_id=SUBJECT,
        now=NOW,
        bot_user_id=BOT,
    )


def test_message_count_counts_only_subject_messages() -> None:
    measured = extract(
        [
            event("subject-1", 0, message_id=1),
            event("other", 1, actor_id=OTHER, message_id=2),
            event("subject-2", 2, message_id=3),
        ]
    )

    assert measured.message_count == 2


def test_message_rate_uses_configured_lookback_window() -> None:
    measured = extract(
        [event("one", 0, message_id=1), event("two", 1, message_id=2)],
        config=FeatureConfig(lookback=timedelta(minutes=10)),
    )

    assert measured.messages_per_minute == pytest.approx(0.2)


def test_average_message_length_uses_raw_character_count() -> None:
    measured = extract(
        [
            event("short", 0, message_id=1, content="abc"),
            event("long", 1, message_id=2, content="12345"),
        ]
    )

    assert measured.average_message_length == 4.0


def test_uppercase_ratio_counts_only_letters_without_interpreting_them() -> None:
    measured = extract([event("caps", 0, message_id=1, content="ABcd Яя 123")])

    assert measured.uppercase_ratio == pytest.approx(0.5)


def test_repeated_word_count_counts_occurrences_beyond_first() -> None:
    measured = extract(
        [
            event("first", 0, message_id=1, content="да да нет"),
            event("second", 1, message_id=2, content="да"),
        ]
    )

    assert measured.repeated_word_count == 2


def test_duplicate_message_count_uses_normalized_exact_text() -> None:
    measured = extract(
        [
            event("first", 0, message_id=1, content="Привет!"),
            event("second", 1, message_id=2, content="привет !"),
        ]
    )

    assert measured.duplicate_message_count == 1


def test_replies_received_can_be_inferred_from_parent_message() -> None:
    measured = extract(
        [
            event("parent", 0, message_id=1, content="вопрос"),
            event(
                "reply",
                10,
                event_type=EventType.MESSAGE_REPLIED,
                actor_id=OTHER,
                message_id=2,
                reply_to=1,
                content="ответ",
            ),
        ]
    )

    assert measured.replies_received_count == 1


def test_reactions_received_counts_additions_but_not_removals() -> None:
    measured = extract(
        [
            event("message", 0, message_id=1),
            event(
                "add",
                1,
                event_type=EventType.REACTION_ADDED,
                actor_id=OTHER,
                target_id=SUBJECT,
                message_id=1,
            ),
            event(
                "remove",
                2,
                event_type=EventType.REACTION_REMOVED,
                actor_id=OTHER,
                target_id=SUBJECT,
                message_id=1,
            ),
        ]
    )

    assert measured.reactions_received_count == 1


def test_longest_reply_thread_measures_elapsed_seconds() -> None:
    measured = extract(
        [
            event("parent", 0, message_id=1),
            event(
                "reply",
                30,
                event_type=EventType.MESSAGE_REPLIED,
                actor_id=OTHER,
                message_id=2,
                reply_to=1,
            ),
        ]
    )

    assert measured.longest_reply_thread_seconds == 30.0


def test_bot_mentions_require_explicit_gateway_observation() -> None:
    measured = extract(
        [
            event("mention", 0, message_id=1, bot_mentioned=True),
            event("plain", 1, message_id=2, bot_mentioned=False),
        ]
    )

    assert measured.bot_mention_count == 1


def test_repeated_message_after_bot_action_is_structural_not_semantic() -> None:
    measured = extract(
        [
            event("first", 0, message_id=1, content="верни"),
            event("bot", 10, event_type=EventType.BOT_REPLIED, actor_id=BOT),
            event("repeat", 20, message_id=2, content="верни"),
        ]
    )

    assert measured.repeated_message_after_bot_action_count == 1


def test_command_use_count_uses_explicit_command_events() -> None:
    measured = extract(
        [
            event("command-1", 0, event_type=EventType.COMMAND_USED),
            event("command-2", 1, event_type=EventType.COMMAND_USED),
            event("other", 2, event_type=EventType.COMMAND_USED, actor_id=OTHER),
        ]
    )

    assert measured.command_use_count == 2


def test_reward_attempt_requires_explicit_intent_flag() -> None:
    measured = extract(
        [
            event(
                "explicit",
                0,
                event_type=EventType.COMMAND_USED,
                reward_request=True,
            ),
            event("ordinary", 1, event_type=EventType.COMMAND_USED),
        ]
    )

    assert measured.reward_request_attempt_count == 1


def test_moderator_deletion_requires_confirmed_flag() -> None:
    measured = extract(
        [
            event(
                "confirmed",
                0,
                event_type=EventType.MESSAGE_DELETED,
                actor_id=None,
                target_id=SUBJECT,
                deleted_by_moderator=True,
            ),
            event(
                "unknown",
                1,
                event_type=EventType.MESSAGE_DELETED,
                actor_id=None,
                target_id=SUBJECT,
                deleted_by_moderator=None,
            ),
        ]
    )

    assert measured.moderator_deletion_count == 1


def test_average_response_time_uses_replies_to_subject_messages() -> None:
    measured = extract(
        [
            event("parent", 0, message_id=1),
            event(
                "reply-1",
                20,
                event_type=EventType.MESSAGE_REPLIED,
                actor_id=OTHER,
                message_id=2,
                reply_to=1,
            ),
            event(
                "reply-2",
                40,
                event_type=EventType.MESSAGE_REPLIED,
                actor_id=OTHER,
                message_id=3,
                reply_to=1,
            ),
        ]
    )

    assert measured.average_response_time_seconds == 30.0


def test_activity_delta_compares_equal_windows_before_and_after() -> None:
    config = FeatureConfig(activity_window=timedelta(seconds=10))
    measured = extract(
        [
            event("before-1", 91, actor_id=OTHER, message_id=1),
            event("before-2", 95, actor_id=OTHER, message_id=2),
            event("subject", 100, message_id=3),
            event("after", 105, actor_id=OTHER, message_id=4),
        ],
        config=config,
    )

    assert measured.activity_delta_after_message == -1.0


def test_conversation_return_requires_gap_and_intervening_participant() -> None:
    config = FeatureConfig(return_gap=timedelta(seconds=60))
    measured = extract(
        [
            event("leaves", 0, message_id=1),
            event("other", 30, actor_id=OTHER, message_id=2),
            event("returns", 120, message_id=3),
        ],
        config=config,
    )

    assert measured.conversation_return_count == 1


def test_repeated_scenario_counts_repeated_event_type_sequences() -> None:
    config = FeatureConfig(scenario_length=2)
    measured = extract(
        [
            event("m1", 0, message_id=1),
            event("r1", 1, event_type=EventType.REACTION_ADDED, target_id=SUBJECT),
            event("m2", 2, message_id=2),
            event("r2", 3, event_type=EventType.REACTION_ADDED, target_id=SUBJECT),
        ],
        config=config,
    )

    assert measured.repeated_scenario_count == 1


def test_empty_context_returns_neutral_zero_measurements() -> None:
    measured = extract([])

    assert measured.message_count == 0
    assert measured.uppercase_ratio == 0.0
    assert measured.average_response_time_seconds is None
