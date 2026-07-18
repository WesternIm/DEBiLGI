"""Neutral, deterministic behavioral measurements over normalized events."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import fmean

from debilgi.events.models import Event, EventType
from debilgi.language.tokenizer import tokenize

MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}
REACTION_TYPES = {EventType.REACTION_ADDED}
BOT_ACTION_TYPES = {EventType.BOT_REPLIED, EventType.BOT_REACTION_ADDED}


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    lookback: timedelta = timedelta(hours=1)
    activity_window: timedelta = timedelta(minutes=2)
    return_gap: timedelta = timedelta(minutes=15)
    scenario_length: int = 3

    def __post_init__(self) -> None:
        if self.lookback <= timedelta(0) or self.activity_window <= timedelta(0):
            raise ValueError("feature windows must be positive")
        if self.return_gap <= timedelta(0) or self.scenario_length <= 1:
            raise ValueError("return gap and scenario length must be meaningful")


@dataclass(frozen=True, slots=True)
class BehaviorFeatures:
    message_count: int
    messages_per_minute: float
    average_message_length: float
    uppercase_ratio: float
    repeated_word_count: int
    duplicate_message_count: int
    replies_received_count: int
    reactions_received_count: int
    longest_reply_thread_seconds: float
    bot_mention_count: int
    repeated_message_after_bot_action_count: int
    command_use_count: int
    reward_request_attempt_count: int
    moderator_deletion_count: int
    average_response_time_seconds: float | None
    activity_delta_after_message: float
    conversation_return_count: int
    repeated_scenario_count: int


class BehaviorFeatureExtractor:
    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def extract(
        self,
        events: Iterable[Event],
        *,
        guild_id: int,
        subject_id: int,
        now: datetime,
        bot_user_id: int | None = None,
    ) -> BehaviorFeatures:
        now = self._aware_utc(now)
        cutoff = now - self.config.lookback
        ordered = sorted(
            (
                event
                for event in events
                if event.guild_id == guild_id and cutoff <= event.utc_timestamp <= now
            ),
            key=lambda event: (event.utc_timestamp, event.event_id),
        )
        messages = [event for event in ordered if event.event_type in MESSAGE_TYPES]
        subject_messages = [event for event in messages if event.actor_id == subject_id]
        message_map = {
            message_id: event
            for event in messages
            if (message_id := self._payload_int(event, "message_id")) is not None
        }

        contents = [self._content(event) for event in subject_messages]
        normalized_messages = [self._normalized_content(content) for content in contents]
        normalized_messages = [content for content in normalized_messages if content]
        all_word_tokens = [
            token
            for content in contents
            for token in tokenize(content)
            if any(character.isalnum() for character in token)
        ]
        word_counts = Counter(all_word_tokens)
        duplicate_counts = Counter(normalized_messages)

        replies_received = self._replies_received(subject_id, messages, message_map)
        reactions_received = self._reactions_received(subject_id, ordered, message_map)
        response_times = self._response_times(subject_id, messages, message_map)

        return BehaviorFeatures(
            message_count=len(subject_messages),
            messages_per_minute=len(subject_messages) / (self.config.lookback.total_seconds() / 60),
            average_message_length=fmean(map(len, contents)) if contents else 0.0,
            uppercase_ratio=self._uppercase_ratio(contents),
            repeated_word_count=sum(count - 1 for count in word_counts.values()),
            duplicate_message_count=sum(count - 1 for count in duplicate_counts.values()),
            replies_received_count=replies_received,
            reactions_received_count=reactions_received,
            longest_reply_thread_seconds=self._longest_thread(subject_id, messages),
            bot_mention_count=self._bot_mentions(subject_messages, bot_user_id=bot_user_id),
            repeated_message_after_bot_action_count=self._repeats_after_bot_action(
                ordered, subject_id, bot_user_id
            ),
            command_use_count=sum(
                event.event_type is EventType.COMMAND_USED and event.actor_id == subject_id
                for event in ordered
            ),
            reward_request_attempt_count=sum(
                event.actor_id == subject_id and event.payload.get("reward_request") is True
                for event in ordered
            ),
            moderator_deletion_count=sum(
                event.event_type is EventType.MESSAGE_DELETED
                and event.target_id == subject_id
                and event.payload.get("deleted_by_moderator") is True
                for event in ordered
            ),
            average_response_time_seconds=(fmean(response_times) if response_times else None),
            activity_delta_after_message=self._activity_delta(subject_messages, messages),
            conversation_return_count=self._conversation_returns(
                subject_id, subject_messages, messages
            ),
            repeated_scenario_count=self._repeated_scenarios(ordered, subject_id),
        )

    @staticmethod
    def _content(event: Event) -> str:
        content = event.payload.get("content", "")
        return content if isinstance(content, str) else ""

    @staticmethod
    def _normalized_content(content: str) -> str:
        return " ".join(tokenize(content))

    @staticmethod
    def _uppercase_ratio(contents: Sequence[str]) -> float:
        cased = [character for content in contents for character in content if character.isalpha()]
        return sum(character.isupper() for character in cased) / len(cased) if cased else 0.0

    @classmethod
    def _replies_received(
        cls, subject_id: int, messages: Sequence[Event], message_map: dict[int, Event]
    ) -> int:
        count = 0
        for event in messages:
            if event.actor_id == subject_id:
                continue
            parent = message_map.get(cls._payload_int(event, "reply_to_message_id") or -1)
            if event.target_id == subject_id or (parent and parent.actor_id == subject_id):
                count += 1
        return count

    @classmethod
    def _reactions_received(
        cls, subject_id: int, events: Sequence[Event], message_map: dict[int, Event]
    ) -> int:
        count = 0
        for event in events:
            if event.event_type not in REACTION_TYPES or event.actor_id == subject_id:
                continue
            message = message_map.get(cls._payload_int(event, "message_id") or -1)
            if event.target_id == subject_id or (message and message.actor_id == subject_id):
                count += 1
        return count

    @classmethod
    def _response_times(
        cls, subject_id: int, messages: Sequence[Event], message_map: dict[int, Event]
    ) -> list[float]:
        times: list[float] = []
        for event in messages:
            if event.actor_id == subject_id:
                continue
            parent = message_map.get(cls._payload_int(event, "reply_to_message_id") or -1)
            if parent and parent.actor_id == subject_id:
                delay = (event.utc_timestamp - parent.utc_timestamp).total_seconds()
                if delay >= 0:
                    times.append(delay)
        return times

    @classmethod
    def _longest_thread(cls, subject_id: int, messages: Sequence[Event]) -> float:
        message_map = {
            message_id: event
            for event in messages
            if (message_id := cls._payload_int(event, "message_id")) is not None
        }
        roots: dict[int, int] = {}

        def root_for(message_id: int, visited: set[int] | None = None) -> int:
            if message_id in roots:
                return roots[message_id]
            visited = visited or set()
            if message_id in visited:
                return message_id
            visited.add(message_id)
            event = message_map.get(message_id)
            parent_id = cls._payload_int(event, "reply_to_message_id") if event else None
            root = root_for(parent_id, visited) if parent_id in message_map else message_id
            roots[message_id] = root
            return root

        chains: dict[int, list[Event]] = defaultdict(list)
        for message_id, event in message_map.items():
            chains[root_for(message_id)].append(event)
        durations = [
            (
                max(item.utc_timestamp for item in chain)
                - min(item.utc_timestamp for item in chain)
            ).total_seconds()
            for chain in chains.values()
            if len(chain) > 1 and any(item.actor_id == subject_id for item in chain)
        ]
        return max(durations, default=0.0)

    @staticmethod
    def _bot_mentions(subject_messages: Sequence[Event], bot_user_id: int | None) -> int:
        count = 0
        for event in subject_messages:
            mentions = event.payload.get("mentioned_user_ids", ())
            explicit = event.payload.get("bot_mentioned") is True
            by_id = (
                bot_user_id is not None and isinstance(mentions, list) and bot_user_id in mentions
            )
            if explicit or by_id:
                count += 1
        return count

    @classmethod
    def _repeats_after_bot_action(
        cls, events: Sequence[Event], subject_id: int, bot_user_id: int | None
    ) -> int:
        last_message_at: dict[str, datetime] = {}
        bot_actions: list[datetime] = []
        repeats = 0
        for event in events:
            is_bot_action = event.event_type in BOT_ACTION_TYPES or (
                bot_user_id is not None and event.actor_id == bot_user_id
            )
            if is_bot_action:
                bot_actions.append(event.utc_timestamp)
            if event.event_type not in MESSAGE_TYPES or event.actor_id != subject_id:
                continue
            normalized = cls._normalized_content(cls._content(event))
            previous = last_message_at.get(normalized)
            action_between = previous and any(
                previous < action < event.utc_timestamp for action in bot_actions
            )
            if normalized and action_between:
                repeats += 1
            if normalized:
                last_message_at[normalized] = event.utc_timestamp
        return repeats

    def _activity_delta(
        self, subject_messages: Sequence[Event], all_messages: Sequence[Event]
    ) -> float:
        if not subject_messages:
            return 0.0
        window = self.config.activity_window.total_seconds()
        deltas: list[int] = []
        for subject_message in subject_messages:
            before = after = 0
            for event in all_messages:
                if event.event_id == subject_message.event_id:
                    continue
                if event.channel_id != subject_message.channel_id:
                    continue
                difference = (event.utc_timestamp - subject_message.utc_timestamp).total_seconds()
                before += int(-window <= difference < 0)
                after += int(0 < difference <= window)
            deltas.append(after - before)
        return fmean(deltas)

    def _conversation_returns(
        self,
        subject_id: int,
        subject_messages: Sequence[Event],
        all_messages: Sequence[Event],
    ) -> int:
        returns = 0
        for previous, current in zip(subject_messages, subject_messages[1:], strict=False):
            if previous.channel_id != current.channel_id:
                continue
            if current.utc_timestamp - previous.utc_timestamp < self.config.return_gap:
                continue
            if any(
                event.actor_id not in {None, subject_id}
                and previous.utc_timestamp < event.utc_timestamp < current.utc_timestamp
                and event.channel_id == current.channel_id
                for event in all_messages
            ):
                returns += 1
        return returns

    def _repeated_scenarios(self, events: Sequence[Event], subject_id: int) -> int:
        related = [
            event.event_type.value
            for event in events
            if event.actor_id == subject_id or event.target_id == subject_id
        ]
        length = self.config.scenario_length
        signatures = Counter(
            tuple(related[index : index + length]) for index in range(len(related) - length + 1)
        )
        return sum(count - 1 for count in signatures.values())

    @staticmethod
    def _payload_int(event: Event | None, key: str) -> int | None:
        if event is None:
            return None
        value = event.payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("feature timestamp must be timezone-aware")
        return value.astimezone(UTC)
