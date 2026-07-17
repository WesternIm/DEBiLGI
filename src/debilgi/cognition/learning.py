"""Shadow learning engine: predict, observe, calculate d, update pattern weights."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from debilgi.cognition.expectations import (
    Expectation,
    ExpectationPredictor,
    OutcomeType,
    Situation,
)
from debilgi.cognition.features import BehaviorFeatureExtractor
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.models import Event, EventType
from debilgi.memory.short_term import ShortTermMemory

logger = logging.getLogger(__name__)

Clock = Callable[[], datetime]
MESSAGE_TYPES = {EventType.MESSAGE_CREATED, EventType.MESSAGE_REPLIED}


@dataclass(frozen=True, slots=True)
class LearningConfig:
    horizon: timedelta = timedelta(minutes=2)
    poll_interval: timedelta = timedelta(seconds=5)

    def __post_init__(self) -> None:
        if self.horizon <= timedelta(0) or self.poll_interval <= timedelta(0):
            raise ValueError("learning intervals must be positive")


class LearningEngine:
    def __init__(
        self,
        repository: WorldModelRepository,
        memory: ShortTermMemory,
        feature_extractor: BehaviorFeatureExtractor,
        predictor: ExpectationPredictor | None = None,
        config: LearningConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.repository = repository
        self._memory = memory
        self._feature_extractor = feature_extractor
        self._predictor = predictor or ExpectationPredictor()
        self.config = config or LearningConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._expiry_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._expiry_task and not self._expiry_task.done():
            return
        await self.expire_overdue(self._utc_now())
        self._expiry_task = asyncio.create_task(
            self._expiry_loop(), name="debilgi-expectation-expiry"
        )

    async def stop(self) -> None:
        if self._expiry_task is None:
            return
        self._expiry_task.cancel()
        await asyncio.gather(self._expiry_task, return_exceptions=True)
        self._expiry_task = None

    async def observe(self, event: Event) -> bool:
        if await self.repository.observation_seen(event.event_id):
            return False

        await self.expire_overdue(event.utc_timestamp)
        expectations = await self._expectations_for_outcome(event)
        for expectation in expectations:
            actual = self._classify_outcome(expectation, event)
            await self.repository.resolve(
                expectation.expectation_id,
                actual,
                resolved_at=event.utc_timestamp,
                resolved_by_event_id=event.event_id,
            )

        if event.event_type in MESSAGE_TYPES and event.channel_id is not None:
            situation = await self._build_situation(event)
            evidence = await self.repository.evidence(event.guild_id)
            prediction = self._predictor.predict(situation, evidence)
            await self.repository.create_expectation(
                event, situation, prediction, self.config.horizon
            )

        await self.repository.mark_observed(event.event_id)
        return True

    async def expire_overdue(self, now: datetime) -> int:
        expired = await self.repository.overdue(self._aware_utc(now))
        resolved = 0
        for expectation in expired:
            result = await self.repository.resolve(
                expectation.expectation_id,
                OutcomeType.NO_RESPONSE,
                resolved_at=expectation.expires_at,
                resolved_by_event_id=None,
            )
            resolved += int(result is not None)
        return resolved

    async def _expectations_for_outcome(self, event: Event) -> list[Expectation]:
        if event.channel_id is None:
            return []
        if event.event_type is EventType.MESSAGE_REPLIED:
            return await self.repository.pending_in_channel(
                event.guild_id, event.channel_id, event.utc_timestamp
            )
        if event.event_type is EventType.REACTION_ADDED:
            message_id = self._payload_int(event, "message_id")
            if message_id is None:
                return []
            expectation = await self.repository.pending_for_message(
                event.guild_id, event.channel_id, message_id, event.utc_timestamp
            )
            return [expectation] if expectation is not None else []
        if event.event_type is EventType.MESSAGE_CREATED:
            return await self.repository.pending_in_channel(
                event.guild_id, event.channel_id, event.utc_timestamp
            )
        return []

    @classmethod
    def _classify_outcome(cls, expectation: Expectation, event: Event) -> OutcomeType:
        if event.event_type is EventType.MESSAGE_REPLIED:
            reply_to = cls._payload_int(event, "reply_to_message_id")
            if reply_to == expectation.trigger_message_id:
                return OutcomeType.REPLY
        if event.event_type is EventType.REACTION_ADDED:
            message_id = cls._payload_int(event, "message_id")
            if message_id == expectation.trigger_message_id:
                return OutcomeType.REACTION
        if event.event_type in MESSAGE_TYPES and event.actor_id == expectation.actor_id:
            return OutcomeType.SAME_ACTOR_MESSAGE
        return OutcomeType.OTHER_MESSAGE

    async def _build_situation(self, event: Event) -> Situation:
        guild_context = await self._memory.guild_context(event.guild_id)
        channel_context = (
            await self._memory.channel_context(event.guild_id, event.channel_id)
            if event.channel_id is not None
            else ()
        )
        actor_features = self._feature_extractor.extract(
            guild_context,
            guild_id=event.guild_id,
            subject_id=event.actor_id or 0,
            now=event.utc_timestamp,
        )
        charge = (
            await self._memory.context_charge(event.guild_id, event.actor_id)
            if event.actor_id is not None
            else 0.0
        )
        content = event.payload.get("content", "")
        content_length = len(content) if isinstance(content, str) else 0
        mentions = event.payload.get("mentioned_user_ids", ())
        return Situation(
            trigger_type=event.event_type.value,
            author_is_bot=event.payload.get("author_is_bot") is True,
            is_reply=self._payload_int(event, "reply_to_message_id") is not None,
            has_mentions=isinstance(mentions, list) and bool(mentions),
            channel_activity=self._bucket(len(channel_context), (2, 8, 20)),
            actor_charge=self._bucket(charge, (0.5, 2.0, 5.0)),
            message_length=self._bucket(content_length, (1, 20, 100)),
            actor_message_rate=self._bucket(actor_features.messages_per_minute, (0.05, 0.25, 1.0)),
            actor_uppercase_ratio=self._bucket(actor_features.uppercase_ratio, (0.1, 0.5, 0.8)),
        )

    async def _expiry_loop(self) -> None:
        seconds = self.config.poll_interval.total_seconds()
        while True:
            await asyncio.sleep(seconds)
            try:
                await self.expire_overdue(self._utc_now())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to expire overdue expectations")

    def _utc_now(self) -> datetime:
        return self._aware_utc(self._clock())

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("learning clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _payload_int(event: Event, key: str) -> int | None:
        value = event.payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    @staticmethod
    def _bucket(value: float, thresholds: tuple[float, float, float]) -> str:
        if value < thresholds[0]:
            return "none"
        if value < thresholds[1]:
            return "low"
        if value < thresholds[2]:
            return "medium"
        return "high"
