"""Orchestration for Cognitive Core v0.2 in strictly shadow-only mode."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import perf_counter

from debilgi.actions.decision import ShadowDecisionEngine
from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.cognition.difference import difference_vector
from debilgi.cognition.models import (
    ActionIntent,
    EntityModel,
    EntityType,
    Episode,
    InternalState,
    SemanticMemory,
)
from debilgi.cognition.world_model import WorldModelRepository
from debilgi.events.models import Event

logger = logging.getLogger(__name__)
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class CognitiveCoreConfig:
    episode_gap: timedelta = timedelta(minutes=10)
    maintenance_interval: timedelta = timedelta(minutes=1)
    raw_text_retention: timedelta = timedelta(days=30)
    consolidation_batch_size: int = 100

    def __post_init__(self) -> None:
        if (
            self.episode_gap <= timedelta(0)
            or self.maintenance_interval <= timedelta(0)
            or self.raw_text_retention <= timedelta(0)
            or self.consolidation_batch_size <= 0
        ):
            raise ValueError("cognitive core intervals and batch size must be positive")


@dataclass(frozen=True, slots=True)
class CognitiveSnapshot:
    episodes: tuple[Episode, ...]
    semantic_memories: tuple[SemanticMemory, ...]
    entity_model: EntityModel | None
    internal_state: InternalState | None
    intents: tuple[ActionIntent, ...]
    differences: tuple[dict[str, object], ...]
    average_processing_ms: float | None


class CognitiveCore:
    def __init__(
        self,
        repository: CognitiveRepository,
        world_model: WorldModelRepository,
        decision_engine: ShadowDecisionEngine | None = None,
        config: CognitiveCoreConfig | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.repository = repository
        self._world_model = world_model
        self._decision_engine = decision_engine or ShadowDecisionEngine()
        self.config = config or CognitiveCoreConfig()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._maintenance_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._maintenance_task and not self._maintenance_task.done():
            return
        await self.maintenance_once(self._utc_now())
        self._maintenance_task = asyncio.create_task(
            self._maintenance_loop(), name="debilgi-cognitive-maintenance"
        )

    async def stop(self) -> None:
        if self._maintenance_task is None:
            return
        self._maintenance_task.cancel()
        await asyncio.gather(self._maintenance_task, return_exceptions=True)
        self._maintenance_task = None

    async def observe(self, event: Event) -> bool:
        if not await self.repository.claim_observation(event.event_id):
            return False
        started = perf_counter()
        try:
            resolutions = await self._world_model.resolved_by_event(event.event_id)
            vectors = [difference_vector(expectation) for expectation in resolutions]
            for expectation, vector in zip(resolutions, vectors, strict=True):
                await self.repository.record_difference(expectation, vector)
            unexpectedness = max((vector.unexpectedness for vector in vectors), default=0.0)
            actual_outcomes = tuple(
                expectation.actual_outcome
                for expectation in resolutions
                if expectation.actual_outcome is not None
            )
            reply_delays = tuple(
                max(0.0, (expectation.resolved_at - expectation.created_at).total_seconds())
                for expectation in resolutions
                if expectation.actual_outcome is not None
                and expectation.actual_outcome.value == "reply"
                and expectation.resolved_at is not None
            )
            stats = await self._world_model.stats(event.guild_id)
            observed_outcome = actual_outcomes[-1].value if actual_outcomes else None
            episode = await self.repository.attach_event(
                event,
                unexpectedness=unexpectedness,
                episode_gap=self.config.episode_gap,
                observed_outcome=observed_outcome,
            )
            await self.repository.mark_raw_retention(event, self.config.raw_text_retention)
            await self.repository.evaluate_prior_intents(event)
            model = await self.repository.update_entity_model(
                event,
                unexpectedness=unexpectedness,
                active_expectations=stats.pending_expectations,
                actual_outcomes=actual_outcomes,
                reply_delays=reply_delays,
            )
            state = await self.repository.update_internal_state(
                event,
                model,
                unexpectedness=unexpectedness,
                active_expectations=stats.pending_expectations,
                importance=episode.importance,
            )
            assessment = await self.repository.create_assessment(event, model, episode)
            evidence = (
                event.event_id,
                episode.episode_id,
                assessment.assessment_id,
                *(expectation.expectation_id for expectation in resolutions),
            )
            intent = self._decision_engine.decide(
                event,
                model,
                state,
                active_expectations=stats.pending_expectations,
                evidence=evidence,
            )
            await self.repository.save_intent(intent)
            await self.repository.finish_observation(
                event.event_id, (perf_counter() - started) * 1000.0
            )
            return True
        except Exception as exc:
            await self.repository.fail_observation(event.event_id, f"{type(exc).__name__}: {exc}")
            raise

    async def maintenance_once(self, now: datetime) -> dict[str, int]:
        now = self._aware_utc(now)
        unvectorized = await self._world_model.unvectorized_resolved(
            self.config.consolidation_batch_size
        )
        for expectation in unvectorized:
            await self.repository.record_difference(expectation, difference_vector(expectation))
        closed = await self.repository.close_stale_episodes(now, self.config.episode_gap)
        consolidated = await self.repository.consolidate(limit=self.config.consolidation_batch_size)
        decayed = await self.repository.decay_semantic(now)
        backfilled = await self.repository.backfill_raw_retention(
            self.config.raw_text_retention, self.config.consolidation_batch_size
        )
        redacted = await self.repository.redact_expired_raw_text(
            now, self.config.consolidation_batch_size
        )
        return {
            "differences": len(unvectorized),
            "closed_episodes": closed,
            "consolidated_episodes": consolidated,
            "decayed_memories": decayed,
            "retention_backfilled": backfilled,
            "raw_texts_redacted": redacted,
        }

    async def snapshot(
        self, guild_id: int, *, user_id: int | None = None, limit: int = 10
    ) -> CognitiveSnapshot:
        entity_model = (
            await self.repository.entity_model(guild_id, EntityType.USER, user_id)
            if user_id is not None
            else await self.repository.entity_model(guild_id, EntityType.SERVER, guild_id)
        )
        state = await self.repository.internal_state(guild_id, user_id or 0)
        return CognitiveSnapshot(
            episodes=tuple(await self.repository.recent_episodes(guild_id, limit)),
            semantic_memories=tuple(await self.repository.semantic_memories(guild_id, limit=limit)),
            entity_model=entity_model,
            internal_state=state,
            intents=tuple(await self.repository.recent_intents(guild_id, limit)),
            differences=tuple(await self.repository.recent_differences(guild_id, limit)),
            average_processing_ms=await self.repository.average_processing_ms(guild_id),
        )

    async def _maintenance_loop(self) -> None:
        delay = self.config.maintenance_interval.total_seconds()
        while True:
            await asyncio.sleep(delay)
            try:
                await self.maintenance_once(self._utc_now())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Cognitive maintenance failed")

    def _utc_now(self) -> datetime:
        return self._aware_utc(self._clock())

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cognitive core clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
