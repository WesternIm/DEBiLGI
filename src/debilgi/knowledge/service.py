"""Background acquisition service that keeps network latency outside the event queue."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from debilgi.cognition.core_repository import CognitiveRepository
from debilgi.events.models import Event
from debilgi.knowledge.models import WikipediaResult
from debilgi.knowledge.policy import KnowledgeQueryPolicy
from debilgi.knowledge.repository import KnowledgeRepository

logger = logging.getLogger(__name__)


class KnowledgeSource(Protocol):
    language: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def search(self, query: str) -> WikipediaResult | None: ...


class KnowledgeAcquisitionService:
    source_name = "wikipedia"

    def __init__(
        self,
        repository: KnowledgeRepository,
        cognitive_repository: CognitiveRepository,
        policy: KnowledgeQueryPolicy,
        source: KnowledgeSource,
        *,
        reply_wait_seconds: float = 1.8,
        queue_size: int = 100,
    ) -> None:
        self.repository = repository
        self.cognitive_repository = cognitive_repository
        self.policy = policy
        self.source = source
        self.reply_wait_seconds = reply_wait_seconds
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=queue_size)
        self._worker: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        return self.policy.enabled

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        if not self.enabled or self._worker is not None:
            return
        await self.source.start()
        self._worker = asyncio.create_task(self._run(), name="debilgi-wikipedia-worker")

    async def stop(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=3.0)
            except TimeoutError:
                logger.warning("Wikipedia queue did not drain before shutdown")
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        await self.source.close()

    async def observe(self, event: Event) -> None:
        if not self.enabled:
            return
        await self.repository.observe(event)
        await self.repository.remember_event_concepts(event)
        intent = await self.cognitive_repository.intent_for_event(event.event_id)
        decision = self.policy.decide(event, intent)
        if not decision.should_query or decision.query is None or decision.reason is None:
            return
        cached = await self.repository.concept_for_alias(decision.query)
        if cached is not None:
            await self.repository.record_cache_hit(
                event,
                query=decision.query,
                reason=decision.reason,
                uncertainty=decision.uncertainty,
                concept_id=cached.concept_id,
                source=self.source_name,
                language=self.source.language,
            )
            return
        query_id = await self.repository.reserve_query(
            event,
            query=decision.query,
            reason=decision.reason,
            uncertainty=decision.uncertainty,
            source=self.source_name,
            language=self.source.language,
        )
        if query_id is None:
            return
        try:
            self._queue.put_nowait((query_id, decision.query))
        except asyncio.QueueFull:
            await self.repository.finish_without_result(query_id, error="acquisition queue is full")

    async def wait_for_event(self, event_id: str) -> str | None:
        if not self.enabled:
            return None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.reply_wait_seconds
        while True:
            status = await self.repository.query_status_for_event(event_id)
            if status is None or status != "pending":
                return status
            remaining = deadline - loop.time()
            if remaining <= 0:
                return status
            await asyncio.sleep(min(0.04, remaining))

    async def context_for_text(self, value: str) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        concepts = await self.repository.concepts_for_text(value)
        return tuple(
            f"{concept.title} — {concept.definition[:500]}"
            for concept in concepts
            if concept.definition
        )

    async def _run(self) -> None:
        while True:
            query_id, query = await self._queue.get()
            try:
                result = await self.source.search(query)
                if result is None or not result.extract.strip():
                    await self.repository.finish_without_result(query_id)
                else:
                    await self.repository.complete_query(
                        query_id,
                        result,
                        source=self.source_name,
                        language=self.source.language,
                    )
                    logger.info("Learned Wikipedia concept %r", result.title)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Wikipedia lookup failed for %r: %s", query, exc)
                await self.repository.finish_without_result(
                    query_id, error=f"{type(exc).__name__}: {exc}"
                )
            finally:
                self._queue.task_done()
