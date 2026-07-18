"""Durable event queue with recovery, replay, and duplicate-processing protection."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable

from debilgi.events.models import Event
from debilgi.events.repository import EventRepository, ProcessingStatus

EventHandler = Callable[[Event], Awaitable[None]]

logger = logging.getLogger(__name__)


class EventPipeline:
    def __init__(
        self,
        repository: EventRepository,
        handlers: Iterable[EventHandler] = (),
        *,
        queue_size: int = 1000,
        worker_count: int = 1,
        max_attempts: int = 3,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self._repository = repository
        self._handlers = tuple(handlers)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._worker_count = worker_count
        self._max_attempts = max_attempts
        self._workers: list[asyncio.Task[None]] = []

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return bool(self._workers)

    async def start(self) -> None:
        if self.running:
            return
        pending_ids = await self._repository.recover_unfinished(self._max_attempts)
        self._workers = [
            asyncio.create_task(self._worker(), name=f"debilgi-event-worker-{index}")
            for index in range(self._worker_count)
        ]
        for event_id in pending_ids:
            await self._queue.put(event_id)
        logger.info("Event pipeline started with %d recovered events", len(pending_ids))

    async def stop(self, *, drain: bool = True) -> None:
        if not self.running:
            return
        if drain:
            await self._queue.join()
        workers, self._workers = self._workers, []
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        logger.info("Event pipeline stopped")

    async def publish(self, event: Event) -> bool:
        inserted = await self._repository.append(event)
        status = await self._repository.processing_status(event.event_id)
        if inserted or status is ProcessingStatus.PENDING:
            await self._queue.put(event.event_id)
        return inserted

    async def replay(self, guild_id: int, limit: int = 100) -> int:
        event_ids = await self._repository.prepare_replay(guild_id, limit)
        for event_id in event_ids:
            await self._queue.put(event_id)
        return len(event_ids)

    async def join(self) -> None:
        await self._queue.join()

    async def wait_until_processed(self, event_id: str, *, timeout: float = 5.0) -> bool:
        """Wait for one event without blocking on unrelated queued observations."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            status = await self._repository.processing_status(event_id)
            if status is ProcessingStatus.PROCESSED:
                return True
            if status is ProcessingStatus.FAILED:
                return False
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.01, remaining))

    async def _worker(self) -> None:
        while True:
            event_id = await self._queue.get()
            try:
                claimed = await self._repository.claim_for_processing(event_id)
                if not claimed:
                    continue
                event = await self._repository.get(event_id)
                if event is None:
                    raise RuntimeError(f"Queued event does not exist: {event_id}")
                for handler in self._handlers:
                    await handler(event)
                await self._repository.mark_processed(event_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Event processing failed for %s", event_id)
                await self._repository.mark_failed(event_id, f"{type(exc).__name__}: {exc}")
            finally:
                self._queue.task_done()
