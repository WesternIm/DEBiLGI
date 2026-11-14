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
        retry_delay: float = 0.25,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay < 0:
            raise ValueError("retry_delay must be non-negative")
        self._repository = repository
        self._handlers = tuple(handlers)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._worker_count = worker_count
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._workers: list[asyncio.Task[None]] = []

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def running(self) -> bool:
        return any(not worker.done() for worker in self._workers)

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

    async def stop(self, *, drain: bool = True, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not self._workers:
            return
        try:
            if drain and self.running:
                try:
                    await asyncio.wait_for(self._queue.join(), timeout=timeout)
                except TimeoutError:
                    logger.warning("Event queue drain timed out; unfinished events remain durable")
        finally:
            workers, self._workers = self._workers, []
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            # Recovery reloads these durable IDs on the next start.
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
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
                handler_index = 0
                retries = 0
                while True:
                    try:
                        while handler_index < len(self._handlers):
                            await self._handlers[handler_index](event)
                            handler_index += 1
                        await self._repository.mark_processed(event_id)
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if not await self._repository.retry_processing(
                            event_id, error, self._max_attempts
                        ):
                            raise
                        # Keep the queue item claimed and preserve successful stages.
                        # A restart still requires handlers to be idempotent.
                        logger.warning("Retrying event %s at handler %d", event_id, handler_index)
                        await asyncio.sleep(min(self._retry_delay * (2 ** min(retries, 10)), 5.0))
                        retries += 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Event processing failed for %s", event_id)
                try:
                    await self._repository.mark_failed(event_id, f"{type(exc).__name__}: {exc}")
                except Exception:
                    logger.exception("Failed to record failure for %s; recovery required", event_id)
            finally:
                self._queue.task_done()
