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
        post_handlers: Iterable[EventHandler] = (),
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
        self._post_handlers = tuple(post_handlers)
        self._ready: dict[str, bool] = {}
        self._completed: dict[str, bool] = {}
        self._signals: dict[tuple[str, str], asyncio.Event] = {}
        self._responses: dict[str, asyncio.Event] = {}
        self._waiters: dict[tuple[str, str], int] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_size)
        self._post_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size)
        self._worker_count = worker_count
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._workers: list[asyncio.Task[None]] = []
        self._post_workers: list[asyncio.Task[None]] = []

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
        self._post_workers = [
            asyncio.create_task(self._post_worker(), name=f"debilgi-learning-worker-{index}")
            for index in range(self._worker_count)
        ] if self._post_handlers else []
        for event_id in pending_ids:
            self._ready.pop(event_id, None)
            self._completed.pop(event_id, None)
            await self._queue.put(event_id)
        logger.info("Event pipeline started with %d recovered events", len(pending_ids))

    async def stop(self, *, drain: bool = True, timeout: float = 30.0) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if not self._workers and not self._post_workers:
            return
        try:
            if drain and self.running:
                try:
                    await asyncio.wait_for(self.join(), timeout=timeout)
                except TimeoutError:
                    logger.warning("Event queue drain timed out; unfinished events remain durable")
        finally:
            workers = [*self._workers, *self._post_workers]
            self._workers, self._post_workers = [], []
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            # Recovery reloads these durable IDs on the next start.
            while not self._queue.empty():
                self._queue.get_nowait()
                self._queue.task_done()
            while not self._post_queue.empty():
                event = self._post_queue.get_nowait()
                self._notify(event.event_id, "processed", False)
                self._responses.pop(event.event_id, None)
                self._post_queue.task_done()
        logger.info("Event pipeline stopped")

    async def publish(self, event: Event) -> bool:
        self._ready.pop(event.event_id, None)
        self._completed.pop(event.event_id, None)
        inserted = await self._repository.append(event)
        status = await self._repository.processing_status(event.event_id)
        if inserted or status is ProcessingStatus.PENDING:
            await self._queue.put(event.event_id)
        return inserted

    async def replay(self, guild_id: int, limit: int = 100) -> int:
        event_ids = await self._repository.prepare_replay(guild_id, limit)
        for event_id in event_ids:
            self._ready.pop(event_id, None)
            self._completed.pop(event_id, None)
            await self._queue.put(event_id)
        return len(event_ids)

    async def join(self) -> None:
        await self._queue.join()
        await self._post_queue.join()

    async def wait_until_processed(self, event_id: str, *, timeout: float = 5.0) -> bool:
        """Wait for one event without blocking on unrelated queued observations."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        return await self._wait_for(event_id, "processed", timeout)

    async def wait_until_ready(self, event_id: str, *, timeout: float = 5.0) -> bool:
        """Only projections consumed by this response precede this signal."""
        self._responses.setdefault(event_id, asyncio.Event())
        return await self._wait_for(event_id, "ready", timeout)

    def finish_response(self, event_id: str) -> None:
        signal = self._responses.pop(event_id, None)
        if signal is not None:
            signal.set()

    def _notify(self, event_id: str, phase: str, success: bool) -> None:
        results = self._ready if phase == "ready" else self._completed
        results[event_id] = success
        if len(results) > 2048:
            results.pop(next(iter(results)))
        signal = self._signals.get((event_id, phase))
        if signal is not None:
            signal.set()

    async def _wait_for(self, event_id: str, phase: str, timeout: float) -> bool:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        results = self._ready if phase == "ready" else self._completed
        key = (event_id, phase)
        signal = self._signals.setdefault(key, asyncio.Event())
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            if event_id in results:
                return results[event_id]
            status = await self._repository.processing_status(event_id)
            if status in {ProcessingStatus.PROCESSED, ProcessingStatus.FAILED}:
                return status is ProcessingStatus.PROCESSED
            if event_id not in results:
                await asyncio.wait_for(signal.wait(), timeout)
            return results.get(event_id, False)
        except TimeoutError:
            return False
        finally:
            self._waiters[key] -= 1
            if not self._waiters[key]:
                self._waiters.pop(key, None)
                self._signals.pop(key, None)

    async def _run_handlers(self, event: Event, handlers, *, offset: int = 0) -> None:
        index, retries = 0, 0
        while index < len(handlers):
            try:
                await handlers[index](event)
                index += 1
            except Exception as exc:
                if not await self._repository.retry_processing(
                    event.event_id, f"{type(exc).__name__}: {exc}", self._max_attempts
                ):
                    raise
                logger.warning("Retrying event %s at handler %d", event.event_id, offset + index)
                await asyncio.sleep(min(self._retry_delay * (2 ** min(retries, 10)), 5.0))
                retries += 1

    async def _record_failure(self, event_id: str, exc: Exception) -> None:
        logger.exception("Event processing failed for %s", event_id)
        try:
            await self._repository.mark_failed(event_id, f"{type(exc).__name__}: {exc}")
        except Exception:
            logger.exception("Failed to record failure for %s; recovery required", event_id)

    async def _worker(self) -> None:
        while True:
            event_id = await self._queue.get()
            claimed, deferred = False, False
            try:
                claimed = await self._repository.claim_for_processing(event_id)
                if not claimed:
                    continue
                event = await self._repository.get(event_id)
                if event is None:
                    raise RuntimeError(f"Queued event does not exist: {event_id}")
                await self._run_handlers(event, self._handlers)
                self._notify(event_id, "ready", True)
                if self._post_handlers:
                    await self._post_queue.put(event)
                    deferred = True
                else:
                    await self._repository.mark_processed(event_id)
                    self._notify(event_id, "processed", True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_failure(event_id, exc)
            finally:
                if claimed and not deferred:
                    if event_id not in self._completed:
                        self._notify(event_id, "processed", False)
                    if event_id not in self._ready:
                        self._notify(event_id, "ready", False)
                    self._responses.pop(event_id, None)
                self._queue.task_done()

    async def _post_worker(self) -> None:
        # Learning from the previous turn cannot block required projections
        # of the next turn. Both queues are bounded and included in drain/recovery.
        while True:
            event = await self._post_queue.get()
            event_id = event.event_id
            try:
                response = self._responses.get(event_id)
                if response is not None:
                    await response.wait()
                await self._run_handlers(event, self._post_handlers, offset=len(self._handlers))
                await self._repository.mark_processed(event_id)
                self._notify(event_id, "processed", True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._record_failure(event_id, exc)
            finally:
                if event_id not in self._completed:
                    self._notify(event_id, "processed", False)
                self._responses.pop(event_id, None)
                self._post_queue.task_done()
