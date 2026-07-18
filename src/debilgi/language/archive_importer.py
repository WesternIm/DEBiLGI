"""Restartable, batched archive import orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Protocol

from debilgi.language.archive_repository import ArchiveRepository
from debilgi.language.models import ArchivedMessage, ArchiveImportJob, ImportStatus

logger = logging.getLogger(__name__)


class ArchiveSource(Protocol):
    async def fetch_batch(self, job: ArchiveImportJob, limit: int) -> Sequence[ArchivedMessage]: ...


class ArchiveImporter:
    def __init__(self, repository: ArchiveRepository, *, batch_size: int = 100) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._repository = repository
        self._batch_size = batch_size

    async def run(self, job_id: str, source: ArchiveSource) -> None:
        await self._repository.mark_running(job_id)
        try:
            while True:
                job = await self._repository.get_job(job_id)
                if job is None:
                    raise LookupError(f"Archive job does not exist: {job_id}")
                if job.status is not ImportStatus.RUNNING:
                    return
                if job.remaining == 0:
                    await self._repository.complete_job(job_id)
                    return

                requested = min(self._batch_size, job.remaining)
                batch = list(await source.fetch_batch(job, requested))
                if not batch:
                    await self._repository.complete_job(job_id)
                    return
                self._validate_batch(job, batch, requested)

                processed = 0
                for message in batch:
                    processed += int(await self._repository.ingest_message(job_id, message))

                await self._repository.advance_job(
                    job_id,
                    last_message_id=batch[-1].message_id,
                    seen=len(batch),
                    processed=processed,
                )
                if len(batch) < requested:
                    await self._repository.complete_job(job_id)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._repository.fail_job(job_id, f"{type(exc).__name__}: {exc}")
            raise

    @staticmethod
    def _validate_batch(
        job: ArchiveImportJob, batch: Sequence[ArchivedMessage], requested: int
    ) -> None:
        if len(batch) > requested:
            raise ValueError("archive source returned more messages than requested")
        previous_id = job.last_message_id
        for message in batch:
            if message.guild_id != job.guild_id or message.channel_id != job.channel_id:
                raise ValueError("archive source crossed guild or channel boundary")
            if previous_id is not None and message.message_id <= previous_id:
                raise ValueError("archive source must return messages in ascending order")
            previous_id = message.message_id


SourceFactory = Callable[[ArchiveImportJob], ArchiveSource]


class ArchiveImportManager:
    def __init__(
        self,
        repository: ArchiveRepository,
        importer: ArchiveImporter,
        source_factory: SourceFactory,
    ) -> None:
        self._repository = repository
        self._importer = importer
        self._source_factory = source_factory
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for job in await self._repository.resumable_jobs():
            self.submit(job)

    def submit(self, job: ArchiveImportJob) -> None:
        active = self._tasks.get(job.job_id)
        if active and not active.done():
            return
        task = asyncio.create_task(
            self._importer.run(job.job_id, self._source_factory(job)),
            name=f"debilgi-archive-{job.job_id}",
        )
        self._tasks[job.job_id] = task
        task.add_done_callback(lambda completed, job_id=job.job_id: self._done(job_id, completed))

    async def pause(self, job_id: str) -> bool:
        changed = await self._repository.pause_job(job_id)
        task = self._tasks.get(job_id)
        if changed and task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return changed

    async def resume(self, job_id: str) -> bool:
        changed = await self._repository.resume_job(job_id)
        if not changed:
            return False
        job = await self._repository.get_job(job_id)
        assert job is not None
        self.submit(job)
        return True

    async def cancel(self, job_id: str) -> bool:
        changed = await self._repository.cancel_job(job_id)
        task = self._tasks.get(job_id)
        if changed and task and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return changed

    async def stop(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._started = False

    def _done(self, job_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(job_id) is task:
            self._tasks.pop(job_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("Archive import %s failed: %s", job_id, error)
