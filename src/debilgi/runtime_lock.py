"""Cross-process guard that keeps one Discord bot attached to a database."""

from __future__ import annotations

import os
from pathlib import Path
from types import TracebackType
from typing import BinaryIO


class InstanceAlreadyRunningError(RuntimeError):
    """Raised when another process owns the bot instance lock."""


class SingleInstanceLock:
    """Hold a non-blocking OS file lock until the bot shuts down."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None

    @classmethod
    def for_database(cls, database_path: Path) -> SingleInstanceLock:
        return cls(database_path.with_name(f"{database_path.name}.instance.lock"))

    def acquire(self) -> None:
        if self._handle is not None:
            raise RuntimeError("instance lock is already acquired by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            _lock(handle)
        except OSError as exc:
            handle.close()
            raise InstanceAlreadyRunningError(
                "another DEBiLGI process is already using "
                f"{self.path.with_suffix('').name}"
            ) from exc
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> SingleInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()


if os.name == "nt":
    import msvcrt

    def _lock(handle: BinaryIO) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock(handle: BinaryIO) -> None:
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(handle: BinaryIO) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
