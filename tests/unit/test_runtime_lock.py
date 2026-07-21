from pathlib import Path

import pytest

from debilgi.runtime_lock import InstanceAlreadyRunningError, SingleInstanceLock


def test_second_process_lock_is_rejected_until_first_releases(tmp_path: Path) -> None:
    path = tmp_path / "debilgi.instance.lock"
    first = SingleInstanceLock(path)
    second = SingleInstanceLock(path)

    first.acquire()
    try:
        with pytest.raises(InstanceAlreadyRunningError):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
