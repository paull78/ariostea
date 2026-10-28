from __future__ import annotations

import fcntl
import os
from pathlib import Path

from ariostea.ports.run_lock import RunLock


class FileRunLock(RunLock):
    """The run lock as an flock on a file next to the index. The operating
    system releases it when the process ends, however it ends, so a crash
    never leaves a stale lock. The file itself stays: deleting a lock file
    races with the next holder opening it."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._fd: int | None = None

    def _lock(self, flags: int) -> int | None:
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            return None
        return fd

    def try_acquire(self) -> bool:
        if self._fd is not None:
            return False
        self._fd = self._lock(fcntl.LOCK_EX | fcntl.LOCK_NB)
        return self._fd is not None

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError("this holder already has the run lock")
        self._fd = self._lock(fcntl.LOCK_EX)

    def release(self) -> None:
        if self._fd is None:
            return
        os.close(self._fd)  # closing the descriptor drops the flock
        self._fd = None

    def held(self) -> bool:
        if self._fd is not None:
            return True
        # flock belongs to the open file, so a probe through a new descriptor
        # is refused even when this process holds the lock through another.
        probe = self._lock(fcntl.LOCK_EX | fcntl.LOCK_NB)
        if probe is None:
            return True
        os.close(probe)
        return False
