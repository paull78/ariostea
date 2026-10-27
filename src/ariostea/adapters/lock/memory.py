from __future__ import annotations

import threading

from ariostea.ports.run_lock import RunLock


class InMemoryRunLock(RunLock):
    """The run lock for tests. `handle()` gives another holder on the same
    lock, which is what another process is to the file lock."""

    def __init__(self, _shared: threading.Lock | None = None) -> None:
        self._shared = _shared if _shared is not None else threading.Lock()
        self._mine = False

    def handle(self) -> InMemoryRunLock:
        return InMemoryRunLock(self._shared)

    def try_acquire(self) -> bool:
        if self._mine:
            return False
        self._mine = self._shared.acquire(blocking=False)
        return self._mine

    def acquire(self) -> None:
        if self._mine:
            raise RuntimeError("this holder already has the run lock")
        self._shared.acquire()
        self._mine = True

    def release(self) -> None:
        if self._mine:
            self._mine = False
            self._shared.release()

    def held(self) -> bool:
        return self._shared.locked()
