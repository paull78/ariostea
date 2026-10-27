from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ariostea.domain.models import IndexReport, Progress
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.run_lock import RunLock


def _nothing() -> None:
    pass


@dataclass(frozen=True)
class RunState:
    """Whether a run is going on in any process."""

    running: bool


class Run:
    """One index run, holding the run lock until `execute` ends."""

    def __init__(self, indexer: IndexVault, lock: RunLock) -> None:
        self._indexer = indexer
        self._lock = lock
        self._done = False

    def execute(self, progress: Callable[[Progress], None] | None = None) -> IndexReport:
        """Index the vault and release the lock, however the run ends."""
        if self._done:
            raise RuntimeError("this run has already been executed")
        self._done = True
        try:
            return self._indexer.index(progress=progress)
        finally:
            self._lock.release()


class IndexRuns:
    """Every index run goes through here, so only one runs at a time across
    every process that writes the index. It starts no threads: how a run is
    delivered, inline or in the background, is the caller's business
    (server-robustness.09)."""

    def __init__(self, indexer: IndexVault, lock: RunLock) -> None:
        self._indexer = indexer
        self._lock = lock

    def begin(self, wait: bool, on_wait: Callable[[], None] = _nothing) -> Run | None:
        """A run holding the lock. When another run holds it: None, or with
        `wait`, call `on_wait` once and block until it is free."""
        if not self._lock.try_acquire():
            if not wait:
                return None
            on_wait()
            self._lock.acquire()
        return Run(self._indexer, self._lock)

    def state(self) -> RunState:
        return RunState(running=self._lock.held())
