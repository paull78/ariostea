from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ariostea.domain.models import IndexReport, Progress, RunRecord
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.run_lock import RunLock
from ariostea.ports.store import IndexAdmin


def _nothing() -> None:
    pass


@dataclass(frozen=True)
class RunState:
    """Whether a run is going on in any process, the last run that ended and
    the last one that completed, as the index records them."""

    running: bool
    last_run: RunRecord | None
    last_completed: RunRecord | None


class Run:
    """One index run, holding the run lock until `execute` ends."""

    def __init__(
        self,
        indexer: IndexVault,
        lock: RunLock,
        writer: IndexAdmin,
        now: Callable[[], datetime],
    ) -> None:
        self._indexer = indexer
        self._lock = lock
        self._writer = writer
        self._now = now
        self._done = False

    def execute(self, progress: Callable[[Progress], None] | None = None) -> IndexReport:
        """Index the vault, record how the run ended, completed or failed, and
        release the lock, however the run ends. A failure is re-raised after
        it is recorded. A process killed partway records nothing."""
        if self._done:
            raise RuntimeError("this run has already been executed")
        self._done = True
        fingerprint = self._indexer.fingerprint
        try:
            try:
                report = self._indexer.index(progress=progress)
            except Exception as exc:
                stats = self._writer.stats()
                self._writer.record_run(
                    RunRecord(self._now(), fingerprint, stats.notes, stats.chunks, 0, str(exc))
                )
                raise
            self._writer.record_run(
                RunRecord(
                    self._now(),
                    fingerprint,
                    report.notes,
                    report.chunks,
                    len(report.skipped),
                    None,
                )
            )
            return report
        finally:
            self._lock.release()


class IndexRuns:
    """Every index run goes through here, so only one runs at a time across
    every process that writes the index. It starts no threads: how a run is
    delivered, inline or in the background, is the caller's business
    (server-robustness.09). Runs record how they ended through `writer`, the
    connection the run holding the lock writes through; `state` reads through
    `reader`, which another thread may use while a run writes. The clock is
    passed in, so time stays outside (vault-hardening.16)."""

    def __init__(
        self,
        indexer: IndexVault,
        lock: RunLock,
        writer: IndexAdmin,
        reader: IndexAdmin,
        now: Callable[[], datetime],
    ) -> None:
        self._indexer = indexer
        self._lock = lock
        self._writer = writer
        self._reader = reader
        self._now = now

    def begin(self, wait: bool, on_wait: Callable[[], None] = _nothing) -> Run | None:
        """A run holding the lock. When another run holds it: None, or with
        `wait`, call `on_wait` once and block until it is free."""
        if not self._lock.try_acquire():
            if not wait:
                return None
            on_wait()
            self._lock.acquire()
        return Run(self._indexer, self._lock, self._writer, self._now)

    def state(self) -> RunState:
        return RunState(
            running=self._lock.held(),
            last_run=self._reader.last_run(),
            last_completed=self._reader.last_completed_run(),
        )
