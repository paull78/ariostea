"""IndexRuns for tests: the in-memory run lock, a run log in memory and a
fixed clock."""

from __future__ import annotations

from datetime import datetime, timezone

from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.domain.models import IndexStats
from ariostea.indexing.index_runs import IndexRuns

NOW = datetime(2026, 10, 27, 9, 41, 5, tzinfo=timezone.utc)


class MemoryRunLog:
    """The run-record half of IndexAdmin, kept in a list. `stats` answers
    for the index's counts when a run fails."""

    def __init__(self, stats=None):
        self.records = []
        self._stats = stats

    def record_run(self, record):
        self.records.append(record)

    def last_run(self):
        return self.records[-1] if self.records else None

    def last_completed_run(self):
        return next((r for r in reversed(self.records) if r.error is None), None)

    def stats(self):
        return self._stats() if self._stats is not None else IndexStats(notes=0, chunks=0)


def index_runs(indexer, lock=None, log=None, now=lambda: NOW):
    log = log if log is not None else MemoryRunLog()
    return IndexRuns(indexer, lock or InMemoryRunLock(), writer=log, reader=log, now=now)
