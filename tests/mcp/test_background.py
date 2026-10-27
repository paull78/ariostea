"""The MCP reindex runs on a thread the adapter owns, and answers at once."""

from __future__ import annotations

import logging
import threading
from types import SimpleNamespace

from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.domain.models import IndexReport, IndexStats, Progress
from ariostea.indexing.index_runs import IndexRuns
from ariostea.mcp.background import BackgroundRuns
from ariostea.mcp.handlers import start_reindex_payload, status_payload
from ariostea.ports.vault import VaultUnavailable

WAIT = 10  # seconds; no test hangs when it fails


class BlockingIndexer:
    """Reports one progress value, then waits for `go` before finishing."""

    def __init__(self, error: Exception | None = None):
        self.go = threading.Event()
        self.reported = threading.Event()
        self.error = error

    def index(self, progress=None):
        if progress is not None:
            progress(Progress(done=1, total=3, embedded=1, skipped=0))
        self.reported.set()
        assert self.go.wait(WAIT)
        if self.error is not None:
            raise self.error
        return IndexReport(3, 9, (), (), ())


class FakeAdmin:
    def stats(self):
        return IndexStats(notes=3, chunks=9, last_indexed=42.0, config_fingerprint="fp")


def _setup(error=None):
    indexer, lock = BlockingIndexer(error), InMemoryRunLock()
    runs = IndexRuns(indexer, lock)
    container = SimpleNamespace(runs=runs, admin=FakeAdmin())
    return indexer, lock, BackgroundRuns(runs), container


def _finish(indexer, background):
    indexer.go.set()
    background.join(WAIT)


def test_start_answers_before_the_run_ends():  # AC-01, unit level
    indexer, _, background, _ = _setup()
    assert background.start()
    assert not indexer.go.is_set()
    assert start_reindex_payload(background) == {
        "started": False,
        "reason": "an index run is in progress",
    }
    _finish(indexer, background)


def test_second_start_is_refused_while_running():  # AC-03, MCP part
    indexer, _, background, _ = _setup()
    assert start_reindex_payload(background) == {"started": True}
    assert not background.start()
    _finish(indexer, background)


def test_start_is_refused_when_another_process_holds_the_lock():
    _, lock, background, _ = _setup()
    other = lock.handle()
    other.try_acquire()
    assert not background.start()
    assert background.progress() is None
    other.release()


def test_progress_is_kept_while_running_and_cleared_after():
    indexer, lock, background, _ = _setup()
    background.start()
    assert indexer.reported.wait(WAIT)
    assert background.progress() == Progress(done=1, total=3, embedded=1, skipped=0)
    _finish(indexer, background)
    assert background.progress() is None
    assert lock.handle().try_acquire()


def test_a_failing_run_frees_the_lock_and_logs(caplog):
    indexer, lock, background, _ = _setup(VaultUnavailable("/gone", "no such folder"))
    with caplog.at_level(logging.ERROR):
        background.start()
        _finish(indexer, background)
    assert background.progress() is None
    assert lock.handle().try_acquire()
    assert "vault /gone is unavailable" in caplog.text


def test_status_reports_running_and_progress():
    indexer, _, background, container = _setup()
    background.start()
    assert indexer.reported.wait(WAIT)
    payload = status_payload(container, background)
    assert payload["running"] is True
    assert payload["progress"] == {"done": 1, "total": 3, "embedded": 1, "skipped": 0}
    _finish(indexer, background)
    payload = status_payload(container, background)
    assert payload["running"] is False and "progress" not in payload


def test_status_without_background_has_no_progress():  # AC-11, unit level
    _, lock, _, container = _setup()
    other = lock.handle()
    other.try_acquire()
    payload = status_payload(container)
    assert payload["running"] is True and "progress" not in payload
    other.release()
