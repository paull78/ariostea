"""IndexRuns: one run at a time through the run lock, with no threads of its
own. Fakes only: the in-memory lock, vault and store."""

from __future__ import annotations

import threading

import pytest
from fixtures.runs import NOW, MemoryRunLog, index_runs
from test_index_vault import FailingEmbed, FakeStore, _indexer, _two_notes

from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.domain.models import RunRecord
from ariostea.ports.vault import VaultUnavailable


def _runs(vault=None, store=None, lock=None, log=None, embed=None):
    lock = lock or InMemoryRunLock()
    store = store if store is not None else FakeStore()
    log = log if log is not None else MemoryRunLog(stats=store.stats)
    return index_runs(_indexer(vault or _two_notes(), embed, store), lock, log), lock


def test_begin_takes_the_lock():
    runs, lock = _runs()
    assert runs.begin(wait=False) is not None
    assert not lock.handle().try_acquire()


def test_begin_refuses_when_another_holds_it():
    runs, lock = _runs()
    other = lock.handle()
    other.try_acquire()
    calls = []
    assert runs.begin(wait=False, on_wait=lambda: calls.append(1)) is None
    assert calls == []


def test_begin_waits_and_says_so_once():
    runs, lock = _runs()
    other = lock.handle()
    other.try_acquire()
    threading.Timer(0.2, other.release).start()
    calls = []
    run = runs.begin(wait=True, on_wait=lambda: calls.append(1))
    assert run is not None
    assert calls == [1]
    assert not other.try_acquire()


def test_begin_on_a_free_lock_does_not_say_it_waits():
    runs, _ = _runs()
    calls = []
    assert runs.begin(wait=True, on_wait=lambda: calls.append(1)) is not None
    assert calls == []


def test_execute_indexes_and_releases():
    store = FakeStore()
    runs, lock = _runs(store=store)
    report = runs.begin(wait=False).execute()
    assert report.notes == 2 and set(store.notes) == {"a.md", "b.md"}
    assert not runs.state().running
    assert lock.handle().try_acquire()


def test_execute_releases_on_failure():
    store = FakeStore()
    _runs(store=store)[0].begin(wait=False).execute()  # the index now holds notes
    runs, lock = _runs(vault=InMemoryVault({}), store=store)
    run = runs.begin(wait=False)
    with pytest.raises(VaultUnavailable):
        run.execute()
    assert lock.handle().try_acquire()


def test_execute_passes_progress():
    runs, _ = _runs()
    seen = []
    runs.begin(wait=False).execute(progress=seen.append)
    assert [p.done for p in seen] == [1, 2]


def test_state_running_follows_the_lock():
    runs, lock = _runs()
    assert not runs.state().running
    other = lock.handle()
    other.try_acquire()
    assert runs.state().running
    other.release()
    assert not runs.state().running


def test_run_executes_once():
    runs, _ = _runs()
    run = runs.begin(wait=False)
    run.execute()
    with pytest.raises(RuntimeError):
        run.execute()


# --- run records (server-robustness.05)


def test_execute_records_a_completed_run():
    runs, _ = _runs()
    runs.begin(wait=False).execute()
    expected = RunRecord(NOW, "fake:v1|noop", notes=2, chunks=2, skipped=0, error=None)
    state = runs.state()
    assert state.last_run == expected and state.last_completed == expected


def test_execute_records_a_failed_run():  # AC-02, failure part
    store, log = FakeStore(), MemoryRunLog()
    log._stats = store.stats
    _runs(store=store, log=log)[0].begin(wait=False).execute()
    before = dict(store.notes)
    runs, _ = _runs(vault=InMemoryVault({}), store=store, log=log)
    with pytest.raises(VaultUnavailable):
        runs.begin(wait=False).execute()
    state = runs.state()
    assert "unavailable" in state.last_run.error
    assert (state.last_run.notes, state.last_run.chunks) == (2, 2)
    assert state.last_completed == log.records[0] and state.last_completed.error is None
    assert store.notes == before


def test_record_is_written_before_the_lock_is_released():
    events = []

    class OrderLock(InMemoryRunLock):
        def release(self):
            events.append("release")
            super().release()

    class OrderLog(MemoryRunLog):
        def record_run(self, record):
            events.append("record")
            super().record_run(record)

    runs, _ = _runs(lock=OrderLock(), log=OrderLog())
    runs.begin(wait=False).execute()
    assert events == ["record", "release"]


def test_any_exception_is_recorded_and_reraised():
    runs, lock = _runs(embed=FailingEmbed(allowed=0))
    with pytest.raises(RuntimeError, match="stopped"):
        runs.begin(wait=False).execute()
    assert runs.state().last_run.error == "stopped"
    assert runs.state().last_completed is None
    assert lock.handle().try_acquire()


def test_state_reads_through_the_reader():
    from ariostea.indexing.index_runs import IndexRuns

    writer, reader = MemoryRunLog(), MemoryRunLog()
    reader.record_run(RunRecord(NOW, "old", 1, 1, 0, None))
    runs = IndexRuns(_indexer(_two_notes()), InMemoryRunLock(), writer, reader, lambda: NOW)
    runs.begin(wait=False).execute()
    assert len(writer.records) == 1
    assert runs.state().last_run.fingerprint == "old"
