"""IndexRuns: one run at a time through the run lock, with no threads of its
own. Fakes only: the in-memory lock, vault and store."""

from __future__ import annotations

import threading

import pytest
from test_index_vault import FakeStore, _indexer, _two_notes

from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.adapters.vault.memory import InMemoryVault
from ariostea.indexing.index_runs import IndexRuns
from ariostea.ports.vault import VaultUnavailable


def _runs(vault=None, store=None, lock=None):
    lock = lock or InMemoryRunLock()
    store = store if store is not None else FakeStore()
    return IndexRuns(_indexer(vault or _two_notes(), store=store), lock), lock


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
