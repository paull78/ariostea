"""One contract for every RunLock adapter: the file lock other processes see
and the in-memory test double must agree on taking, refusing and waiting."""

from __future__ import annotations

import threading
import time

import pytest

from ariostea.adapters.lock.file import FileRunLock
from ariostea.adapters.lock.memory import InMemoryRunLock
from ariostea.ports.run_lock import RunLock


def _file_pair(tmp_path):
    path = tmp_path / "index.db.lock"
    return FileRunLock(path), FileRunLock(path)


def _memory_pair(tmp_path):
    lock = InMemoryRunLock()
    return lock, lock.handle()


@pytest.fixture(params=[_file_pair, _memory_pair], ids=["file", "memory"])
def pair(request, tmp_path):
    a, b = request.param(tmp_path)
    yield a, b
    a.release()
    b.release()


def test_adapters_subclass_the_port():
    assert RunLock in FileRunLock.__mro__
    assert RunLock in InMemoryRunLock.__mro__


def test_held_is_false_on_a_fresh_lock(pair):
    a, b = pair
    assert not a.held() and not b.held()


def test_try_acquire_on_a_free_lock_takes_it(pair):
    a, b = pair
    assert a.try_acquire()
    assert a.held() and b.held()


def test_try_acquire_on_a_taken_lock_refuses(pair):
    a, b = pair
    a.try_acquire()
    started = time.monotonic()
    assert not b.try_acquire()
    assert time.monotonic() - started < 0.5


def test_release_frees_it(pair):
    a, b = pair
    a.try_acquire()
    a.release()
    assert not a.held() and not b.held()
    assert b.try_acquire()


def test_acquire_waits_for_release(pair):
    a, b = pair
    a.try_acquire()
    threading.Timer(0.3, a.release).start()
    started = time.monotonic()
    b.acquire()
    assert time.monotonic() - started >= 0.25
    assert not a.try_acquire()


def test_release_without_holding_does_nothing(pair):
    a, b = pair
    a.try_acquire()
    b.release()
    assert a.held() and not b.try_acquire()
