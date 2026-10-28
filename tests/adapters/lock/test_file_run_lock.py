"""The file lock across processes: another process holding it is seen, and a
process killed with SIGKILL leaves nothing behind (AC-04)."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import pytest

from ariostea.adapters.lock.file import FileRunLock

HOLD = (
    "import sys, time\n"
    "from ariostea.adapters.lock.file import FileRunLock\n"
    "FileRunLock(sys.argv[1]).acquire()\n"
    "print('held', flush=True)\n"
    "time.sleep(600)\n"
)


@pytest.fixture
def holder(tmp_path):
    """A process holding the lock on `tmp_path/index.db.lock`."""
    path = tmp_path / "index.db.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(path)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert proc.stdout.readline().strip() == "held"
        yield path, proc
    finally:
        proc.kill()  # by PID
        proc.wait(timeout=10)


def test_lock_file_is_next_to_the_index(tmp_path):
    path = tmp_path / "index.db.lock"
    lock = FileRunLock(path)
    assert lock.try_acquire()
    assert path.exists()
    lock.release()
    assert path.exists()


def test_another_process_holding_it_is_seen(holder):
    path, _ = holder
    lock = FileRunLock(path)
    assert not lock.try_acquire()
    assert lock.held()


def test_sigkill_releases_it(holder):  # AC-04
    path, proc = holder
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=10)
    lock = FileRunLock(path)
    deadline = time.monotonic() + 1
    while not lock.try_acquire():
        assert time.monotonic() < deadline, "lock still held 1 s after SIGKILL"
        time.sleep(0.05)
    lock.release()
