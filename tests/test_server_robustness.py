"""Server robustness end to end: the real CLI, `watch` and stdio server on the
awkward vault and the default models, several processes on one index. A first
index of this vault takes about 30 seconds: 8 to load the models, 20 to index."""

from __future__ import annotations

import asyncio
import json
import queue
import re
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fixtures.awkward_queries import KNOWN_ANSWERS
from fixtures.awkward_vault import build
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ariostea.adapters.lock.file import FileRunLock

pytestmark = pytest.mark.integration

ARIOSTEA = [sys.executable, "-m", "ariostea.cli"]
WAITING = "waiting for the index run in progress"
QUERY = KNOWN_ANSWERS[0][0]
NOTE = KNOWN_ANSWERS[0][1]
HOLD = (
    "import sys, time\n"
    "from ariostea.adapters.lock.file import FileRunLock\n"
    "FileRunLock(sys.argv[1]).acquire()\n"
    "print('held', flush=True)\n"
    "time.sleep(600)\n"
)


@pytest.fixture(scope="module")
def vault(tmp_path_factory):
    return build(tmp_path_factory.mktemp("robust"))


def _config(vault: Path, root: Path) -> Path:
    """A config on `vault` with a fresh, empty index under `root`."""
    root.mkdir(parents=True, exist_ok=True)
    config = root / "ariostea.toml"
    config.write_text(f'[vault]\npath = "{vault}"\n\n[store]\npath = "{root / "index.db"}"\n')
    return config


def _store(config: Path) -> Path:
    return config.parent / "index.db"


@pytest.fixture(scope="module")
def indexed(vault, tmp_path_factory):
    """A config whose index already holds the vault."""
    config = _config(vault, tmp_path_factory.mktemp("indexed"))
    done = subprocess.run(
        [*ARIOSTEA, "reindex", "--config", str(config)], capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, done.stderr
    return config


@asynccontextmanager
async def _server(config: Path):
    """A stdio server on `config`; yields `call(tool, **args)` -> JSON payload."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "ariostea.cli", "serve", "--config", str(config)]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            async def call(tool, **args):
                result = await session.call_tool(tool, args)
                assert not result.isError, result
                return json.loads("".join(c.text for c in result.content if c.type == "text"))

            yield call


async def _timed(call, tool, **args):
    started = time.monotonic()
    payload = await call(tool, **args)
    return payload, time.monotonic() - started


async def _until(call, check, timeout: float):
    """Poll `status` until `check(payload)` holds; return that payload."""
    deadline = time.monotonic() + timeout
    while True:
        payload = await call("status")
        if check(payload):
            return payload
        assert time.monotonic() < deadline, f"status never matched: {payload}"
        await asyncio.sleep(0.25)


@contextmanager
def _lock_held_elsewhere(config: Path):
    """Another process holding the run lock until the block ends."""
    proc = subprocess.Popen(
        [sys.executable, "-c", HOLD, str(_store(config)) + ".lock"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout.readline().strip() == "held"
        yield
    finally:
        proc.kill()  # by PID
        proc.wait(timeout=10)


def _lines(stream) -> queue.Queue[str]:
    lines: queue.Queue[str] = queue.Queue()

    def pump():
        for line in stream:
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()
    return lines


def _wait_for_line(lines: queue.Queue[str], want, timeout: float) -> list[str]:
    """Read lines until one satisfies `want`; return every line read."""
    seen, deadline = [], time.monotonic() + timeout
    while True:
        line = lines.get(timeout=max(0.1, deadline - time.monotonic()))
        seen.append(line)
        if want(line):
            return seen


def test_reindex_returns_at_once_and_search_answers_mid_run(vault, tmp_path):  # AC-01
    config = _config(vault, tmp_path / "own")

    async def main():
        async with _server(config) as call:
            started, took = await _timed(call, "reindex")
            assert started == {"started": True} and took < 1
            during = await _until(call, lambda s: "progress" in s, timeout=30)
            assert during["running"] is True and during["progress"]["total"] > 0
            refused, took = await _timed(call, "reindex")
            assert refused == {"started": False, "reason": "an index run is in progress"}
            assert took < 1
            found = await call("search_knowledge", query=QUERY, k=3)
            assert "results" in found
            still = await call("status")
            assert still["running"] is True, "the run ended before the search was answered"
            after = await _until(call, lambda s: not s["running"], timeout=300)  # AC-02
            assert after["notes"] > 0 and "progress" not in after
            last = after["last_run"]
            assert last["error"] is None and last["notes"] == after["notes"]
            assert last["chunks"] == after["chunks"] and last["skipped"] == 1  # the Latin-1 note
            assert after["last_indexed"] == last["finished_at"]

    asyncio.run(main())


def test_second_writer_is_refused_or_waits(indexed):  # AC-03
    async def refused_by_the_server():
        async with _server(indexed) as call:
            return await _timed(call, "reindex")

    with _lock_held_elsewhere(indexed):
        refused, took = asyncio.run(refused_by_the_server())
        assert refused == {"started": False, "reason": "an index run is in progress"}
        assert took < 1
        cli = subprocess.Popen(
            [*ARIOSTEA, "reindex", "--config", str(indexed)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            err = _lines(cli.stderr)
            _wait_for_line(err, lambda line: line.strip() == WAITING, timeout=60)
            time.sleep(2)
            assert cli.poll() is None, "the CLI ran while another process held the lock"
        except BaseException:
            cli.kill()
            raise
    out, _ = cli.communicate(timeout=300)
    rest = []
    while not err.empty():
        rest.append(err.get())
    assert cli.returncode == 0, rest
    assert "database is locked" not in out + "".join(rest)
    assert re.search(r"Indexed \d+ notes", out)


def _when(iso: str) -> datetime:
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def test_reads_answer_while_another_process_writes(vault, tmp_path):  # AC-05, AC-10, AC-11
    config = _config(vault, tmp_path / "shared")

    async def main():
        async with _server(config) as call:
            await call("search_knowledge", query="warm up", k=1)  # first inference is slow
            fresh = await call("status")
            assert fresh["last_indexed"] is None and fresh["last_run"] is None
            cli = subprocess.Popen(
                [*ARIOSTEA, "reindex", "--config", str(config)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                writing = await _until(call, lambda s: s["running"] and s["notes"] > 0, timeout=120)
                assert "progress" not in writing  # a run in another process
                for tool, args in [
                    ("search_knowledge", {"query": QUERY, "k": 3}),
                    ("get_note", {"path": NOTE}),
                    ("status", {}),
                ]:
                    _, took = await _timed(call, tool, **args)
                    assert took < 5, (tool, took)
                assert cli.poll() is None, "the CLI run ended before the reads were done"
            finally:
                out, err = await asyncio.to_thread(cli.communicate, timeout=300)
            ended = datetime.now(timezone.utc)
            assert cli.returncode == 0, err
            assert "database is locked" not in out + err

            # AC-10: the server, started before the CLI run, reports it.
            seen = await call("status")
            assert seen["running"] is False
            # last_indexed is to the second, truncated: up to 1 s of the gap
            # is the format, the rest is the CLI exiting after the run.
            assert timedelta(0) <= ended - _when(seen["last_indexed"]) < timedelta(seconds=2)
            assert seen["config_fingerprint"]
            counts = re.search(r"Indexed (\d+) notes, (\d+) chunks, (\d+) skipped", out)
            last = seen["last_run"]
            assert (last["notes"], last["chunks"], last["skipped"]) == tuple(
                int(n) for n in counts.groups()
            )
            again = await call("status")
            assert again["last_indexed"] == seen["last_indexed"]

    asyncio.run(main())


def _chunks_containing(store: Path, word: str) -> int:
    db = sqlite3.connect(store)
    try:
        return db.execute(
            "SELECT count(*) FROM chunks WHERE text LIKE ?", (f"%{word}%",)
        ).fetchone()[0]
    finally:
        db.close()


def _first_pass_done(line: str) -> bool:
    match = re.match(r"(\d+)/([\d,]+) notes", line)
    return bool(match) and int(match[1]) == int(match[2].replace(",", ""))


def test_idle_watch_holds_no_lock_and_waits_for_a_run(indexed):  # AC-06
    watcher = subprocess.Popen(
        [*ARIOSTEA, "watch", "--config", str(indexed)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        err = _lines(watcher.stderr)
        _wait_for_line(err, _first_pass_done, timeout=300)
        time.sleep(1)  # the file watcher starts right after the first pass
        lock = FileRunLock(str(_store(indexed)) + ".lock")
        assert lock.try_acquire(), "an idle watch holds the run lock"
        lock.release()

        vault = Path(re.search(r'path = "(.*)"', indexed.read_text())[1])
        note = vault / "Hobbies" / "Birdwatching.md"
        with _lock_held_elsewhere(indexed):
            note.write_text(note.read_text() + "\nThe loft holds a WAITMARKER_29 crate.\n")
            _wait_for_line(err, lambda line: line.strip() == WAITING, timeout=10)
            time.sleep(1)
            assert not _chunks_containing(_store(indexed), "WAITMARKER_29")
        deadline = time.monotonic() + 10
        while not _chunks_containing(_store(indexed), "WAITMARKER_29"):
            assert time.monotonic() < deadline, "the waiting pass did not run within 10 s"
            time.sleep(0.25)
    finally:
        watcher.terminate()  # by PID
        watcher.wait(timeout=30)


def test_guard_failure_is_reported(indexed):  # AC-02
    vault = Path(re.search(r'path = "(.*)"', indexed.read_text())[1])
    away = vault.with_name(vault.name + "-away")

    async def main():
        async with _server(indexed) as call:
            before = await _until(call, lambda s: not s["running"], timeout=60)
            vault.rename(away)
            try:
                assert await call("reindex") == {"started": True}
                after = await _until(
                    call, lambda s: not s["running"] and s["last_run"]["error"], timeout=60
                )
            finally:
                away.rename(vault)
            assert "unavailable" in after["last_run"]["error"]
            assert after["last_indexed"] == before["last_indexed"]
            assert after["config_fingerprint"] == before["config_fingerprint"]
            assert (after["notes"], after["chunks"]) == (before["notes"], before["chunks"])

    asyncio.run(main())
