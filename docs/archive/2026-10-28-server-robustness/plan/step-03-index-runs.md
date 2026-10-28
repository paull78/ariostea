# Step 03 · IndexRuns, used by CLI reindex

**Milestone:** 1 · **Needs:** 01, 02 · **Covers:** AC-03 (CLI part), AC-12

## Goal
Every index run goes through one use case, `IndexRuns`, which takes the run lock: `begin(wait, on_wait)` returns a `Run` holding the lock or `None`, `Run.execute(progress)` indexes and releases the lock, `state()` reports `running`. The Container exposes `runs` in place of the bare indexer, and CLI `reindex` prints a waiting line on stderr when another run holds the lock, waits, then runs.

## Files touched
- `src/ariostea/indexing/index_runs.py`: new, `IndexRuns`, `Run`, `RunState`.
- `src/ariostea/config/container.py`: `lock = FileRunLock(store_path + ".lock")`; `runs = IndexRuns(indexer, lock)`; Container field `indexer` replaced by `runs`.
- `src/ariostea/mcp/handlers.py`: `reindex_payload(container, progress=None, on_wait=None)` goes through `container.runs.begin(wait=True, on_wait=...)` and `execute`.
- `src/ariostea/cli.py`: `reindex` passes `on_wait` printing `waiting for the index run in progress` on stderr.
- `tests/indexing/test_index_runs.py`: new.
- `tests/test_cli.py`, `tests/test_chunk_memory.py`, `tests/config/test_container.py`: callers of `container.indexer` move to `container.runs`.

## Tests to write first
In `tests/indexing/test_index_runs.py`, with `InMemoryRunLock`, `InMemoryVault`, the fake embedder and the in-memory store fake `tests/indexing/test_index_vault.py` already uses (import or copy it), no threads except where stated:
- `test_begin_takes_the_lock`: `runs.begin(wait=False)` returns a `Run`; `lock.handle().try_acquire()` is False.
- `test_begin_refuses_when_another_holds_it`: `other = lock.handle(); other.try_acquire()`; `runs.begin(wait=False)` is None and `on_wait` was not called.
- `test_begin_waits_and_says_so_once`: `other` holds it; a thread releases `other` after 0.2 s; `runs.begin(wait=True, on_wait=calls.append)` returns a `Run` and `calls` has exactly one entry.
- `test_execute_indexes_and_releases`: `begin(...).execute()` returns the `IndexReport` the indexer gives; afterwards `state().running` is False and `lock.handle().try_acquire()` is True.
- `test_execute_releases_on_failure`: a vault that fails its guard (empty vault, non-empty store); `execute()` raises `VaultUnavailable` and the lock is free.
- `test_execute_passes_progress`: the progress callback hears every listed note.
- `test_state_running_follows_the_lock`: False on a free lock, True while `other` holds it.
- `test_run_executes_once`: a second `execute()` on the same `Run` raises `RuntimeError` (the lock is already released).

In `tests/test_cli.py`:
- `test_reindex_cli_waits_with_one_line` (AC-03 CLI part): a fake `runs` whose `begin(wait=True, on_wait=...)` calls `on_wait()` then returns a run with a fixed report; stderr's first line is `waiting for the index run in progress`, exit 0.
- Existing CLI tests using `SimpleNamespace(indexer=...)` move to a fake `runs` with the same behaviour.

## Implementation notes
- `IndexRuns(indexer: IndexVault, lock: RunLock)` for now; step 08 adds the run record and `now`. It starts no threads (server-robustness.09).
  ```python
  def begin(self, wait: bool, on_wait: Callable[[], None] = _nothing) -> Run | None:
      if self._lock.try_acquire(): return Run(self._indexer, self._lock)
      if not wait: return None
      on_wait(); self._lock.acquire(); return Run(self._indexer, self._lock)
  ```
- `Run.execute(progress=None) -> IndexReport`: `try: return indexer.index(progress) finally: lock.release()`.
- `RunState` frozen dataclass in `domain/models.py` or next to `IndexRuns`; it lives in `indexing/` unless a port needs it. `running = lock.held()`.
- `indexing/` imports only ports, domain and the standard library (vault-hardening.17); the test in `tests/test_architecture.py` already covers the folder.
- Container docstring stays true: it exposes config, use cases, port-typed admin (setup.01). `WatchVault` still takes the indexer in this step; step 04 moves it.
- The eval scripts call `reindex_payload(container)`; its behaviour for them is unchanged (it waits if needed).

## Verify
`uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass; no caller references `container.indexer`.
- `uv run ruff check . && uv run ruff format --check .` is clean.
