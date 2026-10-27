# Step 04 · Watch passes through IndexRuns

**Milestone:** 1 · **Needs:** 03 · **Covers:** AC-06

## Goal
Each `watch` pass is a run: it takes the lock (waiting, with one line saying so, if another run holds it), indexes and releases it. An idle `watch` holds nothing.

## Files touched
- `src/ariostea/indexing/watch_vault.py`: `WatchVault(runs: IndexRuns, vault: Vault)`; `run(..., on_wait=_ignore)`; `_pass` does `runs.begin(wait=True, on_wait=on_wait).execute(progress)`.
- `src/ariostea/config/container.py`: `WatchVault(runs, vault)`.
- `src/ariostea/cli.py`: `watch` passes `on_wait` printing `waiting for the index run in progress` on stderr.
- `tests/indexing/test_watch_vault.py`: existing tests build `WatchVault` through `IndexRuns` with `InMemoryRunLock`; new tests below.
- `tests/test_cli.py`: the fake watcher accepts `on_wait`.

## Tests to write first
- `test_idle_watch_holds_no_lock` (AC-06 first half): `InMemoryVault.changes` yields one batch then blocks on an event; while blocked (checked from the test thread), `lock.handle().try_acquire()` is True. Release the handle before letting the watch go on.
- `test_pass_waits_for_a_run_in_progress` (AC-06 second half): `other` holds the lock; the vault yields a batch; the pass calls `on_wait` once, does not index until `other.release()`, then indexes (the report reaches `on_report`).
- `test_first_pass_waits_too`: `other` holds the lock at start; `on_wait` is called before the first pass's report.
- Existing tests (guard failure on a later pass is a warning, skipped files warned once) still pass unchanged in what they assert.

## Implementation notes
- `InMemoryVault.changes()` in `adapters/vault/memory.py` is what existing watch tests drive; reuse its pattern. Waiting tests use one helper thread and an event, with timeouts on every wait (no test hangs on failure).
- A `VaultUnavailable` from `execute` on a later pass stays a warning; the lock is already released by `Run.execute`.

## Verify
`uv run pytest tests/indexing tests/test_cli.py -q` → all pass; then `uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass; `WatchVault` no longer imports `IndexVault`.
- `uv run ruff check . && uv run ruff format --check .` is clean.
