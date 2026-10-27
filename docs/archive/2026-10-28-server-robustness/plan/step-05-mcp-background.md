# Step 05 · MCP reindex on a background thread

**Milestone:** 1 · **Needs:** 03 · **Covers:** AC-01, AC-03 (MCP part), AC-11 (unit level)

## Goal
The MCP `reindex` tool starts a run on a thread the MCP adapter owns and answers at once: `{"started": true}`, or `{"started": false, "reason": "an index run is in progress"}` when the lock is taken. The adapter keeps the run's latest progress, and `status` reports `running` and, for a run inside this server, `progress`.

## Files touched
- `src/ariostea/mcp/background.py`: new, `BackgroundRuns(runs)` with `start() -> bool` and `progress() -> Progress | None`.
- `src/ariostea/mcp/handlers.py`: `start_reindex_payload(background)`; `status_payload(container, background=None)` adds `running` and `progress`.
- `src/ariostea/mcp/server.py`: `build_server` creates one `BackgroundRuns(container.runs)`; the `reindex` tool returns `start_reindex_payload`; `status` passes the background; tool docstrings updated.
- `src/ariostea/cli.py`: `status` calls `status_payload(container)`.
- `tests/mcp/test_background.py`: new. `tests/mcp/test_handlers.py`, `tests/test_end_to_end.py`: `status_payload` callers.

## Tests to write first
With `IndexRuns` on `InMemoryRunLock` and an indexer fake whose `index(progress)` reports one progress value, then blocks on a `threading.Event` (always with a timeout):
- `test_start_answers_before_the_run_ends` (AC-01 unit level): `start()` returns True while the fake is still blocked.
- `test_second_start_is_refused_while_running` (AC-03 MCP part): a second `start()` returns False; `start_reindex_payload` gives `{"started": False, "reason": "an index run is in progress"}`.
- `test_start_is_refused_when_another_process_holds_the_lock`: `lock.handle().try_acquire()` first; `start()` is False and no thread starts.
- `test_progress_is_kept_while_running_and_cleared_after`: `progress()` equals the reported value while blocked; after the event is set and the thread joins, `progress()` is None and the lock is free.
- `test_a_failing_run_frees_the_lock_and_logs`: the fake raises `VaultUnavailable`; after the thread ends the lock is free, `progress()` is None, and the error is logged (caplog) rather than raised into the server.
- `test_status_reports_running_and_progress`: `status_payload(container, background)` has `running: True` and `progress: {done, total, embedded, skipped}` during the run.
- `test_status_without_background_has_no_progress` (AC-11 unit level): another holder has the lock; `status_payload(container)` has `running: True` and no `progress` key.
- `test_reindex_tool_returns_started` in `tests/mcp/test_server.py`: `build_server` on a fake container; calling the `reindex` tool through `mcp.call_tool` returns `{"started": true}`.

## Implementation notes
- `BackgroundRuns.start()`: `run = runs.begin(wait=False)`; None → False. Else start `threading.Thread(target=self._work, args=(run,), daemon=True, name="ariostea-index")`. Daemon: when the client closes stdin the server exits, the OS releases the flock, and the note being written rolls back as one transaction.
- `_work`: `run.execute(progress=self._keep)`, catching `Exception` and logging it with `logger.exception` (stderr, never stdout on stdio); finally clears the kept progress. Step 08 makes the failure visible under `last_run`.
- The kept progress is one reference assignment; reading it from the event-loop thread needs no lock.
- `status_payload` reads counts through `container.admin` (the reader connection, step 02) and `running` from `container.runs.state()`.
- The search side and the worker share the fastembed embedder; step 06 tests this from two threads (the spec's open risk).

## Verify
`uv run pytest tests/mcp tests/test_end_to_end.py -q` → all pass; then `uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass; the `reindex` tool no longer runs the index on the event loop.
- `uv run ruff check . && uv run ruff format --check .` is clean.
