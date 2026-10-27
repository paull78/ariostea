# Step 08 · Run records

**Milestone:** 2 · **Needs:** 07 · **Covers:** AC-02, AC-10 (store and use case)

## Goal
Every run that ends, completed or failed, stores its record in the index: finish time, fingerprint, notes, chunks, skipped, error. `IndexRuns.state()` gives `running`, the last run and the last completed run. `stats()` stops inventing a time.

## Files touched
- `src/ariostea/domain/models.py`: `RunRecord(finished_at: datetime, fingerprint: str, notes: int, chunks: int, skipped: int, error: str | None)`, frozen; `IndexStats` loses `last_indexed` and `config_fingerprint`.
- `src/ariostea/ports/store.py`: `IndexAdmin.record_run(record)`, `last_run() -> RunRecord | None`, `last_completed_run() -> RunRecord | None`.
- `src/ariostea/adapters/store/sqlite_store.py`: table `runs(kind TEXT PRIMARY KEY, finished_at TEXT, fingerprint TEXT, notes INTEGER, chunks INTEGER, skipped INTEGER, error TEXT)`, rows `last` and `completed`; `finished_at` stored as ISO 8601 UTC.
- `src/ariostea/indexing/index_runs.py`: `IndexRuns(indexer, lock, writer: IndexAdmin, reader: IndexAdmin, now: Callable[[], datetime])`; `Run.execute` records the run before releasing the lock; `RunState(running, last_run, last_completed)`.
- `src/ariostea/config/container.py`: passes the writer and reader stores and `now=lambda: datetime.now(timezone.utc)`.
- `eval/run_blurb_eval.py`: reads the fingerprint from the last completed run, treating none as `""`.
- Tests: `tests/adapters/store/test_sqlite_store.py`, `tests/indexing/test_index_runs.py`, fakes in `tests/indexing/*`, `tests/mcp/*`, `tests/domain/test_models.py`.

## Tests to write first
Store:
- `test_no_runs_on_a_new_index`: `last_run()` and `last_completed_run()` are None.
- `test_completed_run_is_last_and_last_completed`: both return the record, `finished_at` equal and timezone-aware UTC.
- `test_failed_run_leaves_last_completed_alone`: after a completed then a failed record, `last_run().error` is set and `last_completed_run()` is the completed one.
- `test_runs_are_seen_by_another_connection`: a record written through one `SqliteStore` is read through a second on the same file.

Use case, with fakes and a fixed `now`:
- `test_execute_records_a_completed_run`: after `execute()`, `state().last_run` has the fixed time, the indexer's fingerprint, the report's notes, chunks and skipped count, and `error` None; `last_completed` is the same.
- `test_execute_records_a_failed_run` (AC-02 failure part): a vault guard failure; `execute()` raises; `last_run.error` holds the message, `last_completed` is still the previous run, the index unchanged.
- `test_record_is_written_before_the_lock_is_released`: a lock fake records the order of `record_run` and `release`.
- `test_any_exception_is_recorded_and_reraised`: an embedder raising `RuntimeError("boom")` gives `last_run.error == "boom"` and the exception propagates.
- `test_state_reads_through_the_reader`: `state()` calls only the reader fake.

## Implementation notes
- The writer records (it is the connection the run holds); the reader serves `state()`, which the MCP server calls from its own thread while a run writes (step 02).
- A failed run's `notes` and `chunks` are the index's counts after the failure (`writer.stats()`), `skipped` 0.
- A run killed partway records nothing; `last_run` stays the one before.
- `IndexStats` callers: `status_payload`, the store tests, `tests/eval/test_wiki_index.py` (`.notes` only, unchanged).
- `now` is passed in at the composition root, so `indexing/` stays free of the clock (vault-hardening.16).

## Verify
`uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass; `time.time()` is gone from `sqlite_store.py`.
- `uv run ruff check . && uv run ruff format --check .` is clean.
