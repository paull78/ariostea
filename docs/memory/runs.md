# Index runs

Who writes the index and when: the run lock, the background `reindex`, run records and `status`.

### server-robustness.09 · decision · 2026-10-27 · challenge round
Every index run goes through the `IndexRuns` use case behind a `RunLock` driven port (`FileRunLock` on `fcntl.flock`, `InMemoryRunLock` for tests, one contract suite). `begin(wait, on_wait)` returns a `Run` holding the lock, or `None` when another holds it and `wait` is false; `Run.execute(progress)` indexes, records the run and releases the lock. `IndexRuns` starts no threads: CLI `reindex` and `watch` passes run inline, and the MCP adapter (`mcp/background.py`) owns the worker thread and the latest progress. The clock is passed in at the composition root. Merges server-robustness.07 (the port and use case).
**Why:** every writer on one path, so the lock, the record and the progress cannot disagree. Simplicity lens: threads in the use case would braid concurrency into the indexing rules and timing into every test. Hexagonal lens: how a run is delivered is a driving-side concern, like the clock (vault-hardening.16). Rejected: the lock inside `SqliteStore`; `start()` in the use case.

### server-robustness.02 · decision · 2026-10-27 · spec
The MCP `reindex` tool starts the run in the background and returns `{"started": true}` at once; `status` shows its progress and, after, the last run. Searches read the index as it is, each note switching when its own transaction commits. Cause of the old blocking: FastMCP 1.27 calls a sync tool directly on its event loop, so a long tool call held up every request.
**Why:** a first index takes about an hour on a large vault, longer than many clients wait. Rejected: a waiting call on a worker thread (times out); removing the tool.

### server-robustness.03 · decision · 2026-10-27 · spec
One index run at a time, through the flock on `<index>.lock`, released by the operating system however the process ends. The MCP tool refuses while a run goes on (`{"started": false, "reason": ...}`); CLI `reindex` and each `watch` pass print `waiting for the index run in progress`, wait, then run; an idle `watch` holds nothing. The index runs on sqlite's write-ahead log, and the composition root opens two connections: a writer (`any_thread=True`) used only by the run holding the lock, and a reader for search, `get_note` and `status`.
**Why:** two runs at once embed notes twice and race in the delete sweep; a client call must not hang, a person or a change is behind the CLI and `watch`. Rejected: refusing everywhere (a `watch` pass would drop changes); interleaving writers under WAL (duplicate work, the sweep race).

### server-robustness.05 · decision · 2026-10-27 · spec
Every run that ends, from any process, stores its record in the index; `status` reads it: `last_indexed` (ISO 8601 UTC, or null) and `config_fingerprint` from the last completed run, `last_run` (finished_at, notes, chunks, skipped, error) the last that ended, `running` from the lock, `progress` only for a run inside this server. A failed run records its error and leaves `last_indexed` alone; a killed run records nothing.
**Why:** CLI and `watch` run in other processes, so state in the server's memory would be wrong for hours. Rejected: tracking runs in the server only; dropping `last_indexed`.

### server-robustness.13 · decision · 2026-10-27 · step 08
The index keeps two run records in a `runs` table, the last run that ended and the last that completed, each replaced in place. `IndexRuns` records through the writer before releasing the lock; `state()` reads through the reader. A failed run records the index's counts after the failure and 0 skipped.
**Why:** `status` needs only those two, and `watch` runs a pass per change, so a history would grow for nothing. Rejected: an append-only table; JSON in `meta`.

### server-robustness.14 · issue · 2026-10-27 · step 08
On an index migrated from before server-robustness, `config_fingerprint` and `last_indexed` are null until the first run completes, and `eval/run_blurb_eval.py --reuse-index` aborts until then. One `reindex` clears it and embeds nothing when the settings are unchanged. Open, low.
**Why:** a migrated index has no completed run (AC-09, AC-10); a synthetic record would report a finish time that never happened.
