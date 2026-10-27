# Recap: server-robustness

Feature cycle on `feature/server-robustness`, started on 2026-10-27 and finished on 2026-10-28. Source: the four server problems vault-hardening deferred (vault-hardening.01, its recap's open issues).

## What was built

The index can now be written while it is in use. All four fixes in the spec were built (server-robustness.01):

- The MCP `reindex` tool starts a run on a background thread and returns `{"started": true}` at once. Searches, `get_note` and `status` are answered throughout, and `status` shows the run's progress while it goes on (server-robustness.02). On the awkward vault, the call returns in under 1 second and a search is answered mid-run.
- One index run goes on at a time across every process, held by an flock on `<index>.lock` that the operating system releases however the process ends. The MCP tool refuses while a run goes on; `ariostea reindex` and each `watch` pass print `waiting for the index run in progress`, wait, then run; an idle `watch` holds nothing. The index runs on sqlite's write-ahead log, so readers never wait for the writer, and no test output contains "database is locked" (server-robustness.03).
- Each note records the fingerprint it was embedded with, and a run keeps a note only when its hash and fingerprint both match. A run stopped partway, on a first index or after a change of model or chunking, keeps every note it finished: a 30-note first index stopped after 10 embeds only the other 20 on the rerun. An index from the previous release opens with every note given its stored index-wide fingerprint, which is then deleted, and the next run embeds nothing (server-robustness.04, .10).
- Every run that ends stores its record in the index. `status` reports `last_indexed` (ISO 8601 UTC, or null) and `config_fingerprint` from the last completed run, `last_run` with its error if it failed, and `running` from the lock, so a server started before a CLI run reports that run correctly. `progress` appears only for a run inside the server (server-robustness.05, .13).
- The README describes `reindex` as starting a run, lists the `status` fields, and says to delete the index with its `-wal` and `-shm` files (`rm index.db*`) with nothing running; it mentions the `.lock` file and the resume behaviour (server-robustness.08).

Running the index is one use case, `IndexRuns`, behind a new `RunLock` driven port with a file adapter and an in-memory fake that share one contract suite. `IndexRuns` starts no threads: the CLI and `watch` run inline, and the MCP adapter owns the worker thread and the latest progress (server-robustness.07, .09). The composition root opens two store connections, a writer for the run holding the lock and a reader for everything else, and passes in the clock.

Deviations from the spec and plan:

- AC-10 asks for `last_indexed` within 1 second of the CLI's end. The integration test allows 2 seconds: `last_indexed` is truncated to the second, so up to 1 second of the gap is the format, and the rest is the CLI process exiting after its run.
- `ARCHITECTURE.md` was updated too (the indexing flow, the fingerprint section, the ports table), though AC-13 names only the README: it described the retired index-wide fingerprint.
- Step 08 took the status payload code and its unit tests from step 09, since dropping `last_indexed` from `IndexStats` broke `status_payload` at once.
- `reindex_payload` stays synchronous (it waits for the lock), so the CLI and the eval scripts call it unchanged; `eval/run_blurb_eval.py` reads the stored fingerprint from the last completed run.

## How it went

Two milestones. Milestone 1 (a reindex that doesn't block) passed its gate on 2026-10-27, with the full suite at 1,308 passed and 1 skipped in 29 minutes. Milestone 2 (resume and an honest status) passed the final gate on 2026-10-28 with 1,328 passed and 1 skipped in 32 minutes. At the final gate, an index built in step 06 by the code before per-note fingerprints was opened with the new code: every note took the stored fingerprint, `status` read null until the next `reindex`, and that run embedded nothing.

The design moved twice in the challenge round, before any code. The first architecture had `IndexRuns` start the worker thread itself (server-robustness.07). The simplicity and hexagonal lenses moved the thread to the MCP adapter, as the clock had moved to the CLI in vault-hardening, so the use case and its tests have no threads or timing in them (server-robustness.09). The same round retired the index-wide fingerprint: kept beside the per-note ones, the two records would disagree halfway through a change of model (server-robustness.10).

The build ran without a pivot. The open risk at Gate 0, one fastembed embedder used from the server's thread and the worker at once, was tested first in step 06 and held: the same vectors and token counts as one thread, so no lock was added (server-robustness.12). The first index of the awkward vault was timed before the integration tests were written, at 29 seconds (8 to load the models, 20 to index), which left room for AC-01's 10 seconds without enlarging the vault.

Every step's new tests were run against a broken version of the code once: a blocking `BackgroundRuns.start`, a lock probe that never sees another holder, a keep rule that ignores the fingerprint, a failed run recorded as completed. Each break failed at least one test. The 29-minute gate suite ran on a snapshot of the milestone's last commit in a separate worktree, while steps 07 and 08 went on in the main tree.

## Decisions that stand

- server-robustness.01 · the cycle fixes the four deferred server problems and leaves retrieval alone · all four are about who writes the index and when
- server-robustness.02 · MCP `reindex` starts a background run and returns at once · a first index can take an hour, longer than a client waits
- server-robustness.03 · one run at a time through an flock on `<index>.lock`; MCP refuses, CLI and `watch` wait; WAL for readers · two runs at once duplicate work and race in the delete sweep
- server-robustness.04 · each note records its fingerprint; a note is kept only when hash and fingerprint match · makes an interrupted run resumable, including after a model change
- server-robustness.05 · runs store their records in the index and `status` reads them · CLI and `watch` run in other processes
- server-robustness.06 · non-goals: cancel, queueing, cross-process progress, an old-settings count, Windows, other deferred items, retrieval · the owner kept the cycle to the four fixes
- server-robustness.07 · a `RunLock` driven port with file and in-memory adapters, and an `IndexRuns` use case every writer goes through (thread ownership per .09) · the lock, the record and the progress cannot disagree between writers
- server-robustness.08 · to rebuild or empty the index, delete it with its `-wal` and `-shm` files with nothing running · a leftover log would be applied to a new index of the same name
- server-robustness.09 · `IndexRuns` starts no threads; the MCP adapter owns the worker thread · delivery is a driving-side concern, and the use case stays free of timing
- server-robustness.10 · the index-wide fingerprint is retired; run records carry the fingerprint for `status` · one record per fact
- server-robustness.13 · two run records (last, last completed), not a history · `watch` would grow a history without bound

## Discoveries

- server-robustness.12 · one fastembed embedder is safe to use from two threads at once: same vectors and token counts, nothing raised.
- FastMCP 1.27 calls a sync tool directly on its event loop, so a long tool call holds up every other request; this was the cause of the blocking `reindex` (server-robustness.01).

## Open issues and known limitations

- server-robustness.14 · on an index migrated from the previous release, `config_fingerprint` and `last_indexed` are null until the first run completes, and `run_blurb_eval.py --reuse-index` aborts until then. One `reindex` clears it and embeds nothing when the settings are unchanged. Open, low.
- Until an interrupted change of model or chunking finishes, the index holds notes from both settings and search quality is mixed (documented in the README).
- The run lock is a POSIX flock, so the project still runs on macOS and Linux only.
- Cancelling a run, queueing `reindex` calls and progress for runs in other processes stay out of scope (server-robustness.06).

## Takeaways

The challenge round paid for itself again: moving the thread out of the use case before any code existed meant `IndexRuns` was tested with no threads and no timing, and only one small adapter class deals with concurrency. Testing the Gate 0 risk first, in a test of its own, closed it with evidence instead of a lock added just in case.

Running the long gate suite on a snapshot worktree kept the build moving for half an hour that would otherwise have been waiting. Timing one expensive run before writing the integration tests set their deadlines once, and none of them needed changing.

To do differently: when a step changes a shared domain type, check which later step's code it breaks before writing the plan, so step boundaries don't move mid-build as steps 08 and 09 did.
