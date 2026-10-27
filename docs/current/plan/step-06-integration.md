# Step 06 · Cross-process integration tests

**Milestone:** 1 · **Needs:** 04, 05 · **Covers:** AC-01, AC-03, AC-05, AC-06

## Goal
The fixes of milestone 1 proven on the real CLI and stdio server with the default models, on the awkward vault from vault-hardening, and the shared embedder checked from two threads.

## Files touched
- `tests/test_server_robustness.py`: new, `pytestmark = pytest.mark.integration`.
- `tests/test_awkward_vault.py`: `test_reindex_over_stdio_*` (the raw JSON-RPC test) now expects `{"started": true}` and polls `status` until `running` is false; whatever else it asserts about stdout staying protocol-only stays.
- `tests/adapters/embedding/test_fastembed_local.py`: the two-thread test.
- `src/ariostea/adapters/embedding/fastembed_local.py`: only if the two-thread test fails (then a `threading.Lock` around model calls, recorded as a decision).

## Tests to write first
Shared fixture: build the awkward vault (`fixtures.awkward_vault.build`) once per module with a config whose store is a fresh, empty index, so a first index takes about a minute (well over the 10 s AC-01 needs). A helper starts `ariostea serve` over stdio with the `mcp` client, as `_over_mcp` in `test_awkward_vault.py` does, but keeps the session open across steps. A helper holds the run lock from a subprocess (`FileRunLock(store + ".lock").acquire()`, print `held`, sleep until killed).

- `test_reindex_returns_at_once_and_search_answers_mid_run` (AC-01, AC-05 same process): on the empty index, `reindex` returns `{"started": true}` in under 1 s; `status` then shows `running: true` and `progress.total` > 0; a `search_knowledge` call returns while `status` still shows `running: true`; poll `status` until `running` is false (timeout 900 s).
- `test_second_writer_is_refused_or_waits` (AC-03): while the lock helper process holds the lock, a server's `reindex` returns `{"started": false, "reason": ...}` in under 1 s; a CLI `reindex` subprocess prints `waiting for the index run in progress` on stderr and has not exited 2 s later; after the helper is killed it exits 0. Neither output contains `database is locked`.
- `test_reads_answer_while_another_process_writes` (AC-05): during a CLI first index (fresh store) in a subprocess, a server on the same store answers `search_knowledge`, `get_note` and `status` each in under 5 s (the reranker's first load excluded by a warm-up search before the CLI starts).
- `test_idle_watch_holds_no_lock_and_waits_for_a_run` (AC-06): start `ariostea watch` on the indexed store; after its first pass, `FileRunLock(...).try_acquire()` from the test succeeds (release it); with the lock helper holding the lock, edit a note; `watch` prints the waiting line; kill the helper; the edit reaches the index within 10 s.
- `test_embedder_from_two_threads_matches_one` (in `test_fastembed_local.py`, integration): the default model embeds 40 note texts in one thread while another thread embeds 40 queries; the vectors equal (`numpy.allclose`, atol 1e-5) those from the same calls made one after the other, and nothing raises.

## Implementation notes
- Time the first index of the awkward vault once before writing the timeouts (general lesson: time one run of an expensive step before budgeting the rest).
- Stop every subprocess by PID (`terminate`/`kill` on the `Popen`), never by name.
- Every wait has a deadline; a failing test must not hang the suite.
- The SIGKILL check of AC-04 is already covered by step 01's subprocess test on the file adapter; no real-model run needed.

## Verify
`uv run pytest tests/test_server_robustness.py tests/test_awkward_vault.py tests/adapters/embedding -q` → all pass (several minutes); then `uv run pytest -m "not integration" -q` → all pass.

## Done when
- The integration tests pass; no output anywhere contains `database is locked`.
- manual: at the milestone gate, start `ariostea serve` from an MCP client (or the scripted client), call `reindex` on a fresh index, search while it runs, and read `status`.
