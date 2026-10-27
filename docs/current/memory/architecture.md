# Architecture

Where the cycle's changes sit in the ports and adapters layout.

### server-robustness.07 · decision · 2026-10-27 · spec
A `RunLock` driven port (file adapter on `fcntl.flock`, in-memory fake, one contract suite) and an `IndexRuns` use case wrapping `IndexVault`: `run()` waits for the lock (CLI, `watch` passes), `start()` refuses or starts a worker thread (MCP), `state()` reports running, progress and the stored last run. The finish time comes from a `now` function injected at the composition root. The background run writes through its own store connection; WAL lets the server's connection read meanwhile.
**Why:** hexagonal lens, coordination between processes is an outside concern and gets a port with a real and a test adapter, so the use cases test against the fake. Simplicity lens, every writer goes through one path, so the lock, the run record and the progress cannot disagree between CLI, `watch` and MCP. Rejected: the lock inside `SqliteStore` (fewer pieces, but the store gets a second job and the use case cannot be tested with a fake lock alone).
Superseded by server-robustness.09 (who owns the worker thread).

### server-robustness.09 · pivot · 2026-10-27 · challenge round
Supersedes server-robustness.07 for the worker thread. `IndexRuns` starts no threads: `begin(wait, on_wait)` returns a `Run` holding the lock or `None`, and `Run.execute(progress)` indexes, stores the record and releases the lock. The MCP adapter calls `begin(wait=False)`, answers started or refused, and runs `execute` on a thread it owns, keeping the latest progress for `status`; CLI and `watch` call both inline. The rest of .07 stands.
**Why:** simplicity lens, a use case that spawns threads braids concurrency into the indexing rules, and every use-case test has to deal with timing. Hexagonal lens, how a run is delivered (inline, on a thread) is a driving-side concern, like the clock in vault-hardening.16. Rejected: `start()` in the use case (one object knows everything, but threads in the core).

### server-robustness.12 · discovery · 2026-10-27 · step 06
One fastembed embedder (bge-m3) used from two threads at once, one embedding notes and counting tokens and the other embedding queries, gives the same vectors (within 1e-5) and token counts as the same calls one after the other, and raises nothing. The MCP server's search thread and its index thread share the embedder with no lock (`test_embedder_from_two_threads_matches_one`).
**Why:** the open risk at Gate 0; ONNX Runtime sessions run concurrently and the tokenizer is only read. Rejected: a lock in the adapter (would make a search wait for a note's embedding, for no measured gain).
