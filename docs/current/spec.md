# Spec: server-robustness

Cycle: server-robustness · Type: feature · Branch: feature/server-robustness · Status: approved

## Goal
Make the index safe to write while it is being used. A reindex started from an MCP client must not stop the server answering, an interrupted index must pick up where it stopped, two writers must never fail with "database is locked", and `status` must say when the index was last brought up to date. Retrieval and the measured defaults stay as they are.

Source: the non-goals deferred by vault-hardening (vault-hardening.01, recap "Open issues").

## Scope and non-goals
In scope, four fixes:
1. The MCP `reindex` tool no longer blocks the server.
2. An interrupted index (first index, or after a change of model or chunking) resumes instead of re-embedding every note.
3. Concurrent writers (`watch`, CLI `reindex`, MCP `reindex`) never fail with "database is locked".
4. `status.last_indexed` reports when an index run last finished, not the current time.

Non-goals: cancelling a run from a client; queueing MCP `reindex` calls (a second call is refused while a run goes on); progress for runs in other processes (`status` shows only that one is running); a count in `status` of notes embedded under other settings; Windows (the run lock is a POSIX lock, and the project runs on macOS); the other items vault-hardening deferred (symlinked folders, `.canvas` and `.base` files, unused config keys, Claude Desktop docs); retrieval changes of any kind.

## Behaviour
### Fix 1: `reindex` from a client starts a run and returns
The MCP `reindex` tool starts an index run in the background and returns at once with `{"started": true}`. While the run goes on, the server answers every other request: searches and `get_note` read the index as it is, and each note switches from its old chunks to its new ones in one step, when its write commits. `status` reports the run while it goes on (`running: true` and its progress, done and total notes, embedded and skipped so far) and, once it ends, the last run's result: notes, chunks, skipped files, when it finished, and its error if it failed (a vault guard, for example).

```
client: reindex()          -> {"started": true}
client: search_knowledge() -> answered from the index as it is
client: status()           -> {"running": true, "progress": {done, total, embedded, skipped}, ...}
   ... the run ends ...
client: status()           -> {"running": false, "last_run": {notes, chunks, skipped, finished_at, error}}
```

### Fix 2: an interrupted run resumes
Each note in the index records the fingerprint (embedding model, contextualizer, chunking policy) it was embedded with. A run keeps a note when both its content hash and its fingerprint match the current ones, and embeds it again otherwise, storing the current fingerprint with it. A run killed partway, on a first index or after a change of model or chunking, therefore keeps every note it finished, and the next run embeds only the rest. An existing index gets the new column filled from the index-wide fingerprint it already stores, so nobody re-embeds on upgrade; the index-wide fingerprint is then deleted, and the per-note fingerprints are the only record of what each note was embedded with.

```
notes: path | content_hash | fingerprint
each note: hash same and fingerprint same -> keep
           otherwise                      -> embed, store with the current fingerprint
killed at 412/3410 -> the next run keeps 412 and embeds the rest
```

Until an interrupted change of model or chunking is finished, the index holds notes from both settings, and search quality is mixed (a change of vector size still drops every vector at once, as today, bge-m3-default.01).

### Fix 3: one run at a time, readers never wait
An index run holds a lock on a file next to the index (`<index>.lock`) from start to end. It is an operating-system lock, released when the process ends for any reason, so a crash never leaves a stale lock. A writer that finds the lock taken:
- the MCP `reindex` tool returns at once with `{"started": false, "reason": "an index run is in progress"}`;
- CLI `reindex` prints one line on stderr saying it is waiting for the run in progress, waits, then runs;
- a `watch` pass waits the same way, then runs; `watch` holds the lock only during a pass, not while idle.

The index switches to sqlite's write-ahead log, so searches, `get_note` and `status` read while a run writes and never wait for it. An existing index is switched when it is next opened; the log adds two files next to the index (`-wal`, `-shm`).

```
index.db.lock     held for one run, released on exit or crash
watch pass   -----[run]-----------------
MCP reindex       -> {"started": false, "reason": ...}
CLI reindex       "waiting for the index run in progress" ... [run]
searches     answered throughout (write-ahead log)
```

### Fix 4: `status` reports runs from the index
Every run that ends, from any process (CLI `reindex`, a `watch` pass, the MCP tool), stores its result in the index: when it finished, the fingerprint it ran with, notes, chunks, skipped files, and its error if it failed. `status`, over MCP and from the CLI, reads it from there:
- `last_indexed` is the finish time of the last run that completed, as an ISO 8601 UTC string, or `null` if none ever did; a failed run leaves it alone;
- `config_fingerprint` is the fingerprint of the last run that completed, the settings the whole index was last brought up to date with, or `null` if none ever did;
- `last_run` is the last run that ended, completed or failed, with its error;
- `running` is true while any process holds the run lock (fix 3);
- `progress` (done and total notes, embedded and skipped so far) appears only for a run inside this server.

A run killed partway stores nothing; `last_run` stays the one before it.

```
status() -> { notes, chunks, config_fingerprint,
              last_indexed: "2026-10-27T09:41:05Z",
              running: true,                       <- the lock, any process
              progress: {done, total, embedded, skipped},   <- only a run in this server
              last_run: {finished_at, notes, chunks, skipped, error} }
```

### Docs
The README's tools table says that `reindex` starts a run and returns, and lists the `status` fields. Where it says to delete the index file (a full rebuild, emptying the index on purpose), it says to delete the index file together with its `-wal` and `-shm` files, in one line (`rm index.db*`), with no server or `watch` running: after a crash the log file stays behind, and sqlite would apply it to a new index of the same name. It mentions the `.lock` file next to the index.

## Architecture
Running the index becomes one use case that every writer goes through, and the lock becomes a driven port.

```
MCP reindex: run = runs.begin(wait=False)        None -> {"started": false}
             thread(run.execute, progress=keep)  -> {"started": true}   (the adapter owns the thread)
CLI reindex, watch pass: runs.begin(wait=True, on_wait=print).execute(progress=printer)

IndexRuns --begin--> RunLock port: FileRunLock (fcntl.flock on <index>.lock), InMemoryRunLock
Run.execute -------> IndexVault (as today, per-note fingerprint)
            -------> store: record_run / last_run, notes.fingerprint, WAL
now(): passed in by the composition root; IndexRuns starts no threads
```

- `ports/run_lock.py` defines `RunLock`: take the lock or report it taken, wait for it, and tell whether any process holds it. `adapters/lock/file.py` implements it with `fcntl.flock`, released by the operating system when the process ends; an in-memory fake follows the same contract suite (setup.02, vault-hardening.19).
- `indexing/index_runs.py`: `IndexRuns` wraps `IndexVault` and starts no threads. `begin(wait, on_wait)` returns a `Run` holding the lock, or `None` when another run holds it and `wait` is false; with `wait`, it calls `on_wait` once and blocks until the lock is free. `Run.execute(progress)` indexes, stores the run record (completed or failed) and releases the lock. `state()` gives `running` (the lock, any process) and the stored last run. The finish time comes from a `now` function passed in at the composition root, so time stays outside the app (vault-hardening.16).
- The MCP adapter owns the worker thread: `reindex` calls `begin(wait=False)`, answers at once, and runs `execute` on a thread it starts, keeping the latest progress value for `status`. CLI `reindex` and `watch` passes call `begin(wait=True)` and `execute` inline.
- `WatchVault` runs its passes through `IndexRuns`; the Container exposes `IndexRuns` in place of the bare indexer (setup.01).
- The store: `notes.fingerprint`, added to old indexes and filled from the stored fingerprint (the column migration, bge-m3-default.09, now covers `notes` too); `record_run` and `last_run` on `IndexAdmin` in place of `fingerprint` and `set_fingerprint`; `journal_mode=WAL`. `stats()` stops inventing `last_indexed`. A change of vector size still drops the vector table (bge-m3-default.01) and has no fingerprint left to clear: every note's own fingerprint already differs from the new model's.
- The composition root opens two store connections: one that only the run holding the lock uses for writing, opened so that another thread may use it, and one for everything else (search, `get_note`, `status`, the last run record). The write-ahead log lets the first write while the second reads.
- The dependency-rule test (vault-hardening.17) covers the new modules unchanged: `indexing/` imports only ports and the standard library.

## Acceptance criteria
- AC-01 · Over stdio, a `reindex` call on a vault whose indexing takes at least 10 seconds returns `{"started": true}` within 1 second, and a `search_knowledge` call sent right after it is answered before the run ends.
- AC-02 · While the run goes on, `status` reports `running: true` with done and total notes; after it ends, `status` reports `running: false` and the run's notes, chunks, skipped files and finish time under `last_run`. A run that fails its vault guard reports the error under `last_run`, leaves `last_indexed` as it was, and leaves the index unchanged.
- AC-03 · With a run in progress in one process, an MCP `reindex` in another returns `{"started": false}` with the reason within 1 second, and a CLI `reindex` prints the waiting line, waits until the first run ends, then completes with exit status 0; neither prints "database is locked".
- AC-04 · A process killed with SIGKILL during a run leaves no lock behind: the next `reindex` starts at once.
- AC-05 · While a run is writing, `search_knowledge`, `get_note` and `status` from another process are answered without waiting for the run to end.
- AC-06 · `watch` running and idle does not hold the lock: a CLI `reindex` started while `watch` is idle runs at once; a `watch` pass triggered while a CLI run is in progress waits and then runs.
- AC-07 · A first index of 30 notes stopped after 10 have been stored (the process killed, or an exception from the embedder), then run again, embeds only the 20 the first run had not stored.
- AC-08 · On an index built under fingerprint A, a run under fingerprint B stopped partway, then run again under B, re-embeds only the notes still on A; at the end every note records B.
- AC-09 · An index created before this cycle opens without error, every note takes the index-wide fingerprint it stored, the index-wide fingerprint is gone afterwards, and the next run with unchanged settings embeds nothing.
- AC-10 · `status` on a new index reports `last_indexed: null`; after a CLI `reindex` that finishes, a server started earlier reports that run's finish time (within a second of the CLI's end) under `last_indexed`, its fingerprint under `config_fingerprint` and its counts under `last_run`; two `status` calls with no run between them report the same `last_indexed`.
- AC-11 · While a CLI `reindex` runs in another process, `status` from the server reports `running: true` and no `progress`.
- AC-12 · The dependency-rule test passes with `IndexRuns` in `indexing/`, and `RunLock`'s contract suite passes on both the file adapter and the in-memory fake.
- AC-13 · The README tools table describes `reindex` as starting a run and lists the `status` fields; every place that says to delete the index file names the `-wal` and `-shm` files too.

## Testing approach
- `IndexRuns` runs against `InMemoryRunLock`, `InMemoryVault` and the existing test embedder, with no threads: `begin` refusing and waiting, `Run.execute` storing the record of a completed and a failed run and releasing the lock, progress, `state()`. The MCP handler's thread is tested at the handler level: `reindex` answers before `execute` ends.
- `RunLock` has one contract suite run on both adapters. The file adapter is also tested across processes: a subprocess holds the lock, the test sees it taken, kills the subprocess with SIGKILL and takes the lock at once (AC-03, AC-04).
- The store runs on temp files: the per-note fingerprint, `record_run` and `last_run`, WAL mode, a read from a second connection while a write is open (AC-05), and the migration of an index copied from the previous release (AC-09), since only a database built by an earlier release shows a missing column (bge-m3-default.09).
- Resume: an embedder that raises after 10 notes stops a run partway; the rerun embeds only the rest (AC-07, AC-08).
- The MCP handlers and the CLI with fakes: `reindex` payloads, `status` fields, the waiting line.
- Integration (marked `integration`, real models), over stdio on the awkward vault from vault-hardening: `reindex` returns at once and a search is answered mid-run, `status` during and after a run, a CLI run seen by a server started before it (AC-01, AC-02, AC-05, AC-10, AC-11), and a CLI `reindex` and a `watch` pass waiting on a held lock (AC-03, AC-06).
- The dependency-rule test covers the new modules (AC-12).

Test command: `uv run pytest` (full suite, about 25 minutes); `uv run pytest -m "not integration"` for the fast suite.

## Open questions
- none
