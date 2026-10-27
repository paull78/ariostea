# Behaviour

How each fix behaves.

### server-robustness.02 · decision · 2026-10-27 · spec
The MCP `reindex` tool starts the run in the background and returns `{"started": true}` at once; `status` reports progress while it runs and the last run's result (notes, chunks, skipped, finish time, error) after. Searches read the index as it is during the run, each note switching when its own transaction commits.
**Why:** a first index takes about an hour on a large vault, longer than many clients wait for a tool call, and the run must not hold up searches. Rejected: keeping the call waiting on a worker thread (simpler for clients, but the call times out on long runs); removing the tool and leaving indexing to the CLI (simplest, but a client can no longer refresh the index).

### server-robustness.03 · decision · 2026-10-27 · spec
One index run at a time, held by an operating-system lock on `<index>.lock` for the whole run. The MCP `reindex` tool refuses at once when the lock is taken; CLI `reindex` and a `watch` pass say they are waiting, wait, then run; `watch` holds the lock only during a pass. The index uses sqlite's write-ahead log so readers never wait for a writer. Filled without asking: the lock is an OS lock released on process exit, so a crash cannot leave a stale lock.
**Why:** two runs at once embed the same notes twice, and one run's delete sweep can remove a note the other just added. A client call must not hang, so the MCP tool refuses; the CLI and `watch` have a person or a change behind them, so they wait. Rejected: refusing everywhere (a `watch` pass that finds the lock taken would drop changes until the next one); letting writers interleave under WAL and a long busy timeout (least code, but duplicate work and the sweep race).

### server-robustness.04 · decision · 2026-10-27 · spec
Each note row stores the fingerprint it was embedded with; a run keeps a note only when its content hash and its fingerprint both match, so an interrupted run, first index or change of model, keeps what it finished. Existing indexes get the column filled from the stored index-wide fingerprint, so no forced re-embed (setup.04). During an unfinished change of settings the index mixes notes from both, and search quality is mixed until the run completes.
**Why:** the work was never lost (each note commits on its own); only the index-wide fingerprint, written at the end, made the next run discard it. A per-note fingerprint states per note what was true. Rejected: writing the index-wide fingerprint at the start of a run on an empty index (fixes the first index only; after a model change it would either restart or mark old-model notes as current); accepting the restart.

### server-robustness.05 · decision · 2026-10-27 · spec
Every run that ends stores its result in the index (finish time, notes, chunks, skipped, error). `status` reads it, so it is right whatever process ran: `last_indexed` is the last completed run's finish time (ISO 8601 UTC, `null` before any), `last_run` the last run that ended, `running` comes from the run lock, and `progress` only from a run inside this server. Filled without asking: the ISO string in place of today's epoch float, which was never a real time anyway.
**Why:** CLI `reindex` and `watch` run in other processes, so state held in the server's memory would be wrong for hours. Rejected: tracking runs in the server only (less code, wrong status after any CLI run); dropping `last_indexed` (a client could not tell a stale index from a fresh one).

### server-robustness.08 · decision · 2026-10-27 · self-review
To rebuild or empty the index, the README says to delete the index file with its `-wal` and `-shm` files (`rm index.db*`) while nothing has it open. No new command.
**Why:** with the write-ahead log, a crash leaves the log file behind, and sqlite would apply it to a new index of the same name; deleting only `index.db`, as the README says today, becomes unsafe. Rejected: an `ariostea reset` command that takes the run lock first (safer while a server runs, but a new command for a rare action).

### server-robustness.10 · decision · 2026-10-27 · challenge round
The index-wide fingerprint is retired: the migration reads it once to fill `notes.fingerprint`, then deletes it. Each run record stores the fingerprint it ran with, and `status.config_fingerprint` reports the last completed run's. Refines server-robustness.04 and .05.
**Why:** simplicity lens, two records of one fact (index-wide and per note) can disagree, as they would halfway through a model change; one decides, the run record reports. Rejected: keeping the index-wide one for `status` only (less change, two fingerprints to keep in step).
