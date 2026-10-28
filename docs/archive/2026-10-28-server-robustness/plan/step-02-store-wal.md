# Step 02 · Store on WAL, two connections

**Milestone:** 1 · **Needs:** none · **Covers:** AC-05

## Goal
The index runs on sqlite's write-ahead log, and the composition root opens two store connections on it: a writer that any one thread at a time may use (the run holding the lock), and a reader for everything else. A read never waits for a write.

## Files touched
- `src/ariostea/adapters/store/sqlite_store.py`: `SqliteStore(path, dim, *, any_thread=False)`; `PRAGMA journal_mode=WAL` right after connecting, before the schema.
- `src/ariostea/config/container.py`: build `writer = SqliteStore(..., any_thread=True)` for `IndexVault`, `reader = SqliteStore(...)` for search, `GetNote`, `SearchSources` and `admin`.
- `tests/adapters/store/test_sqlite_store.py`: new tests below.
- `tests/config/test_container.py`: the container test below.

## Tests to write first
- `test_index_uses_the_write_ahead_log`: after opening, `PRAGMA journal_mode` on a fresh `sqlite3` connection to the file returns `wal`.
- `test_old_index_switches_to_the_write_ahead_log`: an index built with `old_index` (rollback journal) reads `wal` after `SqliteStore` opens it.
- `test_reader_reads_while_a_write_is_open` (AC-05): writer connection runs `BEGIN EXCLUSIVE` and inserts a note row, uncommitted; a second `SqliteStore` on the same file returns `stats()` and `known_hashes()` within 1 s with the committed content only. Fails on a rollback journal ("database is locked" after the busy timeout).
- `test_any_thread_store_works_from_another_thread`: a store opened with `any_thread=True` on the main thread answers `stats()` from a `threading.Thread`; one opened without it raises `sqlite3.ProgrammingError` there.
- `test_container_reads_and_writes_through_separate_connections` (in `test_container.py`, with the fake embedder setup the file already uses): `container.admin` is not the store the indexer writes through (compare `id`s via the private attribute the test can reach, or check that the indexer's store has `any_thread` set and the admin's does not).

## Implementation notes
- `sqlite3.connect(path, check_same_thread=not any_thread)`. The writer is used by one thread at a time because only the run holding the run lock uses it (server-robustness.03); `check_same_thread=False` only lets that thread differ from the opening one.
- `journal_mode=WAL` is persistent in the file; setting it on an index already in WAL is a no-op. It adds `index.db-wal` and `index.db-shm` (README in step 10, server-robustness.08).
- Keep the 5 s default busy timeout: schema checks at open can still meet a commit in progress.
- Both connections run `_init_schema`; the second sees the tables made by the first.
- Don't change any query. Search code reads through `reader` exactly as before.

## Verify
`uv run pytest tests/adapters/store tests/config -q` → all pass; then `uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass and the fast suite is green.
- `uv run ruff check . && uv run ruff format --check .` is clean.
