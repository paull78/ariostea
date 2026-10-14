# Step 07 · Store adds columns missing from an old index

**Milestone:** 2 · **Needs:** 05 · **Covers:** AC-03 (behaviour for real old indexes)

## Goal
`SqliteStore` opened on an index created before the `context_blurb` column existed adds the column, so the next reindex succeeds on any model (bge-m3-default.09). Indexes that already have it are untouched.

## Files touched
- `src/ariostea/adapters/store/sqlite_store.py`: after the `CREATE ... IF NOT EXISTS` script, add each column of `chunks` that the existing table lacks.
- `tests/adapters/store/test_sqlite_store.py`: tests on a database built with the old schema.
- `tests/indexing/test_index_vault.py`: the owner's failure reproduced through `IndexVault`.

## Tests to write first
A helper `_old_index(path)` creates the schema the owner's index has, copied from it: `notes` as today, `chunks` without `context_blurb`, `chunks_vec` at a given size, and no `chunks_fts` or `meta`.
- `test_opening_an_index_without_context_blurb_adds_the_column`: after `SqliteStore(path, dim=3)` on `_old_index`, `PRAGMA table_info(chunks)` lists `context_blurb`.
- `test_an_old_index_takes_chunks_with_blurbs`: on `_old_index`, `upsert_note` with a blurbed chunk succeeds, and dense retrieval returns the chunk with its blurb.
- `test_reindexing_an_old_index_with_a_new_model_succeeds` (indexing): `_old_index` at size 2 holding one note, then `IndexVault` with a fake embedder of size 3; no exception, every vector has size 3, and sparse retrieval finds a word from the vault (the FTS table was created and filled).

## Implementation notes
- Keep a small list of `(column, type)` added to `chunks` after the first release: today only `("context_blurb", "TEXT")`. For each one missing from `PRAGMA table_info(chunks)`, run `ALTER TABLE chunks ADD COLUMN`. sqlite allows adding a nullable column without a rewrite.
- Run it after the create script, so a fresh database (which already has the column) skips it.
- The missing `chunks_fts` and `meta` tables are created empty by the existing script; the cleared fingerprint forces every note to re-embed, which fills FTS. The indexing test checks that path.

## Verify
`uv run pytest tests/adapters/store tests/indexing -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
- A reindex on a scratchpad copy of the owner's index, with the default model, succeeds.
- manual: the owner's real vault reindexes on the new default (repeat of the gate M2 check): `uv run ariostea reindex`, then `uv run ariostea status` shows a fingerprint starting `fastembed:BAAI/bge-m3`.
