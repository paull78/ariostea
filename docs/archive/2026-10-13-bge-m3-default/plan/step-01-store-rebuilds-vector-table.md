# Step 01 · Store rebuilds the vector table on a size change

**Milestone:** 1 · **Needs:** none · **Covers:** AC-02

## Goal
`SqliteStore` opened with a dimension different from the one its `chunks_vec` table was created with drops and recreates that table at the new size, and clears the stored fingerprint so the next index re-embeds every note. Opened at the same size, nothing changes (bge-m3-default.01).

## Files touched
- `src/ariostea/adapters/store/sqlite_store.py`: read the existing table's size in `_init_schema`, rebuild on mismatch.
- `tests/adapters/store/test_sqlite_store.py`: new tests.

## Tests to write first
- `test_reopening_at_a_new_dimension_accepts_vectors_of_that_size`: create at dim 3, upsert one note, close; reopen at dim 4; upsert a note with 4-dimension vectors; dense retrieval returns it.
- `test_reopening_at_a_new_dimension_clears_the_fingerprint`: set fingerprint `"fake:v1"` at dim 3; reopen at dim 4; `fingerprint()` is `""`.
- `test_reopening_at_the_same_dimension_keeps_vectors_and_fingerprint`: create at dim 3, upsert, set fingerprint; reopen at dim 3; dense retrieval still finds the note and the fingerprint is unchanged.

## Implementation notes
- Read the size from the table's DDL: `SELECT sql FROM sqlite_master WHERE name = 'chunks_vec'`, regex `FLOAT\[(\d+)\]`. An empty table has no rows to measure, so the DDL is the reliable source.
- On mismatch: `DROP TABLE chunks_vec`, recreate at the new size, `DELETE FROM meta WHERE key = 'fingerprint'`. Chunk rows stay; the cleared fingerprint makes `IndexVault` treat every note as changed, and its upsert replaces them.
- Keep this inside the adapter: the port does not change.

## Verify
`uv run pytest tests/adapters/store -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
