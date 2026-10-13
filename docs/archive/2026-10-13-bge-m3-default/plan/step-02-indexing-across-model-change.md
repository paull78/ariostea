# Step 02 · Indexing across a model change

**Milestone:** 1 · **Needs:** 01 · **Covers:** AC-03

## Goal
A test proves the user-visible migration: a vault indexed with one embedder reindexes with another of a different size and fingerprint, with no error and only new-size vectors left.

## Files touched
- `tests/indexing/test_index_vault.py`: one new test using the real `SqliteStore`.

## Tests to write first
- `test_reindexing_with_a_model_of_another_size_replaces_every_vector`: write two small notes under `tmp_path`; index with a real `SqliteStore(dim=2)` and a fake embedder (dim 2, fingerprint `fake:v1`); then open `SqliteStore(dim=3)` on the same file and index with a fake embedder (dim 3, fingerprint `fake:v2`). Assert no exception, every row of `chunks_vec` has `vec_length(embedding) == 3`, the chunk count matches the first run, and `fingerprint()` contains `fake:v2`.

## Implementation notes
- Build `IndexVault` the way the existing tests do (`ObsidianMarkdownParser`, `HeadingAwareChunker`, `NoopContextualizer`), passing the real store.
- If the test passes with no further code after step 01, record that it was born green and prove it can fail by reverting step 01's rebuild locally.

## Verify
`uv run pytest tests/indexing/test_index_vault.py -q` → all pass.

## Done when
- The test passes, and the fast suite passes.
