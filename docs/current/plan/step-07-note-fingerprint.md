# Step 07 · Per-note fingerprint and migration

**Milestone:** 2 · **Needs:** 03 · **Covers:** AC-07, AC-08, AC-09

## Goal
Every note row records the fingerprint it was embedded with, and a run keeps a note only when both its content hash and its fingerprint match the current ones, so an interrupted run resumes. Existing indexes get the column filled from the index-wide fingerprint, which is then deleted.

## Files touched
- `src/ariostea/ports/store.py`: `IndexAdmin.known_hashes()` becomes `known_notes() -> dict[str, StoredNote]`; `fingerprint()` and `set_fingerprint()` are removed; `DocumentWriter.upsert_note(note, chunks, embeddings, fingerprint)`.
- `src/ariostea/domain/models.py`: `StoredNote(content_hash: str, fingerprint: str)`, frozen.
- `src/ariostea/adapters/store/sqlite_store.py`: `notes.fingerprint TEXT NOT NULL DEFAULT ''`; the column migration covers `notes`; on first sight of the column missing, fill it from `meta.fingerprint` and delete that meta row; `_drop_vectors_of_another_size` no longer touches meta; `stats().config_fingerprint` reads `""` for now (step 08 replaces it).
- `src/ariostea/indexing/index_vault.py`: `fingerprint` becomes a public property; keep a note when `stored.content_hash == hash and stored.fingerprint == self.fingerprint`; no `set_fingerprint` at the end.
- `tests/conftest.py`: a `previous_release_index` fixture.
- `tests/adapters/store/test_sqlite_store.py`, `tests/indexing/test_index_vault.py`, `tests/indexing/test_watch_vault.py`, `tests/indexing/test_index_runs.py`, `tests/mcp/test_handlers.py`, `tests/ports/test_protocols.py`: fakes and assertions move from the index-wide fingerprint to the per-note one.

## Tests to write first
- `test_upsert_stores_the_fingerprint`: `known_notes()["a.md"] == StoredNote("h1", "fp-1")`.
- `test_previous_release_index_migrates` (AC-09): `previous_release_index` builds the schema of commit `4c7d0dc` (copy `_init_schema` as it is there: notes without `fingerprint`, chunks with `context_blurb`, vec, fts, meta) with two notes and `meta.fingerprint = 'fake:v1|noop'`; after `SqliteStore` opens it, both notes report that fingerprint, `SELECT count(*) FROM meta WHERE key='fingerprint'` is 0, and opening it again changes nothing.
- `test_pre_meta_index_gets_an_empty_fingerprint`: an `old_index` (no meta table) opens and every note's fingerprint is `""`.
- `test_unchanged_index_embeds_nothing_after_migration` (AC-09): on the migrated index with an `IndexVault` whose fingerprint is `fake:v1|noop` and the same two notes, a run embeds nothing.
- `test_vector_size_change_reembeds_through_note_fingerprints`: an index built at dim 4 opened at dim 8 drops the vec table; a run under the new model's fingerprint embeds every note.
- `test_interrupted_first_index_resumes` (AC-07): 30 notes; an embedder that raises after 10 `embed_documents` calls; the first run raises; the rerun with a working embedder embeds exactly the 20 others.
- `test_interrupted_change_of_settings_resumes` (AC-08): 30 notes indexed under `fake:v1`; a run under `fake:v2` that raises after 10; the rerun under `fake:v2` embeds exactly 20 and every note then records the `fake:v2` fingerprint.
- Existing tests asserting `store.fingerprint() == ...` now assert on the notes' fingerprints.

## Implementation notes
- Migration order in `_init_schema`: drop vectors of another size; create tables; add missing `chunks` columns; if `notes` lacks `fingerprint`: `ALTER TABLE notes ADD COLUMN fingerprint TEXT NOT NULL DEFAULT ''`, then `UPDATE notes SET fingerprint = (SELECT value FROM meta WHERE key='fingerprint')` when that row exists, then `DELETE FROM meta WHERE key='fingerprint'`; one commit. Generalise `_CHUNK_COLUMNS_ADDED_LATER` to a per-table list only if it stays as small as it is (bge-m3-default.09).
- A vector-size change no longer has a fingerprint to clear: every note's own fingerprint names the old model (server-robustness.10). Update the method's docstring.
- The empty-fingerprint parts rule in `IndexVault.fingerprint` stays (vault-hardening.08 era): indexes built before chunking was configurable stored `emb|ctx`.
- The eval helpers that read `stats().config_fingerprint` (`eval/run_blurb_eval.py`) get a real value back in step 08; they are not run in this step.

## Verify
`uv run pytest -m "not integration" -q` → all pass.

## Done when
- The new tests pass; nothing in `src/` calls `fingerprint()` or `set_fingerprint()` on a store.
- `uv run ruff check . && uv run ruff format --check .` is clean.
