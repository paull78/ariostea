# Step 08 · GetNote reads the file

**Milestone:** 3 · **Needs:** 02 · **Covers:** AC-12, AC-13

## Goal
`get_note` returns the note's file as written, for indexed paths only.

## Files touched
- `src/ariostea/search/get_note.py` (new): `GetNote(reader: DocumentReader, vault: Vault)`; `get(path) -> NoteText | NoteMissing` where `NoteText(path, title, text)` and `NoteMissing(path, reason | None)`. Looks the path up with `reader.note_titles([path])` (exact match); absent → `NoteMissing(path, None)`; present → `vault.read(path)`; `NoteUnreadable` → `NoteMissing(path, reason)`.
- `src/ariostea/domain/models.py`: `NoteText`, `NoteMissing`.
- `src/ariostea/config/container.py`: `Container.notes: GetNote` replaces `reader` (setup.01).
- `src/ariostea/mcp/handlers.py`: `get_note_payload` uses `container.notes`; adds `reason` when present.
- `src/ariostea/mcp/server.py`: tool docstring says it returns the note's file text.
- `src/ariostea/ports/store.py`, `src/ariostea/adapters/store/sqlite_store.py`: `read_note` stays only if something else uses it; otherwise remove it and its tests.
- Tests: `tests/search/test_get_note.py` (new), `tests/mcp/test_handlers.py`, `tests/config/test_container.py`.

## Tests to write first
- `test_returns_file_text_for_indexed_note` (AC-12): in-memory vault and fake reader; a long note with frontmatter, code and blank lines comes back equal to the file's text.
- `test_unindexed_path_is_missing_and_not_read` (AC-13): a vault file the index does not hold; the vault records no read.
- `test_traversal_paths_are_missing_and_not_read` (AC-13): `../../etc/passwd`, `/etc/passwd`.
- `test_indexed_but_deleted_file_is_missing_with_reason` (AC-13).
- `test_get_note_payload_shape`: `found`, `note_path`, `title`, `text`, `reason`.
- Integration, in `tests/test_end_to_end.py`: `get_note` on a note long enough for overlapping windows equals the file's text (AC-12 end to end).

## Implementation notes
- vault-hardening.09: the exact index lookup keeps the old traversal safety; `FilesystemVault.read` also refuses escaping paths (step 01) as a second line.
- Text mode means CRLF comes back as `\n` and a BOM is dropped (AC-12 wording).

## Verify
`uv run pytest -m "not integration" -q` → all pass; `uv run pytest tests/test_end_to_end.py -q` → pass.

## Done when
- No caller reads `container.reader`.
