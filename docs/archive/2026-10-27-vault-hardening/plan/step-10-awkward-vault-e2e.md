# Step 10 · Awkward-vault end-to-end test

**Milestone:** 4 · **Needs:** 04, 05, 06, 07, 08 · **Covers:** AC-20, AC-16 (stdio clause), AC-07 and AC-08 end to end

## Goal
The 54-note synthetic vault from the usability assessment is a permanent integration test that drives `reindex`, `watch` and every MCP tool.

## Files touched
- `tests/fixtures/awkward_vault.py` (new): `build(root: Path) -> Path`, ported from `docs/current/assets/make_vault.py` (no module-level side effects, writes under the given root, symlink target inside the root's parent). Adds one note the assets lack (vault-hardening.22): `Clips/Web clip with image.md`, an ordinary note with a heading, a paragraph holding a unique marker word, an inline `![](data:image/png;base64,...)` image whose base64 run is about 20,000 characters with no whitespace, and a closing paragraph.
- `tests/fixtures/awkward_queries.py` (new): the known-answer queries from `docs/current/assets/q.py` with the expected note path.
- `tests/test_awkward_vault.py` (new, `integration`).

## Tests to write first
- `test_reindex_completes_and_names_bad_files`: CLI `reindex` on the built vault with a temp store exits 0 and stderr names the Latin-1 file.
- `test_excluded_files_are_not_searchable`: `.git/notes.md`, `.trash/...`, Excalidraw content and the `.obsidian` plugin README are absent from `search_knowledge` and `search_sources` results for their unique marker strings.
- `test_known_answer_queries_rank_first`: every query from `awkward_queries.py` ranks its note first through the MCP `search_knowledge` tool over stdio (`mcp` client, `uv run ariostea serve --config <temp>`).
- `test_stdio_stdout_is_protocol_only_during_reindex` (AC-16): calling the MCP `reindex` tool produces no non-protocol bytes on the server's stdout (the client session stays valid and the call returns).
- `test_long_base64_note_indexes_within_budget`: after `reindex`, every chunk of `Clips/Web clip with image.md` in the store has at most `chunking.max_tokens` model tokens, and its marker word ranks the note first.
- `test_get_note_returns_files`: non-ASCII, emoji and CJK paths return the file text; `../../etc/passwd` returns `found: false`.
- `test_watch_picks_up_an_edit`: `ariostea watch` in a subprocess; after an edit the new text is searchable within 10 s; stop by PID.

## Implementation notes
- Build the vault and index once per module (fixture with `scope="module"`); the first index of 54 notes takes about 30 s on bge-m3.
- The assets in `docs/current/assets/` are archived with the cycle; the tests must not import from `docs/`.
- Stop subprocesses by PID, never `pkill -f` (general lesson).

## Verify
`uv run pytest tests/test_awkward_vault.py -q` → all pass (integration, a few minutes).

## Done when
- The test passes on the cycle branch, and fails when step 01's dot rule is disabled (checked once by hand, noted in the commit message).
