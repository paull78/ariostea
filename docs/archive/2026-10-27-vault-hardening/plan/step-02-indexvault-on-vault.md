# Step 02 · IndexVault on the Vault port

**Milestone:** 1 · **Needs:** 01 · **Covers:** AC-01, AC-02, AC-04, AC-05, AC-09

## Goal
`IndexVault` reads notes only through the injected `Vault`, refuses a vault it cannot trust, skips unreadable notes and reports them. `scanner.py` is gone. `reindex_payload` reports skipped files and returns the guard error to an MCP client.

## Files touched
- `src/ariostea/indexing/index_vault.py`: constructor takes `vault: Vault`; `index()` takes no path or ignore; returns `IndexReport`.
- `src/ariostea/domain/models.py`: new `IndexReport(notes, chunks, embedded: tuple[str, ...], deleted: tuple[str, ...], skipped: tuple[tuple[str, str], ...])` (path, reason).
- `src/ariostea/indexing/scanner.py`, `tests/indexing/test_scanner.py`: deleted (rules moved to the adapter in step 01).
- `src/ariostea/config/container.py`: build `FilesystemVault(expanduser(config.vault.path), config.vault.ignore)` and inject it.
- `src/ariostea/config/schema.py`: `VaultCfg.ignore` default becomes `[]` (vault-hardening.06).
- `src/ariostea/mcp/handlers.py`: `reindex_payload` calls `index()`, returns `notes`, `chunks`, `skipped` (list of `{path, reason}`); on `VaultUnavailable` returns `{"error": "<message>"}`.
- `src/ariostea/cli.py`: `reindex` summary adds `, N skipped` and prints one warning line per skipped file on stderr. (`watch` keeps working with the old call shape adapted: `WatchVault` calls `index()` with no args; step 04 rewrites it.)
- `src/ariostea/indexing/watch_vault.py`: minimal change, call `self._indexer.index()`.
- `tests/indexing/test_index_vault.py`: rewritten on `InMemoryVault`; `tests/indexing/test_watch_vault.py`, `tests/mcp/test_handlers.py`, `tests/test_cli.py`: adapted.

## Tests to write first
- `test_missing_vault_raises_and_keeps_the_index` (AC-01): `InMemoryVault(available=False)` after a first index; `VaultUnavailable` raised, store notes and chunks unchanged.
- `test_empty_vault_with_notes_in_index_raises_and_keeps_the_index` (AC-02).
- `test_empty_vault_on_empty_index_indexes_zero` (AC-02).
- `test_unreadable_note_is_skipped_and_reported` (AC-04): one bad, two good; report.skipped names it, the good ones are indexed.
- `test_unreadable_note_keeps_its_earlier_chunks` (AC-04): indexed, then made unreadable, still in the store.
- `test_bom_frontmatter_is_not_indexed` (AC-05): chunks contain no `title:` line, title from body or file name.
- `test_excluded_notes_leave_the_index` (AC-09): a store holding `.trash/x.md` from before loses it.
- `test_report_lists_embedded_and_deleted`: an edit and a delete show up in `embedded` and `deleted`.
- `test_reindex_payload_returns_error_on_unavailable_vault`, `test_reindex_payload_lists_skipped`.
- `test_reindex_cli_prints_skipped_warnings_and_count` (CliRunner with a fake container).
Keep every existing behaviour test in `test_index_vault.py` (fingerprint, unchanged-hash skip, delete sweep) passing on the fake.

## Implementation notes
- Order: `paths = vault.notes()` (may raise), then the 0-notes guard: `if not paths and store.known_hashes(): raise VaultUnavailable(vault.location, "no notes found while the index holds N")`. Only then the loop and the delete sweep.
- `content_hash` is computed in `IndexVault` from `NoteFile.text`, as `scanner.py` did (sha256 of the UTF-8 text); unchanged notes must keep matching stored hashes, so hash the same text the old scanner hashed. A BOM note's hash changes once (it did contain the BOM); that note re-embeds once, which is expected.
- A skipped path counts as seen (vault-hardening.04).
- `IndexStats` stays for `status`; `IndexReport` is the indexing result.
- The eval calls `reindex_payload(container)`: keep that entry point and its `notes`/`chunks` keys.

## Verify
`uv run pytest -m "not integration" -q` → all pass; `uv run pytest tests/test_end_to_end.py -q` → passes (real models).

## Done when
- `grep -rn scan_vault src` finds only comments in `eval/`, or nothing.
- manual: on a copy of the synthetic vault (`docs/current/assets/make_vault.py`, store in a scratch dir), `uv run ariostea reindex --config <scratch config>` completes, names the Latin-1 file, and a missing vault path leaves the store's note count unchanged.
