# Step 04 · Vault.changes and WatchVault

**Milestone:** 2 · **Needs:** 03 · **Covers:** AC-03, AC-06, AC-10, AC-17, AC-21

## Goal
Change notification sits behind the `Vault` port. `WatchVault` no longer imports `watchfiles`, keeps running through guard failures and bad files, and reports each pass as a plain value the CLI prints. A test enforces the dependency rule.

## Files touched
- `src/ariostea/ports/vault.py`: add `changes(stop_event=None) -> Iterator[frozenset[str]]`: batches of vault-relative note paths that changed (added, modified or deleted), already filtered by the note rule; empty batches are never yielded.
- `src/ariostea/adapters/vault/filesystem.py`: implement with `watchfiles.watch(root, stop_event=...)`, mapping absolute paths to relative ones and keeping only paths that pass the rule.
- `src/ariostea/adapters/vault/memory.py`: `InMemoryVault` takes scripted batches (each batch can also mutate files or availability before it is yielded).
- `src/ariostea/indexing/watch_vault.py`: `WatchVault(indexer, vault)`; `run(stop_event=None, on_report=..., on_warning=...)`. The first pass lets `VaultUnavailable` propagate (the CLI exits, step 03). Later passes catch `VaultUnavailable` and call `on_warning(message)`. Each pass's `IndexReport` goes to `on_report`; skipped files are warned once per path until they become readable again.
- `src/ariostea/cli.py`: `watch` prints `+N updated, -M deleted` for a report with any embedded or deleted note, nothing otherwise, and warnings on stderr.
- `tests/test_architecture.py`: new dependency-rule test.
- `tests/indexing/test_watch_vault.py`, `tests/adapters/vault/test_vault_contract.py`, `tests/adapters/vault/test_filesystem_vault.py`, `tests/test_cli.py`: updated.

## Tests to write first
- `test_dependency_rule` (AC-21): parse every module under `src/ariostea/{domain,ports,indexing,search}` with `ast`; fail on imports of `ariostea.adapters`, `ariostea.config`, `ariostea.mcp`, `watchfiles`, `fastembed`, `sqlite3`, `sqlite_vec`, `mcp`, `httpx`. Show it fails once by adding `import watchfiles` to `indexing/watch_vault.py` locally, then remove it (record in the commit message).
- Contract: `test_changes_skip_paths_outside_the_rule` (AC-10, in-memory scripted; filesystem: integration-free with a real temp dir, writing `.obsidian/workspace.json` then `a.md`, the first batch contains only `a.md`).
- `test_guard_failure_after_start_warns_and_continues` (AC-03): second batch makes the vault unavailable, third restores it; `on_warning` called once, store unchanged after the second pass, third pass indexes.
- `test_guard_failure_at_start_propagates` (AC-03).
- `test_new_unreadable_note_warns_and_watch_continues` (AC-06), `test_unreadable_note_is_warned_once`.
- `test_report_per_pass` (AC-17): edit one, delete one → one report with 1 embedded and 1 deleted; a batch that changes nothing in the store → CLI prints nothing.
- `test_cli_watch_prints_pass_summary` (CliRunner, fake container).

## Implementation notes
- vault-hardening.15: no filter passes between objects; the adapter applies its own rule.
- vault-hardening.18: on macOS `watchfiles` survives the root being deleted and recreated, so a vanished vault arrives as an ordinary pass whose guard fails.
- Each pass still runs a full `index()` (hash compare over every note). Fine for this cycle; an mtime short-cut is out of scope.
- The `Container` exposes the watcher's inputs: either a `watcher: WatchVault` built in `build_container`, or the vault as a port-typed field. Prefer building `WatchVault` in the container (a use case, setup.01) so the CLI never sees the adapter.

## Verify
`uv run pytest -m "not integration" -q` → all pass; `grep -n watchfiles src/ariostea/indexing/*.py` → no output.

## Done when
- manual: `uv run ariostea watch --config <scratch config>` on the synthetic vault: touching `.obsidian/workspace.json` prints nothing; editing a note prints `+1 updated, -0 deleted`-style line; moving the vault folder away prints a warning and the watcher stays up; moving it back and editing a note indexes again.
