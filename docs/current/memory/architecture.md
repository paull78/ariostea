# Architecture

Where the cycle's changes sit in the ports and adapters layout.

### vault-hardening.13 · decision · 2026-10-26 · spec
A `Vault` port (`notes()`, `read(path)`, errors `VaultUnavailable` and `NoteUnreadable`) with one `FilesystemVault` adapter that owns every file rule: dot segments, Excalidraw, `vault.ignore`, UTF-8 with the BOM dropped, the path check. `IndexVault` keeps the policy (0-notes guard, skip and record, delete sweep) and reports progress as plain values to a callback. `GetNote` becomes a use case (index lookup, then `Vault.read`). `WatchVault` filters events by the vault's rule. `indexing/scanner.py` goes away.
**Why:** the filesystem is a driven actor, so reading it from the application layer was a leak, and fix 6 would have added a second reader with its own rules. One adapter means indexing, `get_note` and `watch` cannot disagree on what a note is, and the use cases test against an in-memory vault. Progress as a plain callback value avoids a port for what is one function. Rejected: keeping `scan_vault` and adding a `read_note_file` helper next to it (less code, but the rules live in two functions and the application layer keeps touching the filesystem).

### vault-hardening.14 · decision · 2026-10-26 · spec
Testing: use cases against an in-memory `Vault` fake, the adapter on real temp files, a seeded random property test for the chunker (no new dependency), CLI through Typer's runner, integration tests for the memory bound and for unchanged wiki chunks at the defaults, and the hands-on test's synthetic-vault generator committed as an end-to-end fixture. The generator, its MCP client and queries were copied from the session scratchpad to `docs/current/assets/` so they survive the session.
**Why:** the awkward vault found every finding in this cycle, so it becomes a permanent regression test and the gates' hands-on run. Unchanged wiki chunks prove retrieval is untouched without a 15-minute eval. Rejected: throwaway hands-on scripts at each gate (nothing stops the same bugs coming back).

### vault-hardening.15 · decision · 2026-10-26 · challenge round
Change notification moves behind the `Vault` port as `changes()`, yielding batches already filtered by the note rule; the filesystem adapter wraps `watchfiles`. `WatchVault` no longer imports a vendor library, and a contract test runs the same cases on the in-memory fake and the filesystem adapter. Refines vault-hardening.13 and .06 (watch filtering).
**Why:** hexagonal lens, a vendor library inside a use case is a driven actor without a port; simplicity lens, the note rule now lives in one place instead of being handed from the adapter to the watcher. Rejected: keeping `watchfiles` in `WatchVault` with a borrowed filter (fewer changes now, two places that must agree).

### vault-hardening.17 · decision · 2026-10-26 · challenge round
A fast test parses the imports of `domain/`, `ports/`, `indexing/` and `search/` with `ast` and fails on any import of adapters, config, the MCP layer, or the vendor libraries (`watchfiles`, `fastembed`, `sqlite3`, `sqlite_vec`, `mcp`, `httpx`). Checked on master: only `indexing/watch_vault.py` importing `watchfiles` breaks it.
**Why:** hexagonal lens, a boundary nothing checks erodes; that is how the filesystem scanner and `watchfiles` ended up inside `indexing/`. Standard library only, no new dependency. Rejected: relying on review, as before.

### vault-hardening.19 · decision · 2026-10-26 · step 01
The note rule (`.md`, no dot segment, not `*.excalidraw.md`, not under an `ignore` prefix) is one function, `is_note` in `adapters/vault/rule.py`, used by `FilesystemVault` and `InMemoryVault`. The port defines only the conversation (`notes()`, `read()`, the two errors).
**Why:** the contract test runs the same cases on both adapters, so the fake must apply the rule the real one does, and one function keeps them from drifting. The rule is a file convention of Obsidian vaults, so it sits on the adapter side. Rejected: a rule function in `ports/vault.py` (puts policy in the port module); a copy in each adapter (two places to agree).

### vault-hardening.20 · discovery · 2026-10-26 · step 01
`sorted(root.rglob("*.md"))` orders paths by segment, not by string: `a/x.md` comes before `a-b/x.md`, though `-` sorts before `/` as a character. `InMemoryVault` sorts with `key=p.split("/")` to match, and a contract case pins it. Also: `issubclass()` refuses runtime-checkable protocols that have properties, so the subclassing test checks the MRO.
**Why:** the eval relies on the scanner's order (`run_blurb_eval.py:174`), and a fake with another order would pass tests the real adapter fails.
