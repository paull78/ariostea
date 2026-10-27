# Vault

The `Vault` port and its adapters: what counts as a note, how files are read, how changes are watched, and the guards that protect the index.

### vault-hardening.02 · decision · 2026-10-26 · spec
Two guards protect the index: the vault path must exist and be a directory, and a scan that finds no notes while the index holds some stops the run. Either way the run errors with the path and the index is unchanged. To empty an index on purpose, delete the index file.
**Why:** the second guard covers the iCloud case, where the folder exists but evicted files show up as `.Name.md.icloud` and the scan finds nothing. Rejected: the path check alone (an existing empty folder still wipes the index); a ratio guard refusing scans that would delete more than some share (needs a threshold and an override flag, cleverer than the problem asks).

### vault-hardening.03 · decision · 2026-10-26 · spec
In `watch`, a guard failing at startup exits with the error, like `reindex`. A guard failing on a later pass prints a warning naming the path, leaves the index alone, and the watcher keeps going; the next change tries again.
**Why:** a brief unmount or an iCloud eviction should not kill a long-running watcher that nobody is looking at. Rejected: exiting on any guard failure (simplest code, but the watcher stays dead until restarted by hand). Open for the build: how watchfiles behaves when the watched folder itself disappears is not checked yet.

### vault-hardening.04 · decision · 2026-10-26 · spec
Notes are read as UTF-8 with a leading BOM dropped. A file that cannot be read (bad encoding, permission, vanished) is skipped with a warning naming it; the run continues, the summary counts skipped files and the MCP `reindex` result lists them. A skipped file counts as present, so an entry indexed earlier stays; a file that has truly gone is deleted by a later run. Dropping the BOM fixes the BOM-frontmatter leak, which moves from non-goal to in scope at no cost.
**Why:** Obsidian itself assumes UTF-8 and shows such files garbled, so skipping matches what the user sees in the app, with no encoding guess. Rejected: falling back to cp1252 (never fails, so UTF-16 or other encodings would be indexed as garbage text). Treating a skipped file as present was chosen without asking: deleting a note's chunks because of a momentary permission or sync error would lose work for nothing.

### vault-hardening.06 · decision · 2026-10-26 · spec
The scanner always skips paths with a segment starting with a dot and every `*.excalidraw.md`; `vault.ignore` only adds prefixes and defaults to empty. `watch` filters its events by the same rule (added without asking: same rule, and it removes the `.obsidian/workspace.json` churn the assessment found).
**Why:** Obsidian itself hides dot-folders, so one rule matches what the user sees, nested folders included, and a custom `ignore` list can no longer drop the defaults by replacing them. Rejected: a longer default `ignore` list (a user list still replaces it, nested dot-folders still indexed, Excalidraw needs glob support).

### vault-hardening.13 · decision · 2026-10-26 · spec
A `Vault` port (`notes()`, `read(path)`, errors `VaultUnavailable` and `NoteUnreadable`) with one `FilesystemVault` adapter that owns every file rule: dot segments, Excalidraw, `vault.ignore`, UTF-8 with the BOM dropped, the path check. `IndexVault` keeps the policy (0-notes guard, skip and record, delete sweep) and reports progress as plain values to a callback. `GetNote` becomes a use case (index lookup, then `Vault.read`). `WatchVault` filters events by the vault's rule. `indexing/scanner.py` goes away.
**Why:** the filesystem is a driven actor, so reading it from the application layer was a leak, and fix 6 would have added a second reader with its own rules. One adapter means indexing, `get_note` and `watch` cannot disagree on what a note is, and the use cases test against an in-memory vault. Progress as a plain callback value avoids a port for what is one function. Rejected: keeping `scan_vault` and adding a `read_note_file` helper next to it (less code, but the rules live in two functions and the application layer keeps touching the filesystem).

### vault-hardening.15 · decision · 2026-10-26 · challenge round
Change notification moves behind the `Vault` port as `changes()`, yielding batches already filtered by the note rule; the filesystem adapter wraps `watchfiles`. `WatchVault` no longer imports a vendor library, and a contract test runs the same cases on the in-memory fake and the filesystem adapter. Refines vault-hardening.13 and .06 (watch filtering).
**Why:** hexagonal lens, a vendor library inside a use case is a driven actor without a port; simplicity lens, the note rule now lives in one place instead of being handed from the adapter to the watcher. Rejected: keeping `watchfiles` in `WatchVault` with a borrowed filter (fewer changes now, two places that must agree).

### vault-hardening.18 · discovery · 2026-10-26 · self-review
Checked on this Mac: `watchfiles.watch` survives its root folder being deleted and recreated. Deleting the folder yields one batch (`added vault`, `deleted vault`), recreating it and adding `b.md` yields `added b.md`, and later changes keep arriving. So a vanished vault reaches `WatchVault` as an ordinary pass whose guard fails (vault-hardening.03), and the watcher carries on. Linux uses inotify, which drops a watch on a deleted folder; not checked, and the project runs on macOS.
**Why:** resolves the open point in vault-hardening.03 before Gate 0.

### vault-hardening.19 · decision · 2026-10-26 · step 01
The note rule (`.md`, no dot segment, not `*.excalidraw.md`, not under an `ignore` prefix) is one function, `is_note` in `adapters/vault/rule.py`, used by `FilesystemVault` and `InMemoryVault`. The port defines only the conversation (`notes()`, `read()`, the two errors).
**Why:** the contract test runs the same cases on both adapters, so the fake must apply the rule the real one does, and one function keeps them from drifting. The rule is a file convention of Obsidian vaults, so it sits on the adapter side. Rejected: a rule function in `ports/vault.py` (puts policy in the port module); a copy in each adapter (two places to agree).

### vault-hardening.20 · discovery · 2026-10-26 · step 01
`sorted(root.rglob("*.md"))` orders paths by segment, not by string: `a/x.md` comes before `a-b/x.md`, though `-` sorts before `/` as a character. `InMemoryVault` sorts with `key=p.split("/")` to match, and a contract case pins it. Also: `issubclass()` refuses runtime-checkable protocols that have properties, so the subclassing test checks the MRO.
**Why:** the eval relies on the scanner's order (`run_blurb_eval.py:174`), and a fake with another order would pass tests the real adapter fails.

### vault-hardening.24 · decision · 2026-10-26 · step 04
`Vault.changes()` yields batches of changed paths that *may hold notes*, not only note paths: it drops a path with a dot segment, an `*.excalidraw.md` drawing, or a path under an `ignore` prefix, and keeps everything else, folders, attachments and the root itself (`.`) included. Refines vault-hardening.15 ("already filtered by the note rule").
**Why:** probed on this Mac: moving a folder out of the vault or into `.trash` reports only the folder (`deleted Proj`), and moving the vault away reports only `deleted .`. A filter keeping `.md` paths would drop both, so trashed notes would stay indexed and a vanished vault would never reach the guard warning (AC-03). An attachment change now costs one hash-compare pass that changes nothing and prints nothing. `.obsidian/` churn is still dropped (AC-10). Rejected: the strict note filter (misses folder moves); guessing whether a deleted path was a folder (it no longer exists to check).
