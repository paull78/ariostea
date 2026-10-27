# Step 01 · Vault port and adapters

**Milestone:** 1 · **Needs:** none · **Covers:** AC-05, AC-09 (adapter level), AC-13 (adapter level)

## Goal
A `Vault` port that owns what a note is and how it is read, with two adapters: `FilesystemVault` (real) and `InMemoryVault` (test), both checked by one contract test. Nothing uses them yet.

## Files touched
- `src/ariostea/ports/vault.py`: new. `NoteFile` value (`path`, `text`, `mtime`), errors `VaultUnavailable(location, reason)` and `NoteUnreadable(path, reason)`, and the `Vault` Protocol: `location: str`, `notes() -> list[str]` (sorted vault-relative POSIX paths; raises `VaultUnavailable`), `read(path) -> NoteFile` (raises `NoteUnreadable`).
- `src/ariostea/adapters/vault/__init__.py`, `src/ariostea/adapters/vault/filesystem.py`: new. `FilesystemVault(root, ignore=())`, subclasses `Vault` (setup.02).
- `src/ariostea/adapters/vault/memory.py`: new. `InMemoryVault(files: dict[str, str | bytes], ignore=(), available=True)`, the test adapter; bytes that do not decode stand for unreadable files.
- `tests/adapters/vault/test_vault_contract.py`: new, parametrised over both adapters.
- `tests/adapters/vault/test_filesystem_vault.py`: new, real-file cases only the filesystem has.
- `tests/ports/test_protocols.py`: add both adapters to the protocol-conformance checks if that file lists adapters.

## Tests to write first
Contract (both adapters):
- `test_notes_are_sorted_relative_posix_paths`: nested notes come back sorted, relative, with `/`.
- `test_dot_segments_are_never_notes`: `.git/a.md`, `.trash/b.md`, `sub/.obsidian/c.md`, `.draft.md` are absent; `ok.md` present.
- `test_excalidraw_drawings_are_never_notes`: `Board.excalidraw.md` absent.
- `test_ignore_adds_prefixes`: with `ignore=["Templates/"]`, `Templates/t.md` absent, the dot rule still applies.
- `test_only_md_files_are_notes`: `a.png`, `b.canvas`, `c.base` absent.
- `test_read_returns_text_with_bom_dropped`: a note starting with `﻿---\n` reads back starting `---`.
- `test_read_unreadable_raises_note_unreadable`: invalid UTF-8 raises `NoteUnreadable` naming the path.
- `test_read_refuses_paths_outside_the_rule`: `../x.md`, `/etc/passwd`, `.git/a.md` raise `NoteUnreadable` and touch nothing.
Filesystem only:
- `test_missing_root_raises_vault_unavailable`, `test_root_that_is_a_file_raises_vault_unavailable`: message names the path.
- `test_permission_denied_raises_note_unreadable` (chmod 000; skip when running as root).
- `test_vanished_file_raises_note_unreadable`: file deleted between `notes()` and `read()`.
- `test_crlf_reads_as_lf`, `test_tilde_root_is_expanded`, `test_mtime_is_the_files`.

## Implementation notes
- The rule: a path is a note when it ends in `.md` (case-sensitive, as today), no segment starts with `.`, it does not end in `.excalidraw.md`, and it does not start with any `ignore` prefix (the old `_is_ignored` semantics: exact match without the trailing slash, or prefix).
- `read()` decodes with `utf-8-sig` (drops a BOM, fails on invalid UTF-8) in text mode so CRLF reads as `\n`, and checks the rule and that the resolved path stays under the root before opening (vault-hardening.09). Catch `UnicodeDecodeError`, `PermissionError`, `FileNotFoundError`, `IsADirectoryError` into `NoteUnreadable`.
- `notes()` keeps `scanner.py`'s `sorted(root.rglob("*.md"))` order: the eval relies on it (`run_blurb_eval.py:174`).
- Errors are port-owned values (hexagonal: no `OSError` crosses the port).
- `changes()` is added in step 04, not here.

## Verify
`uv run pytest tests/adapters/vault -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- Both adapters pass the same contract suite.
- Nothing outside `ports/vault.py`, `adapters/vault/` and the new tests changed.
