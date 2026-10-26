# Plan: vault-hardening

Status: approved · Mode: sequential · Test: `uv run pytest`
Current: wrap. Last stop: step 10 done.

## Milestone 1 · The index cannot be lost  ✅ gate passed 2026-10-26
`reindex` (CLI and MCP) refuses a missing or empty vault, skips unreadable files with a named warning, leaves out dot-folders and Excalidraw, and prints errors as one line. The vault sits behind a port.
- [x] 01 Vault port and adapters          (needs none)
- [x] 02 IndexVault on the Vault port     (needs 01)
- [x] 03 One-line CLI errors, openai_compat removed  (needs 02)

## Milestone 2 · Watch and progress  ✅ gate passed 2026-10-26
`watch` ignores `.obsidian` churn, survives a vanished vault and bad files and reports each pass in one line; `reindex` shows progress; a test enforces the dependency rule.
- [x] 04 Vault.changes and WatchVault      (needs 03)
- [x] 05 Progress callback and printer     (needs 04)

## Milestone 3 · Correct content  ✅ gate passed 2026-10-26
No chunk exceeds the budget, code comments are not headings, `get_note` returns the file.
- [x] 06 Chunker bounds oversized words    (needs none)
- [x] 07 Code fences in chunker and parser (needs 06)
- [x] 08 GetNote reads the file            (needs 02)  ∥ 06, 07

## Milestone 4 · Ready for a new user  ⛔ gate pending
The README is honest about first-run setup, ARCHITECTURE.md shows the Vault port, and the awkward vault is a permanent end-to-end test.
- [x] 09 README, example config, ARCHITECTURE.md  (needs 03, 04, 05, 06, 07, 08)
- [x] 10 Awkward-vault end-to-end test     (needs 04, 05, 06, 07, 08)  ∥ 09

## Notes
- 2026-10-26 · plan updated: step 10 fixture adds a note with a 20,000-character base64 image, plus a test that it indexes within budget · see vault-hardening.22
