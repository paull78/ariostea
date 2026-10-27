# Plan: server-robustness

Status: approved · Mode: sequential · Test: `uv run pytest`
Current: step 03 (not started). Last stop: step 02 done.

## Milestone 1 · A reindex that doesn't block  ⛔ gate pending
MCP `reindex` returns at once and searches are answered while it runs; a second writer is refused (MCP) or waits (CLI, `watch`), and nothing prints "database is locked"; a SIGKILL leaves no lock; `watch` holds the lock only during a pass.
- [x] 01 RunLock port and adapters                (needs none)
- [x] 02 Store on WAL, two connections            (needs none)  ∥ 01
- [ ] 03 IndexRuns, used by CLI reindex           (needs 01, 02)
- [ ] 04 Watch passes through IndexRuns           (needs 03)
- [ ] 05 MCP reindex on a background thread       (needs 03)
- [ ] 06 Cross-process integration tests          (needs 04, 05)

## Milestone 2 · Resume and an honest status  ⛔ gate pending
An interrupted run resumes and old indexes migrate without re-embedding; every run leaves a record in the index, and `status` reports `last_indexed`, `config_fingerprint` and `last_run` truthfully from any process; the README is up to date.
- [ ] 07 Per-note fingerprint and migration       (needs 03)
- [ ] 08 Run records                              (needs 07)
- [ ] 09 Status over MCP and CLI                  (needs 06, 08)
- [ ] 10 README                                   (needs 08)  ∥ 09

## Notes
