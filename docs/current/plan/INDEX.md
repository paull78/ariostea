# Plan: bge-m3-default

Status: approved · Mode: sequential · Test: `uv run pytest -m "not integration"` (fast), `uv run pytest` (all)
Current: wrap. Last stop: step 06 done; final gate next.

## Milestone 1 · existing indexes survive a model change  ✅ gate passed 2026-10-13
An index built at one vector size reindexes cleanly at another, and both fastembed adapters cache models under the home directory. Covers AC-02, AC-03, AC-04.
- [x] 01 store rebuilds the vector table on a size change   (needs none)
- [x] 03 fastembed cache directory for both adapters         (needs none)  ∥ 01
- [x] 02 indexing across a model change                      (needs 01)

## Milestone 2 · bge-m3 is the default  ✅ gate passed 2026-10-13
The chunk size is measured and set, the defaults and the eval's pinned model are bge-m3, and the owner's real vault reindexes cleanly on the new default. Covers AC-01, AC-05, AC-08.
- [x] 04 measure bge-m3 at 256t+64 and apply the rule         (needs 03)
- [x] 05 switch the defaults                                 (needs 02, 04)
- [x] 07 store adds columns missing from an old index        (needs 05)

## Milestone 3 · docs  ⛔ gate pending
README and the tuning write-up describe the new default; the log artifact is republished. Covers AC-06, AC-07.
- [x] 06 README, write-up and artifact                       (needs 07)

## Notes
- 2026-10-13 · plan updated: step 07 (store adds columns missing from an old index) added to milestone 2 after the owner's real-vault reindex failed at gate M2; step 06 now needs 07 · see bge-m3-default.09
