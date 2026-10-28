# Plan: search-speed

Status: approved · Mode: sequential · Test: `uv run pytest && uv run ruff check . && uv run ruff format --check .`
Current: step 03 (in progress). Last stop: starting.

## Milestone 1 · Sweep skeleton on today's reranker  ⛔ gate pending
The sweep runner indexes the wiki once and logs a jina run with recall, search latency (median, p95) and the `search_sources` short count; the control rerun at pool 100 shows whether 157 of 167 reproduces.
- [x] 01 latency and sources helpers      (needs none)
- [x] 02 rerank sweep runner              (needs 01)
- [~] 03 control rerun                    (needs 02)

## Milestone 2 · Full sweep and the pick  ⛔ gate pending
mMiniLM-L12 loads from an empty cache; all 8 runs are logged; the gate shows the table and the configuration the pick rule proposes, for the owner's yes or no.
- [ ] 04 mMiniLM in the reranker adapter  (needs none)
- [ ] 05 full sweep                       (needs 03, 04)

## Milestone 3 · Apply the decision  ⛔ gate pending
Defaults and README match the owner's decision, with memory and scratch-vault search time re-measured.
- [ ] 06 defaults and README              (needs 05)

## Notes
- AC mapping: AC-01 → 05; AC-02 → 01, 02, 03, 05; AC-03 → 04; AC-04 → 02, 03, 05; AC-05 → 06; AC-06 → 01, 02, 05, 06.
