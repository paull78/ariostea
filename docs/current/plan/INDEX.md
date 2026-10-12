# Plan: embedding-sweep

Status: approved · Mode: sequential · Test: `uv run pytest -m "not integration"` (fast), `uv run pytest` (all)
Current: step 01 (not started). Last stop: none.

## Milestone 1 · e5 end to end  ⛔ gate pending
The runner indexes and scores the wiki corpus with e5 and logs a run tagged with the model. Covers AC-01, AC-02, AC-03, AC-05, AC-06.
- [ ] 01 embedder query/passage modes and prefixes   (needs none)
- [ ] 02 eval index takes a model                    (needs none)  ∥ 01
- [ ] 03 runner --model flag                         (needs 01, 02)

## Milestone 2 · all candidates load, costs logged, control reproduced  ⛔ gate pending
All three candidates pass the pilot, every run records its cost, and the control rerun matches the logged numbers or new controls exist. Covers AC-04, AC-07.
- [ ] 04 bge-m3 custom registration                  (needs 01)
- [ ] 05 cost fields on each run                     (needs 03)  ∥ 04
- [ ] 06 pilot and control rerun                     (needs 04, 05)

## Milestone 3 · the sweep and the verdict  ⛔ gate pending
Every run is logged, the artifact is republished, and the write-up gives a verdict against the decision rule. Covers AC-08, AC-09.
- [ ] 07 first-stage runs and the stop rule          (needs 06)
- [ ] 08 hybrid runs                                 (needs 07)
- [ ] 09 write-up, log and artifact                  (needs 08)

## Notes
