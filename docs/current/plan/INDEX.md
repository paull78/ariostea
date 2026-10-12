# Plan: embedding-sweep

Status: approved · Mode: sequential · Test: `uv run pytest -m "not integration"` (fast), `uv run pytest` (all)
Current: step 08 (not started). Last stop: step 07 done; stop rule not triggered.

## Milestone 1 · e5 end to end  ✅ gate passed 2026-10-12
The runner indexes and scores the wiki corpus with e5 and logs a run tagged with the model. Covers AC-01, AC-02, AC-03, AC-05, AC-06.
- [x] 01 embedder query/passage modes and prefixes   (needs none)
- [x] 02 eval index takes a model                    (needs none)  ∥ 01
- [x] 03 runner --model flag                         (needs 01, 02)

## Milestone 2 · all candidates load, costs logged, control reproduced  ✅ gate passed 2026-10-12
All three candidates pass the pilot, every run records its cost, the normalise setting exists, the raw control reproduces or is replaced, and the normalised control's first stage is logged. Covers AC-04, AC-07, AC-10, AC-11, AC-12.
- [x] 04 bge-m3 custom registration                  (needs 01)
- [x] 05 cost fields on each run                     (needs 03)  ∥ 04
- [x] 10 normalise setting                           (needs 04)
- [x] 11 runner --normalize flag                     (needs 05, 10)
- [x] 06 pilot, control rerun, normalised control    (needs 11)

## Milestone 3 · the sweep and the verdict  ⛔ gate pending
Every run is logged, the artifact is republished, and the write-up gives a verdict against the decision rule. Covers AC-08, AC-09.
- [x] 07 first-stage runs and the stop rule          (needs 06)
- [ ] 08 hybrid runs                                 (needs 07)
- [ ] 09 write-up, log and artifact                  (needs 08)

## Notes
- 2026-10-12 · plan updated: steps 10 (normalise setting) and 11 (runner --normalize) added to milestone 2; steps 06 to 09 now measure the normalised control and candidates with unit vectors · see embedding-sweep.08
