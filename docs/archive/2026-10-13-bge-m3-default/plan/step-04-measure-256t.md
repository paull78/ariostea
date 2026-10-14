# Step 04 · Measure bge-m3 at 256t+64 and apply the rule

**Milestone:** 2 · **Needs:** 03 · **Covers:** AC-05 (measurement)

## Goal
The HYBRID run for bge-m3 at 256t+64 is logged against the 160t+40 bge-m3 control, and the chunk-size decision is recorded under the rule in bge-m3-default.02.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<run id>.txt`.
- `docs/current/memory/defaults.md`: a `decision` entry with the result and the chosen chunk size.

## Tests to write first
Exception: running the experiment; no new code.

## Implementation notes
- Command, run in the background (about fifteen minutes):
  `caffeinate -is uv run python -u eval/run_chunk_sweep.py 256t+64 --model BAAI/bge-m3 --normalize --channels HYBRID --experiment "bge-m3 default" --control 2026-10-12-chunk-160t+40-bge-m3-norm-hybrid`
- Runs before step 05 moves `MULTILINGUAL_MODEL`, so `--model` puts the `bge-m3` slug in the id, matching the control's id style.
- Rule: adopt 256t+64 if HYBRID finds at least 155 of 167 and no type loses more than 2 cases against the control (buried 39/40, cross_lingual 41/46, exact_term 39/41, paraphrase 36/40). Count hits as `round(span_recall * n)`.

## Verify
`tail -1 eval/results/runs.jsonl` shows the new run id with a HYBRID channel; the decision entry names the chunk size.

## Done when
- The run is logged and the decision recorded.
