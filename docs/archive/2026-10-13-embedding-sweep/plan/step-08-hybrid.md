# Step 08 · Hybrid runs

**Milestone:** 3 · **Needs:** 07 · **Covers:** protocol step 4, decision rule (supports AC-08)

## Goal
HYBRID is logged for the normalised control and every surviving candidate, all with unit vectors. Each candidate is judged against the normalised control, and the normalised control against the raw control.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<run id>.txt`.
- `docs/current/memory/experiment.md`: a `discovery` entry per candidate with the verdict.

## Tests to write first
Exception: running the experiment; no new code.

## Implementation notes
- Normalised control first, about an hour:
  `caffeinate -is uv run python -u eval/run_chunk_sweep.py 160t+40 --normalize --channels HYBRID --experiment "Embedding sweep" --control 2026-09-24-chunk-160t+40-dense-hybrid-sparse` (or the raw control from step 06 if it was rerun).
- If step 07 triggered the stop rule, stop after that run.
- Then per surviving candidate, about an hour each:
  `caffeinate -is uv run python -u eval/run_chunk_sweep.py 160t+40 --model <name> --normalize --channels HYBRID --experiment "Embedding sweep" --control <normalised control HYBRID id>`
  Run sequentially in the background with `nohup`, output to a log file.
- Decision rule (embedding-sweep.02): hybrid overall at least control + 0.03, no type below control − 0.05, reachable at least 160 of 167. Ties: span MRR, then cost. Applied twice (embedding-sweep.08): candidates against the normalised control, and the normalised control against the raw control.
- If a candidate passes: run `--normalize --channels DENSE,SPARSE` at `256t+64` with that model and log it separately; it is reported but not part of the decision.
- jina-v3 passing is reported with its licence caveat (embedding-sweep.01).

## Verify
`grep -c 'Embedding sweep' eval/results/runs.jsonl` rises by one for the normalised control plus one per surviving candidate (plus one for the 256t+64 pass if run).

## Done when
- The normalised control and every surviving candidate have a logged HYBRID run, with verdict entries for both comparisons.
