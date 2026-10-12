# Step 08 · Hybrid runs

**Milestone:** 3 · **Needs:** 07 · **Covers:** protocol step 4, decision rule (supports AC-08)

## Goal
HYBRID is logged for every surviving candidate and each is judged against the decision rule.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<run id>.txt`.
- `docs/current/memory/experiment.md`: a `discovery` entry per candidate with the verdict.

## Tests to write first
Exception: running the experiment; no new code.

## Implementation notes
- Skip this step if step 07 triggered the stop rule.
- Per candidate, about an hour each:
  `caffeinate -is uv run python -u eval/run_chunk_sweep.py 160t+40 --model <name> --channels HYBRID --experiment "Embedding sweep" --control <control id>`
  Run sequentially in the background with `nohup`, output to a log file.
- Decision rule (embedding-sweep.02): hybrid overall at least control + 0.03, no type below control − 0.05, reachable at least 160 of 167. Ties: span MRR, then cost.
- If a candidate passes: run `--channels DENSE,SPARSE` at `256t+64` with that model and log it separately; it is reported but not part of the decision.
- jina-v3 passing is reported with its licence caveat (embedding-sweep.01).

## Verify
`grep -c 'Embedding sweep' eval/results/runs.jsonl` rises by one per surviving candidate (plus one for the 256t+64 pass if run).

## Done when
- Every surviving candidate has a logged HYBRID run and a verdict entry.
