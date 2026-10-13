# Step 07 · First-stage runs and the stop rule

**Milestone:** 3 · **Needs:** 06 · **Covers:** protocol step 3 (supports AC-08)

## Goal
DENSE, SPARSE and FUSED are logged for every candidate that survived the pilot, all with unit vectors, and the stop rule is applied against the normalised control.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<run id>.txt`.
- `docs/current/memory/experiment.md`: one `discovery` entry with the first-stage result and the stop decision.

## Tests to write first
Exception: running the experiment; no new code.

## Implementation notes
- Per candidate:
  `caffeinate -is uv run python -u eval/run_chunk_sweep.py 160t+40 --model <name> --normalize --channels DENSE,SPARSE,FUSED --experiment "Embedding sweep" --control <normalised control id from step 06>`
- Run them one after another, in the background with output to a log under `eval/results/logs/`, so the session can wait on completion.
- SPARSE should equal the control's 0.533, since normalisation does not touch it. A difference means the chunks changed; note it in the discovery entry.
- Stop rule (embedding-sweep.03, embedding-sweep.08): if no candidate's DENSE overall exceeds the normalised control's, record it; step 08 then runs HYBRID for the normalised control only.

## Verify
`grep -c 'Embedding sweep' eval/results/runs.jsonl` equals the number of surviving candidates plus the control runs.

## Done when
- Every surviving candidate has a logged first-stage run.
- The stop decision is recorded in cycle memory.
