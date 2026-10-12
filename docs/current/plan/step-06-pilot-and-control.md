# Step 06 · Pilot and control rerun

**Milestone:** 2 · **Needs:** 04, 05 · **Covers:** protocol steps 1 and 2 (supports AC-04, AC-07, AC-08)

## Goal
Each candidate is shown to load and behave, its embedding speed is known, and the control is confirmed to reproduce at the current commit, or replaced.

## Files touched
- `eval/pilot_embeddings.py`: new script. For each model: load, print dimension, check that e5's query and passage vectors for the same text differ, embed 100 chunks of the wiki corpus at 160t+40 and print seconds.
- `eval/results/logs/<date>-embedding-pilot.txt`: pilot output.
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<run id>.txt`: control rerun.

## Tests to write first
Exception: an experiment script whose output is the measurement itself; the logic it relies on is tested in steps 01, 04 and 05.

## Implementation notes
- Pilot: `uv run python -u eval/pilot_embeddings.py | tee eval/results/logs/$(date +%F)-embedding-pilot.txt`. First run downloads about 7 GB into `~/.cache/fastembed`.
- A candidate that fails to load is dropped: write an `issue` entry in cycle memory with the error.
- Control rerun: `uv run python -u eval/run_chunk_sweep.py 160t+40 --channels DENSE,SPARSE,FUSED --experiment "Embedding sweep" --control 2026-09-24-chunk-160t+40-dense-hybrid-sparse`. It must match DENSE 0.545, SPARSE 0.533, FUSED 0.629 exactly. If any differs, write a `discovery` entry and rerun with `--channels HYBRID` (about an hour); the new runs become the controls for steps 07 and 08.
- After the runs, run `uv run pytest tests/adapters/embedding -q` so the bge-m3 integration test runs against the cached model.
- `ls` is aliased to eza here; use `/bin/ls` in scripts.

## Verify
`grep -c '"id"' eval/results/runs.jsonl` rises by one (or two with HYBRID); the pilot log lists all three candidates with a dimension and a timing, or an error.

## Done when
- The pilot log is committed and names every candidate's outcome.
- The control rerun is logged, with a memory entry if it did not reproduce.
- manual: open `eval/results/experiment_log.html` and check the new run renders with its cost.
