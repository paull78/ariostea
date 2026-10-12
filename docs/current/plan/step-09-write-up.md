# Step 09 · Write-up, log and artifact

**Milestone:** 3 · **Needs:** 08 · **Covers:** AC-08, AC-09

## Goal
`docs/retrieval-tuning.md` has an "Embedding model" section with the results, cost figures and verdict; the experiment log page is re-rendered and the log artifact republished.

## Files touched
- `docs/retrieval-tuning.md`: new section "## Embedding model" plus reproduction commands.
- `docs/design/2026-10-12-embedding-sweep.md`: status line updated to "implemented and measured <date>".
- `eval/results/experiment_log.html`: re-rendered.

## Tests to write first
Exception: documentation and a publish step.

## Implementation notes
- Section content: the normalisation finding (embedding-sweep.07) and its measured effect, raw control against normalised control; per-type span recall at k=5 for DENSE, SPARSE, FUSED and HYBRID per model against the normalised control; a cost table (download size, index seconds, median query ms); the verdict against the rule; a finding/evidence/reason table like the per-chunk section; the jina licence caveat if relevant; the 256t+64 pass if it ran.
- Write in the neutral register of the rest of the doc: no em dashes, digits for metrics.
- Republish the artifact: `Artifact` publish of `eval/results/experiment_log.html` with `url` https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid (memory: experiment log).
- Changing the default model or turning normalisation on by default is out of scope. If either passes, the write-up recommends it and the owner decides at the final gate.

## Verify
`grep -n "## Embedding model" docs/retrieval-tuning.md` → one match; the artifact publish returns the same URL with a new version.

## Done when
- The section exists with tables for every logged model, a verdict on each candidate and a verdict on normalisation.
- The artifact is republished.
- manual: owner reads the verdict.
