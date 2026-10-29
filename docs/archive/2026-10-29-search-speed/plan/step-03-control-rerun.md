# Step 03 · Control rerun

**Milestone:** 1 · **Needs:** 02 · **Covers:** AC-02, AC-04

## Goal
Today's default (jina-reranker-v2, pool 100) measured by the new runner and logged, with its latency; the gate learns whether 157 of 167 reproduces.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/<id>.txt`: the logged run.
- cycle memory: a discovery entry with the numbers.

## Tests to write first
Exception: a measurement, not code; the runner's logic is tested in step 02.

## Implementation notes
- `uv run python eval/run_rerank_sweep.py jinaai/jina-reranker-v2-base-multilingual:100`, run in the background with output saved under `eval/results/logs/` (setup.03: raw output in the repo at once).
- Expect HYBRID 157 of 167 (span recall 0.940), per type buried 40, cross_lingual 41, exact_term 39, paraphrase 37. A different count means the runner differs from the chunk sweep's path: stop, write an issue entry, do not go on to step 05.
- Republish `eval/results/experiment_log.html` to https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid (`url`); needs `/login` to the owner's other organisation (setup.03). If refused, record it and tell the owner.

## Verify
`python3 -c "import json; r=[json.loads(l) for l in open('eval/results/runs.jsonl')][-1]; print(r['id'], round(r['channels']['HYBRID']['overall']['span_recall']*167), r['cost'], r['sources_short'])"` → the run id, 157, latency fields, a count.

## Done when
- The run is logged with 157 of 167 and its latency.
- The artifact is republished, or the refusal is recorded.
