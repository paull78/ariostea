# Step 05 · Full sweep

**Milestone:** 2 · **Needs:** 03, 04 · **Covers:** AC-01, AC-02, AC-04, AC-06

## Goal
The 7 remaining configurations measured and logged, and the pick rule applied, so the gate can show the owner the table and the proposal.

## Files touched
- `eval/results/runs.jsonl`, `eval/results/experiment_log.html`, `eval/results/logs/`: 7 runs.
- cycle memory: a discovery with the table, a decision entry with the proposed configuration (or none).

## Tests to write first
Exception: a measurement; the code is tested in steps 01, 02 and 04.

## Implementation notes
- One invocation, index built once: `uv run python eval/run_rerank_sweep.py jinaai/jina-reranker-v2-base-multilingual:50 jinaai/jina-reranker-v2-base-multilingual:30 jinaai/jina-reranker-v2-base-multilingual:20 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1:100 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1:50 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1:30 cross-encoder/mmarco-mMiniLMv2-L12-H384-v1:20`, in the background, output kept in `eval/results/logs/`. About 30 to 40 minutes.
- Table per run: hits of 167, per-type hits against the control (buried 40, cross_lingual 41, exact_term 39, paraphrase 37), span MRR, p95 and median ms, sources short count.
- Pass = at least 155, no type down more than 2 (search-speed.02), p95 under 1,000 ms (search-speed.07). Pick = most hits among passing, then span MRR, then lower p95 (search-speed.08). None passing: defaults stay (search-speed.09).
- Republish the log artifact (setup.03).

## Verify
`python3 -c "import json; [print(r['id'], round(r['channels']['HYBRID']['overall']['span_recall']*167), r['cost']['search_ms_p95'], r['sources_short']) for r in map(json.loads, open('eval/results/runs.jsonl')) if '-rerank-' in r['id']]"` → 8 rows.

## Done when
- 8 rerank runs in the log, artifact republished (or refusal recorded).
- The pick is written down with its reason.
