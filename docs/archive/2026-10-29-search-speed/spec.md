# Spec: search-speed

Cycle: search-speed · Type: feature · Branch: feature/search-speed · Status: approved

## Goal
A search answers in under 1 s, measured as the 95th percentile over the 167 wiki eval queries on the wiki index at the defaults, on this M4 Pro (today about 4.7 s when the pool fills with full chunks, search-speed.03; 2.3 to 3.0 s on the 50-note scratch vault, vault-hardening.29), with no measurable loss in retrieval quality. Server memory is measured and reported before and after; a saving is welcome but not a target.

## Scope and non-goals
In scope, both config only and neither forcing a reindex:
- the rerank pool size (`rerank.pool`, today 100);
- the reranker model: jina-reranker-v2 as today, or `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (its fp32 `onnx/model.onnx`), registered as a custom fastembed model.

A changed default is adopted only with the owner's yes (setup.04).

Non-goals: a memory target (memory is measured and reported only); lazy model loading; cutting the text the reranker reads; quantised or GPU (CoreML) runs of jina, which buy nothing here (search-speed.05); turning the reranker off (fused order finds 24 fewer cases, search-speed.03); any change to the embedder or chunking, so no reindex; the open issue server-robustness.14.

## Behaviour
The cycle runs a sweep of HYBRID wiki evals: pool 100, 50, 30 and 20 for each of the two rerankers, 8 runs. jina at pool 100 is today's default, rerun so its latency is measured the same way. Each run is logged and the log artifact republished (setup.03). A run whose container logs a warning (the reranker falling back to fused order) aborts before anything is logged, through the existing `fail_on_warnings` guard.

Among the configurations that pass both bars (AC-01 quality, AC-02 latency), the one with the most hits is proposed as the new default; ties go to span MRR, then lower 95th percentile. The owner says yes or no before any default changes (setup.04). If no configuration passes both bars, the defaults stay and the recap shows the recall and latency table, so the owner can pick a trade-off as a config change or a later cycle.

`search_sources` follows the pool: it groups at most `rerank.pool` reranked chunks by note, so it gets the same speed-up and can return fewer notes than asked when the pool is small.

Running mMiniLM-L12 goes through the production container, so the reranker adapter learns to register it as a custom fastembed model and to fetch its ONNX, config and tokenizer files (search-speed.05). It stays available as a `rerank.model` value whether or not it becomes the default, and the README lists its measured trade-off.

## Acceptance criteria
- AC-01 · The configuration adopted as the new default finds at least 155 of 167 cases on the wiki HYBRID eval (span recall at k=5), with no query type down more than 2 cases against `2026-10-13-chunk-256t+64-bge-m3-norm-hybrid` (157 of 167); the run is logged in `eval/results/runs.jsonl`.
- AC-02 · The same HYBRID run times each of the 167 queries through the production search path after one warm-up query, and its log record holds the median and 95th percentile search time in ms. For the adopted configuration the 95th percentile is under 1,000 ms.
- AC-03 · With `rerank.model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"` and an empty fastembed cache directory, the container builds the reranker without falling back to fused order: it fetches the model's ONNX file, config and tokenizer, and ranks a passage that answers the query above two that don't.
- AC-04 · Every sweep run (8) is in `eval/results/runs.jsonl` with its pool and reranker model in `config`, and the log artifact is republished. A sweep run whose reranker falls back to fused order aborts and logs nothing.
- AC-05 · If the owner approves a configuration, `RerankCfg` defaults match it and the README's search time, memory and reranker sections give the newly measured figures, with memory and scratch-vault search time measured as in vault-hardening.29 (the stdio server on the 50-note scratch vault, after seven searches); if not, the defaults are unchanged and the README gains only the mMiniLM option with its measured trade-off.
- AC-06 · Each sweep run also records for how many of the 167 queries `search_sources` (k=10) returns fewer than 10 notes; the README says `search_sources` sees at most `rerank.pool` chunks, so its note count is bounded by the pool.

## Testing approach
- Fast unit tests: the median and 95th percentile helper with an injected clock; the run record's new latency fields; the config accepting the reranker model and pool; a sweep run whose container warns aborts without logging.
- One integration test (marked `integration`) for AC-03: the real adapter loads mMiniLM-L12 into a temporary cache directory and ranks a relevant passage first.
- The 8 logged HYBRID eval runs are the evidence for AC-01, AC-02 and AC-04. No timing assertion in the test suite.

Test command: `uv run pytest && uv run ruff check . && uv run ruff format --check .`

## Open questions
- none
