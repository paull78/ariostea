# Step 01 · Latency and sources helpers

**Milestone:** 1 · **Needs:** none · **Covers:** AC-02, AC-06

## Goal
Pure helpers the sweep runner uses: per-query search timing summarised as median and 95th percentile, and the count of queries for which `search_sources` returns fewer notes than asked.

## Files touched
- `src/ariostea/eval/rerank_sweep.py` (new): `search_latency_ms(search, queries, clock=time.perf_counter) -> dict` returning `{"search_ms_median": float, "search_ms_p95": float}` after one untimed warm-up call on `queries[0]`; `short_sources(sources, queries, k=10) -> int`, where `sources(query, k)` returns a list of notes.
- `tests/eval/test_rerank_sweep.py` (new).

## Tests to write first
- `test_latency_warms_up_once_then_times_every_query`: a fake search records calls; with 4 queries it is called 5 times, the first untimed.
- `test_latency_median_and_p95_from_injected_clock`: a fake clock giving durations 10, 20, …, 200 ms for 20 queries yields median 105.0 and p95 by the nearest-rank method (ceil(0.95·n)-th smallest) 190.0.
- `test_latency_rounds_to_one_decimal`: values in the dict have at most one decimal, as `cost` values in the log do.
- `test_short_sources_counts_queries_below_k`: a fake returning 10, 3, 10, 0 notes gives 2; it is called with `k=10` each time.
- `test_latency_rejects_no_queries`: an empty list raises `ValueError`.

## Implementation notes
Follow `median_query_ms` in `src/ariostea/eval/chunk_sweep.py` (warm-up rationale, injected clock). Nearest-rank p95 keeps the value an observed timing, no interpolation. Keep the module free of container and adapter imports so it stays a pure helper; the runner (step 02) wires it.

## Verify
`uv run pytest tests/eval/test_rerank_sweep.py -q` → all pass.

## Done when
- The five tests pass and ruff is clean on the new files.
