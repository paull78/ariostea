# Step 02 · Rerank sweep runner

**Milestone:** 1 · **Needs:** 01 · **Covers:** AC-02, AC-04, AC-06

## Goal
`eval/run_rerank_sweep.py` measures (reranker model, pool) configurations on the wiki gold set over one index, and logs each as a run with HYBRID scores, latency and the `search_sources` short count.

## Files touched
- `src/ariostea/eval/rerank_sweep.py`: add `rerank_run_id(date, model, pool) -> str` (`<date>-rerank-<model slug>-pool<pool>`), `rerank_config(config, model, pool) -> Config` (the container's config with `rerank.model` and `rerank.pool` replaced, nothing else), and `rerank_run_config(k, model, pool) -> dict` (the logged `config`: `k`, `embedding`, `chunking` as today's runs plus `rerank: {"model", "pool"}`).
- `eval/run_rerank_sweep.py` (new): CLI `uv run python eval/run_rerank_sweep.py MODEL:POOL ... [--control ID] [--no-log] [--experiment NAME]`; default control `2026-10-13-chunk-256t+64-bge-m3-norm-hybrid`.
- `tests/eval/test_rerank_sweep.py`: tests below.

## Tests to write first
- `test_run_id_names_model_and_pool`: `rerank_run_id("2026-10-28", "jinaai/jina-reranker-v2-base-multilingual", 50)` is `2026-10-28-rerank-jina-reranker-v2-base-multilingual-pool50`.
- `test_rerank_config_changes_only_the_reranker`: model and pool replaced; `embedding`, `chunking`, `store`, `vault` equal to the input; `rerank.enabled` stays true.
- `test_run_config_records_rerank_model_and_pool`: the dict has `rerank == {"model": ..., "pool": ...}` and the default `embedding` and `chunking`.
- `test_parse_grid_rejects_bad_entries`: `"jina:abc"`, `"jina"` and `":50"` raise `ValueError`; `"jinaai/x:20"` gives `("jinaai/x", 20)` (split on the last colon).
- `test_guarded_build_aborts_on_container_warning`: building and querying inside `fail_on_warnings("ariostea.config.container")` with a container logger warning raises `WarnedError` (uses the existing guard; asserts the runner's wrapper, not the guard itself, by passing a fake builder that logs).

## Implementation notes
- Index once with `index_wiki_corpus(WIKI, db)` at the defaults into a temp db. For each configuration build a second container with `build_container(rerank_config(base.config, model, pool))` over the same db, as `fused_channel` does (`src/ariostea/eval/wiki_index.py`). No re-indexing per run.
- Each configuration's container build, scoring and timing run inside `fail_on_warnings(CONTAINER_LOGGER)` (`src/ariostea/eval/blurb_eval.py`, logger `ariostea.config.container`, search-speed.14); a warning aborts the whole sweep before that run is logged.
- Scores: `evaluate_spans` over `make_hybrid_chunk_fn(container, CHUNK_POOL)` for the gold cases and the discriminated cases (`load_discriminated`), as `_measure` in `eval/run_chunk_sweep.py`; `reachable` from `reachable_spans` with the default chunker.
- Latency: `search_latency_ms` on the hybrid search through `search_payload(container, query, k=CHUNK_POOL)` over the 167 queries. Short count: `short_sources` through `search_sources_payload(container, query, k=10)`.
- Record: `make_run(...)` with `channels={"HYBRID": ...}`, `run["discriminated"]`, `run["cost"] = {"search_ms_median", "search_ms_p95", "index_seconds"}`, `run["sources_short"] = <int>`. Refuse duplicates and an unlogged control before indexing, as the chunk sweep does. Save printed tables to `eval/results/logs/<id>.txt`; re-render `experiment_log.html` at the end.
- Label: `"<model slug>, pool <pool>"`; experiment default `"Search speed"`.

## Verify
`uv run pytest tests/eval/test_rerank_sweep.py -q && uv run python eval/run_rerank_sweep.py --help` → tests pass, help prints.

## Done when
- Tests pass, ruff clean.
- `--no-log` with `jinaai/jina-reranker-v2-base-multilingual:20` runs end to end and prints HYBRID scores, latency and short count (about 10 minutes, mostly indexing).
