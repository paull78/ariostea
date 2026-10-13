# Step 05 · Cost fields on each run

**Milestone:** 2 · **Needs:** 03 · **Covers:** AC-07

## Goal
Each run the sweep runner logs carries a `cost` object: seconds to index the corpus and median milliseconds to embed one gold query with the run's model.

## Files touched
- `src/ariostea/eval/chunk_sweep.py`: `median_query_ms(embed_query, queries, clock)`.
- `eval/run_chunk_sweep.py`: time `index_wiki_corpus`, measure query latency, attach `run["cost"]`.
- `tests/eval/test_chunk_sweep.py`: new tests.

## Tests to write first
- `test_median_query_ms_uses_the_clock`: with a fake clock advancing 0.010 s, 0.020 s and 0.030 s around three calls (after the warm-up), the result is 20.0.
- `test_median_query_ms_skips_a_warm_up_call`: the first query is embedded once before timing and not counted.
- `test_median_query_ms_embeds_every_query`: the fake `embed_query` sees every query.

## Implementation notes
- `clock` defaults to `time.perf_counter`; tests pass a fake.
- In `_measure`, wrap `index_wiki_corpus` in `perf_counter` and return the seconds with the scores. Measure latency with a `FastEmbedEmbeddings(model)` instance over all 167 gold queries.
- Log shape: `run["cost"] = {"index_seconds": round(s, 1), "query_ms_median": round(ms, 1)}`, attached like `run["discriminated"]`.
- `render_html` must tolerate the new field. Step 06's control rerun is the first logged run with it, and its manual check opens the page.

## Verify
`uv run pytest tests/eval/test_chunk_sweep.py -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
