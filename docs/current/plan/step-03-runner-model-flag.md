# Step 03 · Runner --model flag

**Milestone:** 1 · **Needs:** 01, 02 · **Covers:** AC-06

## Goal
`eval/run_chunk_sweep.py --model <name>` indexes, counts reachable spans and scores with that model, and logs runs whose id carries the model slug. Without the flag, run ids and logged config are exactly as today.

## Files touched
- `src/ariostea/eval/chunk_sweep.py`: `model_slug(name)`, `sweep_run_id(..., model=None)`, `run_config(k, cfg, model)`.
- `eval/run_chunk_sweep.py`: `--model` (default `MULTILINGUAL_MODEL`), passed to `index_wiki_corpus`, to the token counter and to `run_config`.
- `tests/eval/test_chunk_sweep.py`: new tests.

## Tests to write first
- `test_model_slug_is_the_lowercased_last_path_part`: `model_slug("intfloat/multilingual-e5-large") == "multilingual-e5-large"`, `model_slug("BAAI/bge-m3") == "bge-m3"`.
- `test_run_id_without_a_model_is_unchanged`: `sweep_run_id("2026-10-12", "160t+40", ["DENSE"])` equals today's `2026-10-12-chunk-160t+40-dense`, and so does passing `model=None`.
- `test_run_id_with_a_model_carries_its_slug`: with `model="BAAI/bge-m3"` the id is `2026-10-12-chunk-160t+40-bge-m3-dense`.
- `test_run_config_records_the_model`: `run_config(5, ChunkingCfg(), "m") == {"k": 5, "embedding": "m", "chunking": ChunkingCfg().model_dump()}`.

## Implementation notes
- The runner passes `model=None` to `sweep_run_id` when `--model` equals `MULTILINGUAL_MODEL`, so control reruns keep their id shape.
- The token counter for reachable spans must be built with the run's model, not the constant: chunk boundaries follow the model's tokenizer.
- The duplicate-run check and `--no-log` keep working as before.
- Smoke check, not a test: `uv run python eval/run_chunk_sweep.py 160t+40 --model intfloat/multilingual-e5-large --channels DENSE --no-log` indexes and prints a DENSE table. First use downloads about 2.24 GB.

## Verify
`uv run pytest tests/eval/test_chunk_sweep.py -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
- manual: the e5 smoke run above prints a DENSE table with 167 cases.
