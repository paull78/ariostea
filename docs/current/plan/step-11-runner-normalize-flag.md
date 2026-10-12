# Step 11 · Runner --normalize flag

**Milestone:** 2 · **Needs:** 05, 10 · **Covers:** AC-12

## Goal
`eval/run_chunk_sweep.py --normalize` indexes and queries with unit vectors, logs `normalize: true` in the run config, and adds `-norm` to the run id. Without the flag, ids and config are as in AC-06.

## Files touched
- `src/ariostea/eval/chunk_sweep.py`: `sweep_run_id(..., normalize=False)`, `run_config(k, cfg, model, normalize=False)`.
- `eval/run_chunk_sweep.py`: `--normalize`, passed to `index_wiki_corpus`, to the latency embedder and to `run_config`; the label gains ", normalised".
- `tests/eval/test_chunk_sweep.py`: new tests.

## Tests to write first
- `test_run_id_with_normalize_ends_the_spec_part_with_norm`: `sweep_run_id("2026-10-12", "160t+40", ["DENSE"], normalize=True) == "2026-10-12-chunk-160t+40-norm-dense"`; with `model="BAAI/bge-m3"` too it is `2026-10-12-chunk-160t+40-bge-m3-norm-dense`.
- `test_run_id_without_normalize_is_unchanged`: `normalize=False` gives the AC-06 ids.
- `test_run_config_records_normalize_only_when_on`: `run_config(5, cfg, "m", normalize=True)["normalize"] is True`; `"normalize" not in run_config(5, cfg, "m")`.

## Implementation notes
- The duplicate-run check must use the same id function with the same arguments.
- The reachable-span counter does not need normalisation: it only counts tokens.
- Smoke check, not a test: `uv run python eval/run_chunk_sweep.py 160t+40 --normalize --channels DENSE --no-log` prints DENSE overall 0.593, the scratch number in embedding-sweep.07.

## Verify
`uv run pytest tests/eval/test_chunk_sweep.py -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
- manual: the smoke run prints DENSE overall 0.593.
