# Step 05 · Switch the defaults

**Milestone:** 2 · **Needs:** 02, 04 · **Covers:** AC-01, AC-05 (default), AC-08

## Goal
A fresh install and a plain eval run both use bge-m3, with the chunk size step 04 chose. An existing vault on the old model reindexes cleanly.

## Files touched
- `src/ariostea/config/schema.py`: `EmbeddingCfg.local_model = "BAAI/bge-m3"`; `ChunkingCfg` defaults to 256 and 64 if step 04 adopted it.
- `src/ariostea/adapters/embedding/fastembed_local.py`: constructor default `BAAI/bge-m3`.
- `src/ariostea/eval/wiki_index.py`: `MULTILINGUAL_MODEL = "BAAI/bge-m3"`, comment updated (bge-m3-default.03).
- `ariostea.example.toml`: bge-m3 as the default with one line on why; the old model as a commented option; chunking values if changed.
- Tests: `tests/config/test_schema.py` (defaults), `tests/eval/test_chunk_sweep.py` and `tests/eval/test_wiki_index.py` (pinned model), plus any test that asserted the old defaults.

## Tests to write first
- `test_embedding_defaults_to_the_measured_model` (rewrite): asserts `BAAI/bge-m3`.
- `test_chunking_defaults_to_the_measured_policy` (update only if step 04 adopted 256t+64): asserts 256, 64, `model_tokens`.
- `test_eval_pins_the_shipped_model`: `MULTILINGUAL_MODEL == EmbeddingCfg().local_model`.
- `test_old_model_run_ids_carry_its_slug`: `sweep_run_id("d", "160t+40", ["HYBRID"], model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")` contains `paraphrase-multilingual-mpnet-base-v2`.

## Implementation notes
- Grep for every hard-coded `paraphrase-multilingual-mpnet-base-v2` in `src/`, `tests/` and `eval/` and decide each one: a default moves, a reference to past runs stays.
- The integration tests that load the default model now download bge-m3 if missing (2.27 GB, already cached here).
- Run the full suite, not only the fast one: the integration tests exercise the real default.

## Verify
`uv run pytest -q` → all pass (full suite, about twelve minutes).

## Done when
- All listed tests pass, and the full suite passes.
- manual: owner runs `uv run ariostea reindex` on the real vault (indexed with the old model) and then `uv run ariostea status`; the reindex finishes without error and status shows a fingerprint starting `fastembed:BAAI/bge-m3`.
