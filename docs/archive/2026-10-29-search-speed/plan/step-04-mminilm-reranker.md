# Step 04 · mMiniLM in the reranker adapter

**Milestone:** 2 · **Needs:** none · **Covers:** AC-03

## Goal
`rerank.model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"` builds a working reranker: the adapter registers it as a custom fastembed model (fp32 `onnx/model.onnx`, search-speed.13) and fastembed fetches its files into an empty cache.

## Files touched
- `src/ariostea/adapters/rerank/fastembed_rerank.py`: `_CUSTOM_MODELS` and `_register_if_custom`, mirroring `src/ariostea/adapters/embedding/fastembed_local.py` (ask fastembed's registry, not a local flag).
- `tests/adapters/rerank/test_fastembed_rerank.py` (new or extended): the integration test.
- `tests/config/test_container.py` or `tests/config/test_schema.py`: config accepts the model.

## Tests to write first
- `test_mminilm_loads_into_an_empty_cache_and_ranks` (`integration`): with `FASTEMBED_CACHE_PATH` set to a `tmp_path`, `FastEmbedReranker("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")` loads; the cache holds `onnx/model.onnx`, `config.json` and `tokenizer.json`; for the query "What is the capital of France?" the passage "Paris is the capital of France." ranks above "Bananas are rich in potassium." and "The Moon orbits the Earth."; and the same check passes for an Italian passage "Parigi è la capitale della Francia." against the English query (multilingual).
- `test_container_builds_mminilm_without_fallback` (`integration`): `build_container` with that `rerank.model` logs no warning on `ariostea.config.container` (use `fail_on_warnings`).
- `test_registration_is_idempotent`: constructing two rerankers with the model in one process does not raise (fastembed refuses a second registration).

## Implementation notes
- `TextCrossEncoder.add_custom_model(model=..., sources=ModelSource(hf=...), model_file="onnx/model.onnx", size_in_gb=0.47, license="apache-2.0", description=...)`. Check the signature in the installed fastembed.
- search-speed.05 pitfall: fastembed downloads a model's files only when it has no snapshot yet; the scratch scripts already put a partial snapshot in `~/.cache/fastembed`. The empty-cache test is the real check; if the user's cache holds a partial snapshot from the scratch runs, delete `~/.cache/fastembed/models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1` before step 05 (scratch files only, made in this cycle).

## Verify
`uv run pytest tests/adapters/rerank -q -m integration` → passes, and `uv run pytest -m "not integration" -q` → passes.

## Done when
- The integration tests pass on an empty cache.
