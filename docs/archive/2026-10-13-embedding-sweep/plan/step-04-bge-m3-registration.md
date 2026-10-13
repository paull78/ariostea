# Step 04 · bge-m3 custom registration

**Milestone:** 2 · **Needs:** 01 · **Covers:** AC-04

## Goal
`FastEmbedEmbeddings("BAAI/bge-m3")` works with no manual setup: the adapter registers the model with fastembed the first time it is requested, once per process.

## Files touched
- `src/ariostea/adapters/embedding/fastembed_local.py`: `_CUSTOM_MODELS` table and a `_register_if_custom(name)` guard.
- `tests/adapters/embedding/test_fastembed_local.py`: unit and integration tests.

## Tests to write first
- `test_custom_model_is_registered_before_loading`: with a fake `TextEmbedding` recording `add_custom_model` calls, constructing `FastEmbedEmbeddings("BAAI/bge-m3")` calls `add_custom_model` with `model="BAAI/bge-m3"`, `dim=1024`, CLS pooling, `normalization=True`, `model_file="onnx/model.onnx"`, `additional_files` containing `"onnx/model.onnx_data"`, before the fake is instantiated.
- `test_custom_model_is_registered_once`: constructing it twice calls `add_custom_model` once.
- `test_known_models_are_not_registered`: the control model triggers no `add_custom_model` call.
- Integration, `test_bge_m3_loads_and_embeds_1024_dims`: skipped unless the model is in the fastembed cache; asserts `dimension == 1024` and that a query is closer to its matching document than to an unrelated one.

## Implementation notes
- Registration: `TextEmbedding.add_custom_model(model="BAAI/bge-m3", pooling=PoolingType.CLS, normalization=True, sources=ModelSource(hf="BAAI/bge-m3"), dim=1024, model_file="onnx/model.onnx", additional_files=["onnx/model.onnx_data"], license="mit", size_in_gb=2.27)`.
- fastembed raises `ValueError` on a second registration of the same name. Guard by checking `TextEmbedding.list_supported_models()` for the name before registering, not with a module flag alone, so a model registered elsewhere is not re-registered.
- Pitfall: the repo's `onnx/` folder also holds `Constant_7_attr__value`. If loading fails with a missing external-data file, add it to `additional_files`. If loading still fails, record an issue entry and drop bge-m3 from the sweep (design doc, Candidates).
- The integration test checks the cache by looking for `models--BAAI--bge-m3` under `FASTEMBED_CACHE_PATH` or `~/.cache/fastembed`.

## Verify
`uv run pytest tests/adapters/embedding -m "not integration" -q` → all pass. After step 06 downloads the model: `uv run pytest tests/adapters/embedding -q` → the bge-m3 test runs and passes.

## Done when
- Unit tests pass, and the fast suite passes.
