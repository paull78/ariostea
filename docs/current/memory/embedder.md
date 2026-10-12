# Embedder adapter

### embedding-sweep.04 · discovery · 2026-10-12 · step 01
Two step 01 tests passed before any implementation: `test_control_fingerprint_is_unchanged` and `test_count_tokens_ignores_the_prefix`. Both guard behaviour the change had to keep, not add. A deliberate break (a suffix on the fingerprint, the passage prefix inside `count_tokens`) made both fail, so they are real guards.
**Why:** recorded per the build rule on tests born green; no action needed.

### embedding-sweep.05 · discovery · 2026-10-12 · step 03
fastembed 0.8.0 warns that `intfloat/multilingual-e5-large` "now uses mean pooling instead of CLS embedding" and suggests pinning 0.5.1 or `add_custom_model` to keep CLS. Mean pooling is what the e5 model card prescribes, so the current fastembed behaviour is the correct one and the warning needs no action.
**Why:** a run on CLS pooling would measure a misconfigured e5. Rejected: pinning fastembed 0.5.1 or registering a CLS variant, both of which restore the wrong pooling.

### embedding-sweep.09 · discovery · 2026-10-12 · step 04
bge-m3 loads through `add_custom_model` with `onnx/model.onnx` plus `onnx/model.onnx_data`; the `Constant_7_attr__value` file the plan flagged was not needed. Load and dimension probe took 25 s after the 2.27 GB download; vectors are 1024-dimensional with length 1.0. The early-passing test `test_known_models_are_not_registered` guards kept behaviour; a mutation that registers unconditionally made it fail.
**Why:** retires the one open risk from gate 0 on bge-m3 loading; the born-green note follows the build rule.

### embedding-sweep.10 · issue · 2026-10-12 · step 04
Outside the eval, nothing pins fastembed's model cache, so the production container downloads models into fastembed's default, the system temp directory (`/var/folders/.../T/fastembed_cache` here). `src/ariostea/eval/wiki_index.py` documents that macOS purges that directory and leaves dangling snapshot links that break or hang the next load, and pins `~/.cache/fastembed` for the eval only. The server is exposed to the same failure. bge-m3 now sits in both caches (4.5 GB in total) because the step 04 check ran outside the eval.
**Why:** pre-existing and outside this cycle's scope; raised at gate M2 for the owner to schedule. Not fixed here.

### embedding-sweep.11 · discovery · 2026-10-12 · step 10
`test_normalize_off_returns_the_raw_vectors` passed before any implementation because it guards today's behaviour; forcing normalisation on made it fail. The container test goes through a new `_build_embeddings(cfg)` helper, mirroring `_build_reranker`, instead of a full `build_container` call as the step file said.
**Why:** born-green note per the build rule; the helper keeps the container test free of a real store and model, like its reranker sibling.
