# Embedder and reranker adapters

The fastembed adapters: query and passage modes, prefixes, custom models, caching, downloads.

### embedding-sweep.05 · discovery · 2026-10-12 · step 03
fastembed 0.8.0 warns that `intfloat/multilingual-e5-large` "now uses mean pooling instead of CLS embedding" and suggests pinning 0.5.1 or `add_custom_model` to keep CLS. Mean pooling is what the e5 model card prescribes, so the current fastembed behaviour is the correct one and the warning needs no action.
**Why:** a run on CLS pooling would measure a misconfigured e5. Rejected: pinning fastembed 0.5.1 or registering a CLS variant, both of which restore the wrong pooling.

### embedding-sweep.09 · discovery · 2026-10-12 · step 04
bge-m3 loads through `add_custom_model` with `onnx/model.onnx` plus `onnx/model.onnx_data`; the `Constant_7_attr__value` file the plan flagged was not needed. Load and dimension probe took 25 s after the 2.27 GB download; vectors are 1024-dimensional with length 1.0. The early-passing test `test_known_models_are_not_registered` guards kept behaviour; a mutation that registers unconditionally made it fail.
**Why:** retires the one open risk from gate 0 on bge-m3 loading; the born-green note follows the build rule.

### embedding-sweep.10 · issue · 2026-10-12 · step 04
Outside the eval, nothing pins fastembed's model cache, so the production container downloads models into fastembed's default, the system temp directory (`/var/folders/.../T/fastembed_cache` here). `src/ariostea/eval/wiki_index.py` documents that macOS purges that directory and leaves dangling snapshot links that break or hang the next load, and pins `~/.cache/fastembed` for the eval only. The server is exposed to the same failure. bge-m3 now sits in both caches (4.5 GB in total) because the step 04 check ran outside the eval.
**Why:** pre-existing and outside this cycle's scope; raised at gate M2 for the owner to schedule. Not fixed here.
Superseded by bge-m3-default.13.

### bge-m3-default.13 · decision · 2026-10-13 · step 03
Both fastembed adapters, `FastEmbedEmbeddings` and `FastEmbedReranker`, load models from `fastembed_cache_dir()` in `src/ariostea/adapters/fastembed_cache.py`: `FASTEMBED_CACHE_PATH` if set, else `~/.cache/fastembed`, read at call time. The eval's environment-variable pin in `wiki_index.py` is gone. Resolves embedding-sweep.10.
**Why:** fastembed's default, the system temp directory, is purged by macOS and leaves broken model links that make the next load fail or hang. Rejected: a config option for the path (nobody needs it yet; the environment variable covers the exception).

### server-robustness.12 · discovery · 2026-10-27 · step 06
One fastembed embedder (bge-m3) used from two threads at once, one embedding notes and counting tokens and the other embedding queries, gives the same vectors (within 1e-5) and token counts as the same calls in sequence, and raises nothing. The MCP server's search thread and its index thread share the embedder with no lock (`test_embedder_from_two_threads_matches_one`).
**Why:** ONNX Runtime sessions run concurrently and the tokenizer is only read. Rejected: a lock in the adapter (a search would wait for a note's embedding).

### search-speed.13 · decision · 2026-10-28 · challenge round
`FastEmbedReranker` registers `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` with fastembed (`_CUSTOM_MODELS`, `_register_if_custom`, asking fastembed's registry first since a second registration raises) and loads its fp32 `onnx/model.onnx`. Pitfall: fastembed downloads a model's files only when no snapshot exists yet, so files added to a model already in the cache (an extra ONNX file, a custom model's config and tokenizer) must be fetched explicitly, or the model's cache folder deleted first.
**Why:** the arm64 int8 file would tie the model to the CPU type and change its scores. Rejected: the `model_qint8_arm64` file (0.67 s against 0.95 s at pool 50).

### search-speed.18 · issue · 2026-10-28 · step 04
Open, machine fact: the Hugging Face token in `~/.cache/huggingface/token` has expired, and huggingface_hub sends it even for public repositories, so a model not cached yet fails to download with a 401 and the reranker falls back to fused order. The integration test patches `huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN`; eval runs that download set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. The token is the owner's: leave it alone (`hf auth login` refreshes it).
**Why:** cost a debugging round in search-speed; any new model download on this machine hits it.
