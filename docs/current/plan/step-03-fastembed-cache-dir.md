# Step 03 · Fastembed cache directory for both adapters

**Milestone:** 1 · **Needs:** none · **Covers:** AC-04

## Goal
`FastEmbedEmbeddings` and `FastEmbedReranker` load models from `FASTEMBED_CACHE_PATH` when set and `~/.cache/fastembed` otherwise, never from fastembed's default in the system temp directory, which macOS purges (embedding-sweep.10).

## Files touched
- `src/ariostea/adapters/fastembed_cache.py`: new, `fastembed_cache_dir() -> str`.
- `src/ariostea/adapters/embedding/fastembed_local.py`: pass `cache_dir` to `TextEmbedding`.
- `src/ariostea/adapters/rerank/fastembed_rerank.py`: pass `cache_dir` to `TextCrossEncoder`.
- `src/ariostea/eval/wiki_index.py`: remove the `os.environ.setdefault("FASTEMBED_CACHE_PATH", ...)` pin and its comment, keep a one-line pointer to the helper.
- `tests/adapters/test_fastembed_cache.py`: new; `tests/adapters/embedding/test_fastembed_local.py` and `tests/adapters/rerank/test_fastembed_rerank.py`: fakes accept and record `cache_dir`.

## Tests to write first
- `test_cache_dir_defaults_under_home`: with `FASTEMBED_CACHE_PATH` unset (monkeypatch.delenv), returns `str(Path.home() / ".cache" / "fastembed")`.
- `test_cache_dir_follows_the_environment`: with it set to `tmp_path`, returns that path.
- `test_embedder_passes_the_cache_dir`: the fake `TextEmbedding` records `cache_dir` equal to `fastembed_cache_dir()`.
- `test_reranker_passes_the_cache_dir`: a fake `TextCrossEncoder` monkeypatched into `fastembed_rerank` records the same.

## Implementation notes
- Read the environment at call time, not import time, so tests and users can change it.
- Existing fakes in `test_fastembed_local.py` take `model_name` only; widen them to `**kwargs`.
- bge-m3 is already in `~/.cache/fastembed` (and in the temp cache, embedding-sweep.10); after this step the temp copy is unused. Deleting it is the owner's call.

## Verify
`uv run pytest tests/adapters -m "not integration" -q` → all pass; `uv run pytest -m "not integration" -q` → all pass.

## Done when
- All listed tests pass, and the fast suite passes.
