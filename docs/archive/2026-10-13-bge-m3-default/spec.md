# Spec: bge-m3-default

Cycle: bge-m3-default · Type: feature · Branch: feat/bge-m3-default · Status: approved

Background: project memory embedding-sweep.17, .18 and .19; `docs/retrieval-tuning.md`, "Embedding model".

## Goal

Make `BAAI/bge-m3` the default embedding model, with the chunk size measured for it, so a fresh install gets the configuration that found 155 of 167 wiki cases instead of 147. Fix the two things that would make the switch fail or decay on a real machine: an existing index cannot take vectors of a new size, and the server keeps its models in a directory macOS purges.

## Scope and non-goals

In scope:
- One measurement: HYBRID for bge-m3 at 256t+64, compared with bge-m3 at 160t+40, to fix the default chunk size.
- `EmbeddingCfg.local_model`, the adapter default, `ariostea.example.toml` and the chunking default (if the measurement says so) move to the new configuration.
- The store rebuilds its vector table when the configured dimension differs from the one on disk, so a vault indexed with the old model re-embeds instead of failing (bge-m3-default.01).
- Both fastembed adapters, embedder and reranker, keep their models under `~/.cache/fastembed` unless `FASTEMBED_CACHE_PATH` says otherwise (embedding-sweep.10).
- The eval's pinned model `MULTILINGUAL_MODEL` moves to bge-m3, so a plain eval run measures what ships (bge-m3-default.03).
- README: "Where it stands" and a fourth learning-journey trial; `docs/retrieval-tuning.md`: the decision section and the 256t+64 result; the experiment log and its artifact.

Non-goals:
- Turning `embedding.normalize` on by default. bge-m3 returns unit vectors already.
- Changing the reranker, fusion or contextual indexing.
- Regenerating or extending the gold set.
- Chunk sizes other than 160t+40 and 256t+64.

## Behaviour

- **Fresh install:** with no `[embedding]` section, the server indexes with bge-m3, downloading it once (2.27 GB) into `~/.cache/fastembed`.
- **Existing vault on the old model:** the next index sees a changed fingerprint and a different vector size, rebuilds the vector table, and re-embeds every note. No error, no manual step.
- **Explicit old model:** a config with `local_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"` keeps working, indexes at 768 dimensions and keeps its fingerprint.
- **Model cache:** models load from `~/.cache/fastembed`; setting `FASTEMBED_CACHE_PATH` moves it. Nothing is written to the system temp directory.

### Measurement

`uv run python eval/run_chunk_sweep.py 256t+64 --model BAAI/bge-m3 --normalize --channels HYBRID` against `2026-10-12-chunk-160t+40-bge-m3-norm-hybrid` (155 of 167; cross_lingual 41 of 46). 256t+64 becomes the default chunk size if it finds at least 155 of 167 hybrid hits and no query type loses more than 2 cases; otherwise 160t+40 stays (bge-m3-default.02).

## Architecture

- `SqliteStore` reads the dimension of the existing `chunks_vec` table on open. If it differs from the configured one, it drops and recreates `chunks_vec`. The chunk rows stay; the fingerprint change already forces every note to re-embed.
- A small shared helper returns the fastembed cache directory (`FASTEMBED_CACHE_PATH`, else `~/.cache/fastembed`); `FastEmbedEmbeddings` and `FastEmbedReranker` pass it as `cache_dir`. The eval's environment-variable pin in `wiki_index.py` becomes redundant and goes.

## Acceptance criteria

- AC-01 · `EmbeddingCfg().local_model == "BAAI/bge-m3"`, `FastEmbedEmbeddings()` loads bge-m3, and `ariostea.example.toml` names it as the default.
- AC-02 · A store created at dimension 3 with one vector, reopened at dimension 4, accepts a 4-dimension vector; reopened at the same dimension, it keeps its vectors.
- AC-03 · Indexing a small vault with a fake 3-dimension embedder, then again with a fake 4-dimension embedder of a different fingerprint, succeeds and leaves only 4-dimension vectors.
- AC-04 · `FastEmbedEmbeddings` and `FastEmbedReranker` pass `cache_dir` equal to `FASTEMBED_CACHE_PATH` when set and `~/.cache/fastembed` otherwise.
- AC-05 · The 256t+64 HYBRID run is logged against the bge-m3 160t+40 control. `ChunkingCfg()` defaults to 256 and 64 if that run finds at least 155 of 167 hits with no type down more than 2 cases, and stays at 160 and 40 otherwise.
- AC-06 · README "Where it stands" describes bge-m3 as the default with its measured numbers, and the learning journey has a fourth trial covering the embedding sweep and normalisation.
- AC-07 · `docs/retrieval-tuning.md` records the 256t+64 result and the final decision; the log artifact is republished.
- AC-08 · `MULTILINGUAL_MODEL` is `BAAI/bge-m3`; `run_chunk_sweep.py` without `--model` logs `embedding: BAAI/bge-m3` and puts no model slug in the run id; `--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2` still runs the old model, with its slug in the id.

## Testing approach

- Unit tests for the store's dimension check with real sqlite-vec on temporary files (AC-02).
- An indexing test with fake embedders of different dimensions and fingerprints (AC-03).
- Unit tests with fake fastembed classes recording `cache_dir` (AC-04), config tests for the defaults (AC-01), and run id and config tests for the eval's pinned model (AC-08).
- Integration: the existing embedder tests load the default model, now bge-m3.
- AC-05 to AC-07 are checked by reading the log and the docs.

Test command: `uv run pytest -m "not integration"` for the fast suite, `uv run pytest` for everything.

## Open questions

- none
