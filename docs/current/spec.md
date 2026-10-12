# Spec: embedding-sweep

Cycle: embedding-sweep · Type: feature · Branch: feat/embedding-sweep · Status: approved

Full rationale, candidate table and confounds: `docs/design/2026-10-12-embedding-sweep.md`.

## Goal

Find out whether a newer multilingual embedding model lifts cross-lingual retrieval enough to change the default. Cross-lingual hybrid span recall at k=5 is 0.696 against 0.90 or better on the other three types. Sparse retrieval gets one case in 46 there, so the dense channel, and therefore the embedding model, is the lever.

## Scope and non-goals

In scope:
- The local embedder embeds documents as passages and queries as queries, with the prefixes a model requires.
- `BAAI/bge-m3` registered as a custom fastembed model.
- The eval index and the chunking sweep runner take an embedding model.
- Cost fields in the experiment log: indexing time and query embedding latency.
- Three candidates against the control: `intfloat/multilingual-e5-large`, `BAAI/bge-m3` and `jinaai/jina-embeddings-v3`. jina is CC BY-NC 4.0, so it can become the default only on a separate licence call (embedding-sweep.01).
- The sweep itself at 160t+40, logged and written up in `docs/retrieval-tuning.md`, with the log artifact republished.

Non-goals:
- Changing the default model. If a candidate passes, the owner decides separately, because the change forces every vault to reindex.
- Query translation at search time.
- Swapping the reranker.
- Re-tuning chunking for a new model. The one exception: if a candidate passes the decision rule, a DENSE,SPARSE pass at 256t+64 with that model runs and is reported separately, outside the decision.
- Regenerating the gold set.
- Remote embedding APIs.

## Behaviour

- **Production, control model:** a vault indexed with `paraphrase-multilingual-mpnet-base-v2` keeps its fingerprint, so upgrading does not trigger a reindex. Its embeddings do not change.
- **Production, e5:** configuring `intfloat/multilingual-e5-large` indexes passages with a `passage: ` prefix and embeds queries with a `query: ` prefix. Its fingerprint differs from an unprefixed one.
- **Production, bge-m3:** configuring `BAAI/bge-m3` loads the model from its Hugging Face ONNX export, with no manual registration by the user.
- **Eval:** `uv run python eval/run_chunk_sweep.py 160t+40 --model <name> --channels DENSE,SPARSE,FUSED` indexes the wiki corpus with that model, scores the channels, and logs a run whose id carries the model slug: the part of the name after the last `/`, lowercased, e.g. `multilingual-e5-large`. Without `--model` the runner behaves and names runs exactly as today.

### Experiment protocol

Fixed before the runs. Details in the design doc.

1. **Control reproduces.** Rerun the control at 160t+40 with DENSE, SPARSE and FUSED. It must match the logged 0.545, 0.533 and 0.629. If not, rerun HYBRID too and use the new runs as controls.
2. **Pilot.** Load each candidate, check the dimension, check that e5 embeds a query and a passage differently, time 100 chunks. Drop and record any that fail.
3. **First stage.** DENSE, SPARSE and FUSED per surviving candidate.
4. **Production pipeline.** HYBRID per surviving candidate, unless no candidate beats the control's dense overall in step 3. In that case, stop and report (embedding-sweep.03).

**Decision rule**, unchanged from earlier experiments (embedding-sweep.02). A candidate is recommended only if hybrid overall span recall beats the control by at least 0.03, no type loses more than 0.05, and at least 160 of 167 spans stay reachable. Ties go to span MRR, then lower cost.

## Architecture

- `FastEmbedEmbeddings` calls `passage_embed` and `query_embed` and applies a per-model prefix table. The `EmbeddingProvider` port does not change.
- `count_tokens` counts raw text without the prefix.
- The fingerprint carries the prefix scheme only for models that have one.
- The adapter keeps a table of extra fastembed models it registers on first use. bge-m3 is the only entry.
- `wiki_config` and `index_wiki_corpus` take a `model` defaulting to `MULTILINGUAL_MODEL`. `wiki_channels` reads the model from the container's config instead of the constant.
- `run_chunk_sweep.py` gains `--model` and records cost fields on each run.

## Acceptance criteria

- AC-01 · For the control model, `embed_documents` and `embed_query` return the same vectors as fastembed's plain `embed` on the same text, and the fingerprint is still `fastembed:sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- AC-02 · For `intfloat/multilingual-e5-large`, the text sent to fastembed starts with `passage: ` for documents and `query: ` for queries, and the fingerprint differs from the unprefixed model name.
- AC-03 · `count_tokens` returns the same count for a text whether or not the model has a prefix.
- AC-04 · Constructing `FastEmbedEmbeddings("BAAI/bge-m3")` registers the custom model once per process and yields 1024-dimension vectors.
- AC-05 · `wiki_config` and `index_wiki_corpus` without a `model` argument build exactly the config they build today. With a model, the config and the dense channel both use that model.
- AC-06 · `run_chunk_sweep.py` without `--model` produces the same run ids and logged config as today. With `--model`, the run id contains the model slug and the logged `embedding` field is that model.
- AC-07 · Each logged run carries the seconds taken to index the corpus and the median milliseconds to embed one gold query with the run's model.
- AC-08 · Every sweep run is in `eval/results/runs.jsonl` against the control, `experiment_log.html` is re-rendered, and the log artifact is republished.
- AC-09 · `docs/retrieval-tuning.md` has a section with the per-type tables for each candidate, the cost figures, and a verdict against the decision rule.

## Testing approach

- Unit tests with a fake fastembed model for the prefix table, the query and passage routing, the fingerprint and `count_tokens`. They cover AC-01 to AC-03.
- Integration tests, marked `integration`, that load real models. They cover AC-01 and AC-04. The e5 and bge-m3 loads download about 2.2 GB each, so they skip when the model is not cached.
- Unit tests for the `model` parameter in `wiki_index` and for run ids and logged config in the runner. They cover AC-05 to AC-07.
- AC-08 and AC-09 are checked by reading the log and the write-up at the end of the cycle.

Test command: `uv run pytest -m "not integration"` for the fast suite, `uv run pytest` for everything.

## Open questions

- none
