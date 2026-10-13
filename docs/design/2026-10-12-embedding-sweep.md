# Design: Embedding model sweep, aimed at cross-lingual

**Status:** implemented and measured 2026-10-12; results in `docs/retrieval-tuning.md`, "Embedding model". The normalise setting was added at gate M1, after the first runs showed the store ranks raw vectors.
**Date:** 2026-10-12

## Problem

Cross-lingual is the one query type the pipeline handles badly. At the adopted
chunking (160 model tokens, overlap 40), span recall at k=5 is:

| channel | buried | cross_lingual | exact_term | paraphrase | overall |
|---|---|---|---|---|---|
| dense | 0.575 | 0.500 | 0.561 | 0.550 | 0.545 |
| sparse | 0.850 | 0.022 | 0.805 | 0.525 | 0.533 |
| fused (no reranker) | 0.825 | 0.304 | 0.707 | 0.725 | 0.629 |
| hybrid (production) | 1.000 | 0.696 | 0.951 | 0.900 | 0.880 |

Control runs: `2026-09-24-chunk-160t+40-dense-hybrid-sparse` and
`2026-09-28-chunk-160t+40-fused`.

Sparse cannot match a query in one language against an article in another, so
on cross-lingual it contributes one case in 46 and mostly adds noise to the
fused pool: fused cross-lingual (0.304) is lower than dense alone (0.500). Dense
retrieval is the only first-stage channel that works on this type, and the
reranker can only reorder what the first stage hands it. Both contextual
experiments hurt cross-lingual most (blurbs 0.609, per-chunk context 0.630 and
0.652), which points at the embedding model rather than the text being indexed.

The model in use, `paraphrase-multilingual-mpnet-base-v2`, is a 2021 model
distilled from paraphrase pairs. fastembed truncates its input at 384 tokens.
Newer multilingual models were trained for retrieval on query and passage
pairs, which is the task here.

## Candidates

| model | dim | download | input (fastembed) | prefixes | licence | fastembed |
|---|---|---|---|---|---|---|
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (control) | 768 | 1.0 GB | 384 | none | Apache 2.0 | native |
| `intfloat/multilingual-e5-large` | 1024 | 2.24 GB | 512 | `query: ` and `passage: `, required | MIT | native |
| `BAAI/bge-m3` | 1024 | 2.27 GB | 8192 | none | MIT | custom registration |
| `jinaai/jina-embeddings-v3` | 1024 | 2.29 GB | 1024 | task ids, handled by fastembed | CC BY-NC 4.0 | native |

All four use the XLM-RoBERTa sentencepiece tokenizer, so chunking by 160 model
tokens should cut the corpus in nearly the same places under every model. The
runner's coverage line and chunk count confirm this per model; if they differ
noticeably, the write-up says so.

Notes per candidate:

- **e5-large** needs `query: ` before queries and `passage: ` before passages.
  fastembed does not add them. Without them its retrieval quality drops, so a
  run without prefixes would measure a misconfiguration.
- **bge-m3** is not in fastembed's list. The Hugging Face repository ships an
  ONNX export (`onnx/model.onnx` with external weights in
  `onnx/model.onnx_data`), which `TextEmbedding.add_custom_model` can register
  with CLS pooling, normalisation and 1024 dimensions. This is the one candidate
  whose loading is unproven; the pilot checks it first.
- **jina-embeddings-v3** selects a retrieval task per call, query or passage.
  fastembed's `query_embed` and `passage_embed` set it. Its licence is
  non-commercial. Measuring it costs nothing extra, but it becomes the default
  only on the owner's call.

## Approach

Three ways to get there:

1. **Extend the chunking sweep runner with a `--model` flag** (recommended). The
   runner already indexes, scores every channel, counts reachable spans, scores
   the discriminated-out cases and logs against a control. The model is one more
   axis of the same run, and the flag also makes a later chunking pass for a new
   model a single command.
2. **A separate embedding runner.** Cleaner name, but it would duplicate the
   runner's indexing, scoring and logging for no new behaviour.
3. **Translate queries at search time with a chat model.** Rejected for now. It
   adds an LLM call to every search and a runtime dependency on a chat server,
   while a better embedding model costs nothing extra per query beyond a larger
   forward pass.

### 1. The adapter embeds queries and passages as the model expects

`FastEmbedEmbeddings` today calls `embed` for both documents and queries. It
changes to call `passage_embed` for documents and `query_embed` for queries,
and applies a small per-model prefix table (only e5 has an entry). The
`EmbeddingProvider` port already separates `embed_documents` from
`embed_query`, so the port does not change.

- `count_tokens` keeps counting the raw text without the prefix, so chunk sizes
  stay in the same units across models. The prefix adds a handful of tokens at
  embedding time, far inside every candidate's input limit.
- The fingerprint includes the prefix scheme only when a model has one. The
  control's fingerprint stays `fastembed:sentence-transformers/paraphrase-multilingual-mpnet-base-v2`,
  so existing vaults do not reindex because of this change.

### 2. bge-m3 registration

The adapter keeps a table of extra models it knows how to register with
fastembed, holding one entry for bge-m3, and registers an entry the first time
that model is requested. Registration is process-wide in fastembed, so it
happens once. If the pilot shows the export does not load or does not match the
reference embeddings closely enough to trust, bge-m3 leaves the sweep and the
write-up records why.

### 3. The eval index takes a model

`wiki_config` and `index_wiki_corpus` gain a `model` parameter that defaults to
`MULTILINGUAL_MODEL`, so every existing caller measures what it measured
before. `wiki_channels` stops building its own embedder from the constant and
reads the model from the container's config, so the dense channel always
queries with the model the index was built with.

### 4. The runner takes a model

`eval/run_chunk_sweep.py` gains `--model`, defaulting to the control model.

- The run id gains a model slug only when the model is not the default, so
  every logged id stays valid and a candidate run cannot clash with a control.
- The log entry already records `embedding` in its config; it now carries the
  model actually used.
- Each run also records its cost: seconds to index the corpus and median
  milliseconds to embed one query, measured over the gold queries. These do
  not enter the decision rule, but a 2 GB model that wins by one case is a
  different trade from one that wins by ten.

## Experiment protocol

Fixed now, before the real runs.

**Step 0: control still reproduces.** Rerun the control at 160t+40 with
DENSE, SPARSE and FUSED at the current commit. It must match the logged numbers
exactly (0.545, 0.533, 0.629). Code has changed since those runs (fusion keeps
the first-seen candidate, the store returns blurbs). If the numbers move, rerun
HYBRID for the control as well and use the new runs as controls.

**Step 1: pilot.** For each candidate: load it, check the dimension, check that
e5 embeds a query and a passage differently, embed 100 chunks and time them.
Candidates that fail to load are dropped and recorded.

**Step 2: first stage.** DENSE, SPARSE and FUSED for each surviving candidate at
160t+40. SPARSE does not depend on the model and should equal the control; it
is a check that the chunks did not change. FUSED shows first-stage gains the
reranker might later hide, the lesson of the context experiments.

**Step 3: production pipeline.** HYBRID for every surviving candidate, unless
no candidate beats the control's dense overall in step 2; then the sweep stops
and reports before spending the hybrid hours. Beyond that one stop, step 2 does
not pick candidates: with at most three, each gets its hybrid run. About an hour
each on CPU, dominated by the reranker, which is the same for every candidate.

**Primary metric.** Hybrid overall span recall at k=5, span MRR as tie-breaker.

**Decision rule**, unchanged from the earlier experiments so the results stay
comparable. A candidate becomes the recommended default only if:

- hybrid overall span recall beats the control by at least 0.03; and
- no query type loses more than 0.05 hybrid span recall against the control; and
- at least 160 of 167 spans stay reachable.

Among candidates that pass, the higher span MRR wins, then the lower cost.

**Hypothesis, stated in advance.** Dense cross-lingual rises clearly above
0.500 and pulls hybrid cross-lingual up from 0.696. One cross-lingual case is
0.022 on that type, and five cases on cross-lingual alone move the overall
number by 0.03. A model that only helps cross-lingual can therefore pass the
rule on its own, and one that helps cross-lingual while losing on buried or
exact-term will show it per type.

Every run goes into `eval/results/runs.jsonl` with the control as its
comparison, the shaded page is re-rendered, and the log artifact is
republished.

## Confounds

- **The gold set was filtered with the control model's channels.** The
  discrimination gate dropped 29 cases every channel answered at rank 1, and
  the ambiguity gate judged competitors the control model retrieved. The set is
  biased toward what the control finds hard, which flatters any change. The
  runner already scores the 29 dropped cases; a candidate that loses them is
  reported as such.
- **The chunk size was tuned for the control model.** 160 tokens sits near its
  128-token training length. The candidates were trained on longer inputs and
  may prefer longer chunks. The sweep holds chunking fixed so it measures the
  model alone. If a candidate wins, a DENSE, SPARSE pass at 256t+64 with that
  model is a one-line follow-up, reported separately and not part of this
  decision.
- **The reranker may absorb the gain.** That is what happened with both context
  experiments: first-stage numbers rose and hybrid did not. FUSED is in the
  protocol to make this visible rather than inferred.
- **Query latency.** A model about twice the size of the control embeds each
  query more slowly on CPU. The cost fields measure it.

## If a candidate passes

Changing the default embedding model forces every vault to reindex, so it is
the owner's call, made on the measured gain and cost. The change itself is
`EmbeddingCfg.local_model`, the adapter default, `ariostea.example.toml`,
`MULTILINGUAL_MODEL` and the README's "Where it stands". The fastembed download
grows from 1 GB to about 2.3 GB at most, and the index grows by a third (1024 against
768 dimensions), a few MB on the wiki corpus.

## Scope

In scope: the adapter's query and passage handling, e5 prefixes, bge-m3
registration, the model parameter through the eval index and the runner, cost
fields in the log, the sweep, and the write-up in `docs/retrieval-tuning.md`.

Out of scope:

- Query translation at search time.
- Swapping the reranker.
- Re-tuning chunking for a new model, beyond the single follow-up pass above.
- Regenerating the gold set. It would make every past run incomparable.
- Remote embedding APIs through `openai_compat`. The tool is meant to run
  locally.

## Owner decisions

Open, each with the default the plan will follow unless the owner says
otherwise:

1. **Candidates:** e5-large and bge-m3, with jina-embeddings-v3 measured as
   well but adoptable only on a separate licence call. Default: measure all
   three.
2. **Decision rule:** unchanged from chunking and contextual retrieval.
   Default: keep it.
3. **Compute and disk:** about three hours of CPU for HYBRID plus indexing time
   the pilot will measure, and about 7 GB of model downloads in
   `~/.cache/fastembed`. Default: proceed, with the stop after step 2
   described in the protocol.
