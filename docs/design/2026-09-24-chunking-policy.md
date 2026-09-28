# Design: Chunking policy, measured

**Status:** approved 2026-09-24, ready for implementation planning
**Date:** 2026-09-24

## Problem

Chunking is the first experiment the span-anchored wiki eval was built for, and
the current policy has never been measured. `HeadingAwareChunker` splits a note
at every heading and cuts any section longer than 512 whitespace-separated words
into consecutive, non-overlapping pieces. The 512 is a constructor default; the
container builds the chunker with no arguments, so there is no way to change it
without editing code.

Measured on the wiki corpus, that policy collides with the embedding model the
eval and the owner's own config use, `paraphrase-multilingual-mpnet-base-v2`:

| measurement | value |
|---|---|
| chunks in the corpus | 1549 |
| median chunk length | 157 words, 239 subword tokens |
| 90th percentile | 456 words, 694 subword tokens |
| chunks longer than the model's 512-token input | 299 (19%) |
| chunks longer than 128 tokens, the model's training length | 1116 (72%) |
| gold spans that end past token 512 of their chunk | 15 of 167 |

The 512 cap counts words, but the model counts subword tokens, and Italian and
Spanish text runs well over one token per word. fastembed silently truncates at
512 tokens, so for one chunk in five the dense channel embeds only the head of
the text. The 15 spans past the cut cannot be found by dense retrieval at any
rank. The model was also trained on 128-token inputs, so embedding quality on
the other long chunks is likely diluted well before the hard cut.

The reranker (`jina-reranker-v2-base-multilingual`) is not affected: it reads
1024 tokens and the longest chunk is 928. Truncation is a dense-channel problem.
Hybrid recovers from it only when the sparse channel happens to pull the chunk
into the fused candidate pool.

## Pilot

A scratch sweep over the chunker's size cap, dense and sparse channels only.
Hybrid was skipped because the CPU reranker makes it an hour per configuration.
The 512 row reproduces the committed baseline exactly, which confirms the scratch
harness matches `run_wiki_eval.py`.

Span recall at k=5:

| type | 512 | 256 | 160 | 96 | 64 |
|---|---|---|---|---|---|
| **dense** buried | 0.325 | 0.475 | 0.550 | **0.650** | 0.500 |
| **dense** cross_lingual | 0.391 | 0.370 | 0.370 | **0.478** | 0.370 |
| **dense** exact_term | 0.317 | 0.439 | 0.439 | 0.463 | **0.561** |
| **dense** paraphrase | 0.300 | 0.425 | 0.600 | **0.625** | 0.575 |
| **dense overall** | 0.335 | 0.425 | 0.485 | **0.551** | 0.497 |
| **sparse** buried | **0.925** | 0.825 | 0.825 | 0.775 | 0.600 |
| **sparse** cross_lingual | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 |
| **sparse** exact_term | 0.878 | **0.902** | 0.780 | 0.683 | 0.707 |
| **sparse** paraphrase | **0.675** | 0.575 | 0.500 | 0.425 | 0.375 |
| **sparse overall** | **0.605** | 0.563 | 0.515 | 0.461 | 0.413 |
| chunks | 1549 | 2007 | 2674 | 3891 | 5439 |
| spans reachable | 167 | 164 | 158 | 155 | 144 |

Three findings:

1. **Dense gains 64% from smaller chunks**, peaking near 96 words, which is
   roughly the model's 128-token training length. Every type improves except
   cross-lingual, which is flat until 96.
2. **Sparse loses at every step.** BM25 needs enough surrounding text for a
   query's terms to co-occur in one chunk. Smaller chunks split them apart.
3. **Coverage falls as chunks shrink.** With no overlap, a span that straddles a
   cut is contained in no chunk and scores zero on every channel. That is 23 of
   167 spans at 64 words. Part of the small-chunk dense drop at 64 is this, not
   worse retrieval.

The two channels pull in opposite directions, so the production number, hybrid,
could go either way. It has to be measured, not inferred.

## Approach

Make the chunking policy configurable, add the two mechanisms the pilot shows
are missing, and run a pre-registered sweep on the full hybrid pipeline.

### 1. A `[chunking]` config section

```toml
[chunking]
max_tokens = 512        # size cap per chunk
overlap = 0             # tokens repeated from the previous chunk
unit = "words"          # "words" | "model_tokens"
```

Defaults reproduce today's behaviour exactly, so the change is inert until
someone opts in. The container passes the section to `HeadingAwareChunker`.

### 2. Overlap

A sliding window inside each section: each piece after the first repeats the
last `overlap` units of the one before. This is the standard fix for spans cut at
a boundary, and the pilot's coverage row is the direct measure of whether it
works. Overlap never crosses a heading, because the heading split is the part of
the policy that already respects document structure.

### 3. Sizing in model tokens

With `unit = "model_tokens"`, the chunker counts with the embedding model's own
tokenizer, so `max_tokens = 128` means 128 of the tokens the model actually sees,
in every language. Word counting is what let Italian and Spanish chunks overrun
the input window. This needs the chunker to receive a token-counting function,
which the container can build from the embedding adapter. The `Chunker` port does
not change.

### 4. A sweep runner

`eval/run_chunk_sweep.py` takes a list of configurations, indexes the corpus once
per configuration, and prints the per-type table for each channel plus a
coverage line. A `--channels` flag allows cheap dense-and-sparse passes before
committing an hour per configuration to hybrid. It reuses the existing index and
scoring code; the only new piece is building the container from a chunking
config.

Every configuration the runner measures is appended to the experiment log,
`eval/results/runs.jsonl`, with the 512-word baseline as its control, and the
shaded page is re-rendered. The owner compares configurations there, so a run
that is not in the log did not happen as far as the decision is concerned.

## Experiment protocol

Fixed now, before the real runs, so the result cannot be chosen after seeing it.

**Grid.** Sizes 96, 128, 160 and 256 in model tokens, each with overlap 0 and
overlap at a quarter of the size, plus today's 512-word policy as the control.
Nine configurations. Dense and sparse first on all nine, then hybrid on the
control and the three best by dense-plus-sparse overall.

**Primary metric.** Hybrid overall span recall at k=5, with span MRR as the
tie-breaker.

**Decision rule.** Adopt a configuration only if:

- hybrid overall span recall beats the control by at least 0.03, about five
  cases out of 167, which is the smallest change this set can resolve; and
- no query type loses more than 0.05 hybrid span recall against the control; and
- at least 160 of 167 spans stay reachable.

If nothing passes, today's policy stays and the config section ships anyway, so
the next experiment can vary chunking without a code change.

## Confounds

- **The gold set was filtered against today's chunking.** The discrimination
  gate dropped 29 cases every channel answered at rank 1, and the ambiguity gate
  judged competitors retrieved from the 512-word index. The committed set is
  therefore biased toward what the current policy finds hard, which flatters any
  change. The sweep also scores the 29 discriminated-out cases, recoverable from
  `gold_rejected.json`, and reports if a candidate policy loses them.
- **k counts chunks, not text.** Five 96-token chunks are less text than five
  512-word ones. That makes small chunks look worse on note-level recall and is
  the right bias for a tool whose results fill a context window, so it stays.
- **The best size depends on the embedding model.** Production defaults to
  `bge-small-en-v1.5`, not the multilingual model measured here. The adopted
  default should be stated as tuned for the multilingual model.

## Scope

In scope: the config section, overlap, token-unit sizing, the sweep runner, the
sweep itself, and changing the default if the decision rule passes.

Out of scope, each its own later cycle:

- Different granularities per channel, such as small chunks for dense and larger
  ones for sparse. The opposite trends in the pilot make this the most promising
  follow-up, but it changes the store schema.
- Small-to-big retrieval, where the index matches small chunks and returns their
  parent section.
- Contextual blurbs. Per earlier measurement, the reranker scores the raw chunk
  text and never sees the blurb, which has to be fixed before blurbs are worth
  re-measuring.
- Re-indexing a live vault. A chunking change invalidates every stored chunk, so
  the owner reindexes once after adopting a new default.

## Owner decisions

1. **Decision rule:** accepted as written.
2. **Token-unit sizing as the default:** depends on how large the improvement
   is. The sweep reports the measured gain and the owner decides; the default
   does not change without that call.
3. **Compute budget:** not answered. The plan proceeds with four hybrid runs,
   about four hours of CPU, and reports before spending more.
