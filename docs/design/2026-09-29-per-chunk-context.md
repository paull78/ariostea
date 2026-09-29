# Design: Per-chunk context, measured

**Status:** approved 2026-09-29, ready for implementation planning
**Date:** 2026-09-29

## Problem

Note-level blurbs made wiki retrieval worse (hybrid 0.880 to 0.844, and 0.814
when the reranker saw them; `docs/retrieval-tuning.md`, "Contextual blurbs").
One blurb shared by dozens of chunks of a long article pulls those chunks
together, so the right chunk is harder to pick out of its article.

Anthropic's Contextual Retrieval writes context for each chunk instead: the
prompt holds the whole document and one chunk and asks for a short text that
situates that chunk in the document. Every chunk gets different context, which
targets exactly the failure above.

## Pilot

Local `qwen2.5-14b-instruct-mlx` over LM Studio, article first and chunk last
so consecutive calls share the article as a prefix (memory
`per-chunk-context-pilot`):

| article | tokens | first chunk | later chunks |
|---|---|---|---|
| rennet | 2.1k | 24 s | 5-10 s |
| racing-bicycle | 3.6k | 31 s | 5-12 s |
| caffe-espresso-it | 3.7k | 34 s | 6-25 s |
| double-bass, `--parallel 4` | 22.7k | 270 s | timed out at 600 s |
| double-bass, `--parallel 1` | 22.7k | 570 s | 12-20 s |

LM Studio reuses the cached article only with one parallel slot. A full run of
the 4,246 wiki chunks is about 15 hours.

## Change: a per-chunk contextualizer

- `ContextualCfg.granularity: Literal["note", "chunk"] = "note"`. The default
  keeps today's behaviour.
- New adapter `LLMChunkContextualizer(chat, model_name)` implementing the
  existing `Contextualizer` port, selected by `_build_contextualizer` when
  `granularity == "chunk"`.
  - System message: the instructions and `<document>{full_doc}</document>`,
    byte-identical for every chunk of a note, so the article is a reusable
    prefix.
  - User message: `<chunk>{text}</chunk>` and Anthropic's instruction ("give a
    short succinct context to situate this chunk within the overall document
    for the purposes of improving search retrieval of the chunk; answer only
    with the succinct context"), plus "write it in the same language as the
    chunk".
  - Chunks are processed in order, one call each.
  - `embedding_text = contextual_text(context, chunk.text)`; the context is
    stored in `context_blurb`. The reranker's `use_context` works unchanged.
  - A failed or empty call degrades that chunk alone to plain text, with a
    warning; the note's other chunks keep their context.
  - Fingerprint `llm-chunk:{model}`, distinct from note-level `llm:{model}`.
- `build_container(config, *, wrap_chat=None)`: an optional function applied to
  the contextualizer's chat provider. Production passes nothing; the eval
  passes a `CachingChat` wrapper so contexts persist to disk as they arrive.
  The Container still exposes no adapters.

## Experiment

`eval/run_blurb_eval.py --granularity chunk`:

- Wraps the contextualizer's chat in `CachingChat` at
  `eval/results/indexes/chunk-context-cache.jsonl` (gitignored). A crash or a
  rerun reuses every context already generated.
- Defaults for chunk mode: timeout 900 s (the first long-article call took
  570 s), `max_tokens` 200 (the pilot saw up to 135).
- A separate kept index, `chunk-context-160t+40.db`.
- Coverage gate per chunk: all 4,246 chunks must have a context (the existing
  check already flags any note with a chunk that lacks one).
- The contexts dump is one entry per chunk,
  `{note_path: {ordinal: context}}`.
- `--reuse-index` checks for `llm-chunk:{model}` in chunk mode.
- `--preview NOTE[,NOTE]`: contextualize only those notes through the cache and
  print the contexts, without indexing or scoring. Used for the language check;
  the full run then reuses those cached answers.
- Same three arms and controls as the note-level run, logged as experiment
  "Per-chunk context".

Procedure: `lms load qwen2.5-14b-instruct-mlx --context-length 32768
--parallel 1`, preview one Italian and one Spanish article and check the
language, then run detached overnight (about 15 hours of contexts, then about
2 hours of hybrid scoring).

## Decision rule

Unchanged: at least +0.03 overall span recall, no query type below -0.05,
against the no-blurb controls. The write-up also compares with the note-level
arms.

## Out of scope

Making per-chunk indexing practical for real vaults (about 10 s per chunk on
this machine). Worth pursuing only if the measurement shows a gain.
