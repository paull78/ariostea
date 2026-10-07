# Design: Blurb-aware reranking, measured

**Status:** approved 2026-09-29, ready for implementation planning
**Date:** 2026-09-29

## Problem

Contextual indexing asks an LLM for one short blurb per note (topic and key
entities, about 50 words) and prepends it to every chunk of that note. The
combined `embedding_text` is what gets embedded and what goes into the FTS
index. The blurb is also stored alone in `chunks.context_blurb`.

The reranker never sees it. `FastEmbedReranker` scores `rc.chunk.text`, the raw
chunk, and `RetrievedChunk` has no field that could carry the blurb anyway: the
store's `dense()` and `sparse()` queries do not select it. Measured on the small
contextual corpus in 2026-07, blurbs lifted sparse (buried recall@5 0.20 to
0.80) and dense (buried MRR 0.45 to 1.00), but hybrid stayed flat. The reranker
decides the final order and judges text without the context the first stages
used.

Contextual indexing has also never been measured on the wiki gold set. Wiki
articles are long (about 3,800 words on average, 16,000 at most), so at the
current chunking default of 160 model tokens one article yields dozens of
chunks that all share one blurb. The blurb can help find the right article but
cannot tell its chunks apart, and it pulls all their vectors toward the same
topic. Blurbs may be flat or negative here even on the raw channels.

## Goal

Two questions, answered separately:

1. Do note-level blurbs help retrieval on the wiki gold set?
2. Given blurbs, does letting the reranker score blurb plus chunk help hybrid?

## Change: carry the blurb to the reranker

1. **Domain.** `RetrievedChunk` gains `context_blurb: str | None = None`. The
   default keeps every existing constructor call valid.
2. **Store.** `SqliteStore.dense()` and `sparse()` select `c.context_blurb`
   alongside the columns they already read and set it on each
   `RetrievedChunk`. No new query or join.
3. **Fusion.** `RRFFuser` builds fresh `RetrievedChunk`s from an internal
   `_Entry` per chunk, so as written it would drop the new field silently.
   `_Entry` gains `context_blurb`, taken from the first result that creates
   the entry (dense and sparse read the same row, so they agree), and the
   output carries it.
4. **Reranker.** `FastEmbedReranker(model_name, use_context=False)`. With
   `use_context` on and a non-empty blurb, the scored passage is
   `f"{blurb}\n\n{chunk.text}"`, byte-for-byte the `embedding_text` format
   `LLMContextualizer` builds, so every stage judges the same text. Otherwise
   it scores `chunk.text` as today. The returned `RetrievedChunk`s are the
   inputs with a new score; `chunk.text` is never modified, so search results
   still show raw text.
5. **Config.** `RerankCfg.use_context: bool = False`, passed by
   `_build_reranker`. `NoopReranker` ignores it.

No port changes and no reindex: the column already exists and the fingerprint
does not cover reranking. With contextual indexing off every blurb is `None`,
so the flag has no effect.

**Storage and memory.** The blurb is text, written once per note and copied
into every chunk row of that note at index time. This design only reads that
copy; it adds no storage. The duplication is real, though, and worth sizing.
On the wiki corpus at the current default there are 4,246 chunks from 79 notes
(up to 224 per note). At about 350 bytes per blurb, the `context_blurb` column
holds about 1.5 MB where one copy per note would take 28 KB. The FTS index
holds the blurb a second time inside each chunk's indexed text. For scale,
each chunk also stores a 3 KB vector (768 floats) and about 540 bytes of text,
so the blurb copies are roughly 10 to 15% of the database. Moving the column
to `notes` would save the 1.5 MB, but it needs a schema migration and a join
in every search, and it cannot remove the FTS copy: BM25 has to see the
blurb's words in each chunk's document. Not worth it at this scale; revisit
if vaults grow by orders of magnitude. At query time the cost is nothing
measurable: the store reads one string per candidate (100 at most), and
fusion and the reranker pass references to it, never copies.

The passage stays well inside the reranker's 1024-token input: chunks are
capped at 160 embedding-model tokens and the blurb adds about 70.

**Default.** `use_context` ships off. It becomes the default only if arm 3
beats arm 2 under the decision rule below, and only with the owner's approval,
as with chunking.

## Experiment: three arms

All arms use the production chunking default (160 model tokens, overlap 40),
the multilingual embedding model, k=5, the committed gold set, and also score
the 29 discriminated-out cases.

| arm | blurbs | reranker scores | source |
|---|---|---|---|
| 1 | off | raw text | already logged: `2026-09-24-chunk-160t+40-dense-hybrid-sparse` (DENSE 0.545, SPARSE 0.533, HYBRID 0.880), `2026-09-28-chunk-160t+40-fused` (FUSED 0.629) |
| 2 | on | raw text | new |
| 3 | on | blurb + text | new, HYBRID only |

Arms 2 and 3 share one index. DENSE, SPARSE and FUSED do not involve the
reranker, so they are measured once, under arm 2.

**Reproducibility check first.** Before building the blurb index, a
`--no-log` DENSE,SPARSE pass of 160t+40 on the current code must reproduce arm
1's logged 0.545 and 0.533. If it does not, something changed since
`2026-09-24` and arm 1 must be re-measured before any comparison.

### Blurb generation

Production behaviour, unchanged: the whole note goes to the LLM. For the eval:

- Local model over LM Studio's OpenAI-compatible endpoint, the one used for the
  2026-07 measurement (`qwen2.5-14b-instruct-mlx`) unless the owner picks
  another. Loaded with a context window of at least 32k tokens so the longest
  article fits. The 20 GB judge must not be resident (`lms unload --all`
  first).
- `ContextualCfg.timeout` raised for the run (default 300 s) so long articles
  are not cut off by the 30 s production default.
- **Coverage gate.** After indexing, every one of the 79 notes must have a
  non-empty blurb, checked with the existing `find_uncontextualized_notes`.
  Any miss aborts the run before scoring. A partly blurbed index would read as
  "blurbs don't help".

Blurb generation is the expensive, flaky step (79 long prompts, perhaps 20 to
40 minutes). The blurbed index is therefore written to a kept path under
`eval/results/indexes/` (gitignored), and the runner can reuse it with
`--reuse-index` instead of regenerating. Each HYBRID pass is about an hour on
CPU, so a full run is about three hours.

### Runner

A new `eval/run_blurb_eval.py`, separate from the chunk sweep: the sweep builds
one throwaway index per chunking spec, and this experiment needs one kept index
scored two ways. It reuses the shared pieces (`wiki_config`,
`index_wiki_corpus`, `wiki_channels`, `fused_channel`, `evaluate_spans`,
`results_log`). The LLM endpoint comes from the same `ARIOSTEA_CTX_*`
environment variables as `run_contextual_eval.py`. Arm 3's HYBRID channel is a
second container over the same database, built with
`RerankCfg(use_context=True)` and no reindex.

The runner logs three entries to `eval/results/runs.jsonl`, each with the
control it must be compared against:

| entry | channels | control |
|---|---|---|
| arm 2 | DENSE, SPARSE, HYBRID | `2026-09-24-chunk-160t+40-dense-hybrid-sparse` |
| arm 2, fused | FUSED | `2026-09-28-chunk-160t+40-fused` |
| arm 3 | HYBRID | `2026-09-24-chunk-160t+40-dense-hybrid-sparse` |

The run config records the chat model, its timeout and the blurb coverage. The
log page is re-rendered and the artifact republished afterwards.

## Decision rule

The chunking rule, set before running: at least +0.03 overall span recall, no
query type below -0.05.

- **Blurbs help** if arm 2 or arm 3 passes against arm 1 on HYBRID. FUSED is
  checked too, because hybrid masked a real defect in the chunking work.
- **`use_context` helps** if arm 3 passes against arm 2 on HYBRID.

Arm 1 hybrid is already 0.880, so a pass needs 0.910 or better. Little
headroom is left; a flat hybrid result is plausible and is a finding, not a
failure. DENSE and SPARSE deltas are reported per query type either way, since
they show whether note-level blurbs blur chunks within long articles.

## Testing

- Store: `dense()` and `sparse()` return the stored blurb, and `None` for a
  plain index.
- Fusion: RRF output keeps `context_blurb` for chunks from either channel.
- Reranker: with `use_context` on, the passage handed to the cross-encoder is
  blurb, blank line, text; with it off, or with no blurb, it is raw text. The
  returned chunk text is unchanged. Tested with a fake cross-encoder, no model
  download.
- Config and container: `use_context` defaults to false and reaches the
  reranker.
- Runner: pure helpers (run entries, coverage abort) unit-tested without a live
  model; the full run is manual.

## Out of scope

- Blurbs per chunk. If note-level blurbs come out flat or negative on long
  articles, that is the next experiment.
- Truncating the note sent to the LLM.
- Showing the blurb in search results.
- Regenerating the gold set.
