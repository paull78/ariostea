# Tuning retrieval: chunking

How Ariostea's chunking policy was measured and replaced, from building an evaluation set that
could detect a difference to the sweeps that picked the new default. Every number here comes from
`eval/results/runs.jsonl`, and every figure is a view of the experiment log page rendered from
it.

## The instrument

Before this work, the eval corpus was 19 short documents, one chunk each. Chunking could not be
measured on it at all: there was nothing to chunk differently.

The replacement is a snapshot of 79 Wikipedia articles in 6 topic clusters (board games, cheese,
coffee, cycling, sailing, string instruments), pinned to revision IDs so it rebuilds byte for
byte. Articles in a cluster share vocabulary, so every query has near misses to reject.

On top of it sits a gold set of 167 queries. Each points at the **text span** that answers it,
not just the note, and a retrieved chunk counts as a hit only when it contains that span whole.
The same labels stay valid under any chunking policy, which is what makes chunking measurable.

Queries come in four types, one per retrieval mechanism:

| type | stresses | cases |
|---|---|---|
| paraphrase | dense embeddings | 40 |
| exact_term | BM25 full-text search | 41 |
| buried | facts deep inside a long section | 40 |
| cross_lingual | Italian and Spanish queries over English articles | 46 |

A local model (`qwen2.5-14b-instruct`) generated candidates from 505 passages. A different
local model (`qwen3.6-35b-a3b`) judged them. They passed five gates: automatic span checks, the
adversarial judge, a filter dropping queries every channel already answers at rank 1, an
ambiguity gate rejecting queries another passage answers just as well, and a human spot review
of 19 sampled cases, 17 of which were sound. 338 candidates were rejected, each recorded with
its reason in `eval/wiki/gold_rejected.json`.

## Baseline

The first reading, with the original policy: split at every heading, then cut sections longer
than 512 words into consecutive pieces with no overlap.

| channel | span recall@5 | span MRR |
|---|---|---|
| dense | 0.335 | 0.209 |
| sparse (BM25) | 0.605 | 0.502 |
| hybrid (fused, reranked) | 0.832 | 0.789 |

Hybrid is what the MCP server serves. Dense at 0.335 stood out as far too low for a
multilingual embedding model on this material.

## Why chunking first

Measuring what the embedding model actually reads explained the dense number.

| measurement | value |
|---|---|
| chunks in the corpus | 1549 |
| median chunk | 157 words, 239 model tokens |
| chunks longer than the model's 512-token input | 299 (19%) |
| chunks longer than 128 tokens, the model's training length | 1116 (72%) |
| gold spans ending past token 512 of their chunk | 15 of 167 |

The size cap counted words, while `paraphrase-multilingual-mpnet-base-v2` counts subword
tokens, and Italian and Spanish run well over one token per word. fastembed truncates silently
at 512 tokens, so for 1 chunk in 5 the dense channel embedded only the start of the text. 15
answers sat past the cut, unreachable by dense retrieval at any rank.

The reranker (`jina-reranker-v2-base-multilingual`) reads 1024 tokens, and the longest chunk is
928, so hybrid's final ranking always saw whole chunks. Truncation hurt only the dense channel's
contribution to the candidate pool.

## Pilot: smaller chunks

A scratch sweep over the size cap, dense and sparse only, since hybrid costs about an hour per
configuration on CPU.

![Chunk size pilot](images/chunking-pilot.png)

Dense improved at every step down to 96 words, from 0.335 to 0.551. Sparse got worse at every
step, from 0.605 to 0.461: BM25 needs enough surrounding text for a query's terms to land in the
same chunk. And coverage fell. With no overlap, an answer that straddles a cut is contained in
no chunk and scores zero everywhere; at 64 words that was 23 of 167 answers.

The two channels pulled in opposite directions, so hybrid could go either way. It had to be
measured.

## Making chunking configurable

The pilot needed code changes before a real sweep could run.

- A `[chunking]` section in the config: `max_tokens`, `overlap`, and `unit` (`"words"` or
  `"model_tokens"`). The defaults reproduce the original chunker byte for byte, and a test pins
  that against the whole corpus.
- Overlap: each piece repeats the tail of the previous one, never across a heading.
- Sizing in model tokens, counted with the embedding model's own tokenizer, so a cap of 128
  means the same thing in every language.
- A chunker fingerprint in the index. Before this, the index fingerprint covered the embedding
  model and the contextualiser but not the chunker, so changing the policy would have left
  every unchanged note with its old chunks and no warning.

## The experiment log

Every run is appended to `eval/results/runs.jsonl` with its configuration, commit, coverage,
and every metric per channel and query type, compared against a named control run.
`eval/render_results.py` turns it into a page where each cell is shaded by its change against
the control: green for a gain, magenta for a loss, grey within ±0.03. That band is the smallest
change 167 queries can resolve, about 5 cases. Every cell also prints its signed change, so the
colour never carries the information alone.

## Decision rule

Fixed before any hybrid run, so the result could not be picked after seeing it. A policy
replaces the default only if:

- hybrid span recall beats the control by at least 0.03;
- no query type loses more than 0.05; and
- at least 160 of 167 answers stay reachable.

## Sweep, dense and sparse

Nine policies: the control, and sizes of 96, 128, 160 and 256 model tokens, each with no
overlap and with a quarter overlap.

![Dense and sparse sweep](images/chunking-sweep-dense-sparse.png)

Overlap mattered most. Every policy with overlap kept all or nearly all answers reachable;
without it, 3 of the 4 token sizes fell below the 160 floor. And every overlapping policy beat
the control on the mean of dense and sparse, gaining more on dense than it lost on sparse.

The sweep also scored the 29 cases the gold set had dropped for being too easy. The gold set
was filtered against the original chunking, so it leans toward what that chunking finds hard,
which flatters any change. Small chunks without overlap lost up to 11 of those 29 on dense.
With overlap, the 160 and 256-token policies kept nearly all of them.

The three best eligible policies, 256, 128 and 160 tokens with overlap, were within 0.006 of
each other, too close to separate without hybrid.

## Sweep, hybrid

![Hybrid sweep](images/chunking-sweep-hybrid.png)

| policy | hybrid span recall | change | worst type | result |
|---|---|---|---|---|
| 160 tokens, overlap 40 | 0.880 | +0.048 | exact_term +0.024 | passes |
| 128 tokens, overlap 32 | 0.844 | +0.012 | exact_term −0.025 | fails |
| 256 tokens, overlap 64 | 0.838 | +0.006 | cross_lingual −0.021 | fails |

160 tokens with 40 of overlap improved every type, including cross-lingual, the weakest track
(0.630 to 0.696). Span MRR rose from 0.789 to 0.810, and it kept all 29 easy cases.

The large dense gains mostly did not carry through to hybrid. The reranker was already
recovering many of those answers from the larger chunks, so hybrid moved by a few cases where
dense moved by 30.

## Confirmation

The 160-token result was the best of three tries, and its neighbours had barely moved. So one
more policy ran between them before any decision: 192 tokens with 48 of overlap.

| policy | hybrid span recall | change | answers gained |
|---|---|---|---|
| 128 tokens, overlap 32 | 0.844 | +0.012 | 2 |
| 160 tokens, overlap 40 | 0.880 | +0.048 | 8 |
| 192 tokens, overlap 48 | 0.850 | +0.018 | 3 |
| 256 tokens, overlap 64 | 0.838 | +0.006 | 1 |

192 landed with the neighbours, not with 160. The likeliest reading is that sizing in model
tokens with overlap gives hybrid a small, steady gain of about +0.01 to +0.02, below the
decision rule's threshold, and that part of 160's +0.048 is luck.

Some things hold regardless. None of the four overlapping policies lowered hybrid overall. All
of them lifted dense span recall by 0.18 to 0.23 and kept all 167 answers reachable. None lost
any of the 29 easy cases on hybrid.

That left the default undecided: a small, possibly lucky hybrid gain against a forced reindex
of every vault. The next measurement settled it.

## Without the reranker

The server reranks by default, but it silently falls back to fused order when reranking is
disabled in the config or its model fails to load. The hybrid runs could not show what the
chunking change does in that case, so a FUSED channel measured it: the same dense and sparse
retrieval and the same RRF fusion, with the reranker switched off.

![Fused sweep](images/chunking-fused.png)

| policy | fused span recall | change | worst type | span MRR change |
|---|---|---|---|---|
| 512 words (control) | 0.551 | | | |
| 128 tokens, overlap 32 | 0.617 | +0.066 | paraphrase −0.075 | +0.103 |
| 160 tokens, overlap 40 | 0.629 | +0.078 | buried +0.025 | +0.117 |
| 192 tokens, overlap 48 | 0.641 | +0.090 | paraphrase 0.000 | +0.125 |
| 256 tokens, overlap 64 | 0.611 | +0.060 | paraphrase −0.050 | +0.116 |

Here the effect is large and steady. Every policy gains at least +0.06, 2 to 3 times the
rule's threshold, and span MRR rises by more than 0.10 in all four. Cross-lingual gains most,
from 0.152 to as much as 0.304. Under the decision rule 160 and 192 tokens pass cleanly, 256
tokens passes with paraphrase exactly at the −0.05 limit, and 128 tokens fails on paraphrase.

It also explains the hybrid result. Without the reranker the 512-word default scores 0.551;
with it, 0.832. The reranker was absorbing most of the damage the word cap did, so fixing the
cap showed up in hybrid as only a few cases.

## Decision

The default is now 160 model tokens with 40 of overlap. It is the only policy that passed the
decision rule both with the reranker (+0.048, every type up) and without it (+0.078, every type
up). The hybrid gain is probably smaller than +0.048, as the confirmation run suggests, but no
overlapping policy made hybrid worse, and the fallback path gains a lot.

The change is framed as a fix: the 512-word cap handed the embedding model inputs it silently
truncated. Upgrading re-chunks and re-embeds an existing vault once, because the chunking
policy is now part of the index fingerprint. The original policy stays available as
`max_tokens = 512`, `overlap = 0`, `unit = "words"`.

## Reproducing

```bash
uv run python eval/run_wiki_eval.py                        # the baseline tables
uv run python eval/run_chunk_sweep.py 512w 160t+40 --channels DENSE,SPARSE
uv run python eval/run_chunk_sweep.py 160t+40              # adds hybrid, about an hour
uv run python eval/run_chunk_sweep.py 160t+40 --channels FUSED --control <512w fused run id>
uv run python eval/render_results.py                       # rebuild the log page
```

The figures are the log page opened with URL parameters that pin a view, then captured at 2x.
For example:

```
experiment_log.html?view=table&channels=HYBRID&experiments=Baseline|Chunk%20sweep:%20full
```

`view=table` hides everything but the scale and the table. `channels` (comma-separated) and
`experiments` (separated by `|`, since names can contain commas) filter the columns and row
groups, and `caption` adds a heading.
