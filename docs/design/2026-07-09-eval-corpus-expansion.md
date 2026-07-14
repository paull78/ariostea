# Design: A discriminating eval corpus

**Status:** approved, ready for implementation planning
**Date:** 2026-07-09

## Problem

The current retrieval eval is too small and too easy to support meaningful
experiments. Concretely:

- **19 documents, ~92 lines total** — each document is only a few lines, so it produces a
  single chunk. Chunking-policy experiments are therefore impossible: there is nothing to
  chunk differently.
- **Gold labels are note-level and binary** — `query → [expected_note.md]`. This can score
  "did we retrieve the right note," but not *which passage* won, which is what chunking and
  contextual-blurb experiments depend on.
- **Documents are topically disjoint** (pasta, chess, astronomy…). With one plausible target
  per query, every retrieval method scores near-perfect and improvements are invisible. The
  discriminating power of an eval comes from the *near-misses* a method must reject.

We want an eval instrument on which experiments in **chunking policy, contextual blurbs,
BM25/FTS, and (later) graph retrieval** produce measurable, attributable differences.

## Approach

A hybrid corpus (previously "approach C"): **Wikipedia articles** as the document corpus,
**LLM-generated queries and labels** for most tracks, with structure preserved so a
graph-retrieval track can be added later without rebuilding.

Wikipedia is a strong fit: articles have real headings and length (chunking), dense
inter-article links that map onto Obsidian wikilinks (graph retrieval), and parallel
articles across languages (cross-lingual). It is licensed CC BY-SA 4.0 — free to use with
attribution and share-alike.

### Core design principles

1. **The corpus unit is a topic cluster, not an article.** Each cluster is 10–15
   closely-related, mutually-linked articles that share vocabulary and subtopics (e.g.
   *string instruments*: violin, viola, cello, guitar, mandolin). The near-misses inside a
   cluster are the hard negatives that make experiments discriminate. Density within a
   cluster matters more than breadth across topics.

2. **Relevance is span-anchored, at dual granularity.** Gold labels point at the
   answer-bearing **text span** in the source document, plus the containing note. A retrieved
   chunk counts as a hit if it *contains* the span. This survives re-chunking, so the same
   gold is valid across every chunking policy — the exact variable under test never
   invalidates the labels.

3. **LLM-generated labels are always validated.** Generated queries and spans pass automatic
   checks, an adversarial second-model check, a discrimination filter, and human spot-review
   before entering the gold set.

## Scope (this phase)

**Medium tier**, chosen for enough distractor density and per-track sample size without an
unreviewable labeling burden:

| Dimension | Target |
| --------- | ------ |
| Clusters | 5–6 |
| Articles | ~60–80 |
| Queries | ~150 |
| Languages | English throughout; it/es parallels on 1–2 clusters |

**Graph retrieval is deferred.** The corpus is built graph-ready (densely-linked clusters,
wikilinks preserved), but only single-hop gold is generated now. Multi-hop queries are added
when the graph-retrieval feature is actually built, to avoid labeling against a design that
does not yet exist.

## Components

### 1. Corpus acquisition and conversion

An offline script fetches articles via the Wikipedia API, **pinned to specific revision IDs**
so the corpus is frozen and reproducible. Conversion to Obsidian-flavored Markdown:

- Preserve headings and section structure.
- Rewrite links to *other articles in the same corpus* as `[[wikilinks]]`; links to articles
  outside the corpus become plain text.
- Strip infoboxes, reference lists, and navigation chrome.
- Frontmatter records source URL, revision ID, and license.

Output: `eval/wiki/<cluster>/<article>.md`, kept separate from the existing flat
`eval/corpus/*.md` smoke fixtures. `eval/wiki/NOTICE` gives CC BY-SA attribution per article
(see "Attribution" below). The corpus data is CC BY-SA; repository code remains MIT.

**Attribution.** `eval/wiki/NOTICE` is authored before any Wikipedia text is committed. It
states the CC BY-SA 4.0 license, records that articles were modified (converted to Markdown,
links rewritten to wikilinks, infoboxes/references stripped), and carries a per-article table
of title + revision-permalink. The corpus build script appends one row per fetched article,
so attribution is maintained mechanically as the snapshot is pinned. The README notes that
`eval/wiki/` is third-party CC BY-SA data distinct from the MIT-licensed code.

### 2. Gold generation pipeline

An offline script (`eval/generate_gold.py`) using the existing `OpenAICompatChat` port. For
selected passages, the model emits a query anchored to a specific answer span, deliberately
covering query *types* that stress each retrieval track:

- `paraphrase` — semantically restates the passage (favors dense retrieval).
- `exact_term` — hinges on a rare literal token (favors BM25/FTS).
- `buried` — targets a fact buried in a long document (favors contextual blurbs).
- `cross_lingual` — query in it/es whose answer span is in the English article.

**Validation gate** (all must pass):

1. **Automatic:** the answer span must literally exist in the cited note; reject queries
   answerable from the article title alone.
2. **Adversarial second-model check:** a different prompt confirms the span answers the query
   and the query is unambiguous.
3. **Discrimination filter:** drop queries that *every* channel already answers at rank 1 —
   they carry no experimental signal. Keep the hard ones.
4. **Human spot-review** of a sample (feasible at Medium size).

Generated gold is committed to the repository, so evaluation is deterministic and needs no
network or LLM at eval time.

### 3. Gold schema

Extends the current format with `type` and span-level labels:

```json
{
  "query": "how is a violin tuned",
  "query_lang": "en",
  "type": "buried",
  "scenario": "buried",
  "expected_notes": ["string-instruments/violin.md"],
  "answer_spans": [
    {
      "note": "string-instruments/violin.md",
      "text": "The violin is tuned in perfect fifths: G, D, A, E."
    }
  ]
}
```

Committed as `eval/wiki/gold.json`. The existing gold sets are retained as a fast smoke test.

### 4. Harness upgrade

- **Span-containment metric:** a retrieved chunk is a hit if it contains any `answer_span`
  (whitespace-normalized comparison). Valid across chunking policies.
- Report recall@k / MRR / nDCG at **both** note-level (`expected_notes`) and span-level
  (`answer_spans`).
- **Per-`type` breakdown** so each experiment reads its own signal (a BM25 change surfaces in
  `exact_term`; a blurb change in `buried`).
- **Difficulty guard:** report a dense-only baseline per cluster; flag clusters scoring
  ≥ ~0.95 as too easy.

## Testing

- Corpus conversion: wikilink rewriting limited to in-corpus articles; heading/structure
  preserved; frontmatter/license present.
- Gold generation: schema validity; every answer span exists verbatim in its cited note;
  validation gate rejects known-bad fixtures.
- Harness: span-containment hit logic; per-type aggregation; note-level vs span-level metrics.

## What this unlocks

This corpus is the instrument for the improvement roadmap. The experiments it enables
sequence naturally:

1. **Chunking policy** — size, overlap, heading-awareness.
2. **Contextual blurbs** — re-measured against real buried facts.
3. **BM25 / FTS tuning** — analyzers, query construction, weighting.
4. **Graph retrieval** (later) — the corpus is already wikilink-ready.

Each becomes its own design → plan → implementation cycle, measured on this eval.

## Out of scope

- Multi-hop / graph-retrieval gold (deferred; structure preserved).
- Changes to production retrieval code — this phase builds evaluation capability only.
- Fetching Wikipedia at eval time — the corpus is a committed, pinned snapshot.
