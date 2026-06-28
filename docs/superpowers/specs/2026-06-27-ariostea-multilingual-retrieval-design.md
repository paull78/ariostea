# Multilingual Retrieval Improvements — Design

> **Status:** approved design (brainstormed 2026-06-27). Supersedes the scattered
> multilingual notes in the PRD §19 by turning them into committed, ordered roadmap items.
> Parent spec: [`2026-06-12-ariostea-rag-design.md`](2026-06-12-ariostea-rag-design.md).

## 1. Goal

Improve retrieval for two multilingual scenarios on a mixed-language (initially EN/IT) vault:

1. **Cross-lingual query↔note** — ask in one language, find notes written in another.
2. **Fair ranking across languages** — a strong cross-lingual match must not be buried
   beneath mediocre same-language matches.

Every improvement is **measured**, not assumed: a cross-lingual evaluation harness lands
first and gates each subsequent change.

## 2. The problem (grounded)

What we already have and have observed (PRD §19, RRF memory):

- **Multilingual dense embeddings** (`paraphrase-multilingual-mpnet-base-v2`) already find
  cross-lingual matches *semantically*. This half works today.
- **BM25 sparse is structurally monolingual** — blind to a different-language match.
- **RRF rewards cross-channel agreement.** A chunk that ranks #1 in dense but is absent
  from sparse (the typical cross-lingual case) scores *below* a mediocre chunk present in
  both channels. Observed live: an Italian "dadi" note buried under an English query for
  dice. This is simultaneously the cross-lingual failure *and* the ranking-fairness failure
  — same root cause.

The fix is to stop treating fusion as the final ranker, and to add a ranking stage that
judges true query↔passage relevance regardless of which channel surfaced the candidate —
**provided that ranking stage is itself multilingual.**

## 3. Chosen approach — "Rerank-first, measure, then decide"

Ordered, with each step gated on the eval harness:

1. **Eval harness** (measurement) — *first*.
2. **Multilingual reranking** — the Phase 6 reranker, with the model-is-multilingual
   constraint made explicit. Likely subsumes most of the problem.
3. **Re-measure.** Only if a cross-lingual gap remains:
   - optional cheap interim: `WeightedFuser` (dense tilt);
   - conditional backlog: BGE-M3 learned multilingual sparse.

Rationale: reranking is the recorded corrective and is already on the roadmap; building the
heavier learned-sparse machinery speculatively violates YAGNI. The eval tells us whether we
even need it.

## 4. Components

### 4.1 Cross-lingual evaluation harness (new — Component 1)

**What it does.** Runs a fixed set of queries through the real retrieval pipeline and reports
quality metrics, broken down by language direction, so any change can be compared
before/after.

**Pieces:**

- **Committed bilingual fixture vault** — `eval/corpus/` with a handful of short notes on
  shared topics, some in English, some in Italian (e.g. a note about board games in EN and a
  `giochi da tavolo` note in IT). Committed so the eval is deterministic and shareable, and
  so it can gate CI without depending on anyone's personal vault.
- **Gold set** — `eval/gold.yaml`: a list of cases, each
  `{query, query_lang, expected: [note_path, ...], direction}` where `direction ∈
  {en→it, it→en, same}`. ~15–25 cases, including same-language controls so we can see we
  don't regress monolingual quality while fixing cross-lingual.
- **Scorer** — `eval/run_eval.py`: builds a container against the fixture vault, indexes it,
  runs each gold query through `SearchKnowledge`, computes **recall@k** (is an expected note
  among the top-k results' notes) and **MRR** (reciprocal rank of the first expected note),
  and prints a table aggregated overall and **per direction**. Exit non-zero if a configured
  threshold regresses (so it can gate).
- **Optional personal gold set** — `eval/gold.local.yaml` (gitignored) pointing at the real
  vault, for higher-signal spot checks.

**Architecture fit.** The harness is a *driving adapter* — a consumer of the use cases, the
same role as `cli.py` and the MCP server. It calls `SearchKnowledge` through the public
`Container` API and introduces **no new ports**. It depends inward only.

**Metrics math (unit-testable in isolation):**

- `recall@k = 1 if expected_notes ∩ top_k_notes else 0`, averaged over cases.
- `MRR = mean(1 / rank_of_first_expected)`, `0` if no expected note retrieved.

### 4.2 Multilingual reranking (Component 2 — makes PRD Phase 6 concrete)

**What it does.** After hybrid retrieval + fusion produce a large candidate pool, a
cross-encoder re-scores each candidate against the query and returns the top-k by true
relevance.

- **Port** — `Reranker` (already specced in the PRD as `ports/rerank.py`):
  `rerank(query: str, candidates: Sequence[RetrievedChunk], top_n: int) -> list[RetrievedChunk]`.
- **Default adapter** — `FastEmbedReranker` wrapping a **multilingual** cross-encoder,
  default model **`BAAI/bge-reranker-v2-m3`** (ONNX via fastembed, no API key). This is the
  single most important decision in this design: an English-only reranker (e.g. an
  ms-marco MiniLM) would score the cross-lingual passage low and *reintroduce* the exact
  failure we are fixing.
- **`NoopReranker`** — identity passthrough, for opt-out and as the deterministic test double.
- **Pipeline change.** `RRFFuser` is demoted to a *recall gatherer*: raise its pool size
  (e.g. fuse top ~100–150) and let the reranker pick the final `top_k`. Flow becomes:

  ```
  embed_query → dense() + sparse() → RRF.fuse(large pool) → Reranker.rerank(→ top_k)
  ```

- **Config** — new `[rerank]` section: `enabled` (default true), `model`, `pool` (candidates
  fed to the reranker), and the existing `top_k` as the final count.
- **Degradation (LSP).** If the reranker model is unavailable, the search falls back to the
  fused order (effectively `NoopReranker`) with a logged warning — a degraded ranking, never
  a failed search.
- **Acceptance gate.** The eval harness must show measurable recall@k / MRR improvement on
  the `en→it` and `it→en` directions versus the no-rerank baseline, with **no regression** on
  `same`.

> **Result (measured 2026-06-28, `jina-reranker-v2-base-multilingual`, fixture vault).**
> Partial, honest win — no regression:
>
> | direction | recall@1 (base→rerank) | MRR@5 (base→rerank) |
> |-----------|------------------------|---------------------|
> | `it→en`   | 0.000 → **0.500**      | 0.500 → **0.750**   |
> | `en→it`   | 0.000 → 0.000          | 0.500 → 0.500       |
> | `same`    | 0.750 → 0.750 (held)   | 0.875 → 0.875 (held)|
> | overall   | 0.375 → **0.500**      | 0.688 → **0.750**   |
>
> Reranking improved `it→en` and overall and regressed nothing. `en→it` stayed flat — but
> that is largely a **fixture confound**: each `en→it` case declares only the Italian note
> correct while the corpus also holds an equally-relevant *English* twin on the same topic, so
> for an English query the reranker ranking the English note first is defensible. Clean
> cross-lingual measurement needs target topics that exist *only* in the other language.
> **Follow-up:** expand the gold set with single-language-only topics (and more cases per
> direction — 2 is noisy) before judging `en→it`. Default model is Jina (the design's named
> alternative) because `bge-reranker-v2-m3` is absent from the installed fastembed.

### 4.3 `WeightedFuser` (Component 3 — optional interim, YAGNI)

A `Fuser` adapter variant that tilts fusion toward the dense channel (or softens the
single-channel penalty), giving a cheap cross-lingual lift **before** the reranker lands.
Build this *only* if a quick win is wanted in the interim; reranking is expected to subsume
it. Not a destination. Own contract + unit tests if built.

### 4.4 BGE-M3 learned multilingual sparse (Component 4 — conditional backlog)

Replace/augment BM25 with **BGE-M3's learned sparse mode**, making the *sparse channel
itself* cross-lingual so cross-channel agreement happens before fusion. This is the
root-cause fix for the sparse side, but it is a bigger lift (heavier model, larger index) and
**overlaps with reranking** — once a multilingual reranker is the final judge, pre-fusion
sparse quality matters far less. **Trigger condition:** pursue only if the post-rerank eval
*still* shows a cross-lingual gap. Fits as a swappable retriever adapter; no use-case change.

> Explicitly *not* pursued: **query-translation CLIR** (PRD §19 option a). Word-sense
> ambiguity (EN "dice" → IT "dadi" the game vs. "dice" = "he says") plus a translator
> dependency fight the local-first/zero-key ethos. Learned sparse is the cleaner path if the
> sparse side ever needs fixing.

## 5. Data flow (target, after Component 2)

```
Query
  → EmbeddingProvider.embed_query        (multilingual dense)
  → ChunkRetriever.dense()  ┐
  → ChunkRetriever.sparse() ┘ → Fuser.fuse(large pool)   (recall gatherer)
  → Reranker.rerank(top_k)               (multilingual; final ranker)
  → results   (→ SearchSources rollup unchanged)
```

## 6. Error handling

- **Reranker unavailable** → fall back to fused order, log a warning; search still returns.
- **Eval harness** is a dev/CI instrument; its failures never affect the runtime package.
- Fixture vault is committed and tiny, so eval indexing is fast and deterministic.

## 7. Testing

- **Scorer math** — unit tests for `recall@k` and `MRR` against synthetic ranked lists
  (no models needed); these run in the fast suite.
- **End-to-end eval** — `run_eval.py` against the fixture vault; loads real models, so it is
  `@pytest.mark.integration` / manual, not in the fast suite.
- **Reranker** — contract test against the `Reranker` port (run for every adapter incl.
  `NoopReranker`); a `SearchKnowledge` unit test with a fake reranker that reverses order,
  asserting the use case applies the reranker and truncates to `top_k`.
- **WeightedFuser / BGE-M3** — their own contract + unit tests when/if built.

## 8. Roadmap deltas (the deliverable)

To apply to the PRD roadmap (§17) and §19:

1. **New item, before Phase 6 — "Cross-lingual eval harness."**
   *Deliverable:* committed bilingual fixture vault + gold set + recall@k/MRR scorer with
   per-direction breakdown. *Acceptance:* `run_eval.py` reports a baseline; scorer math unit-
   tested. (Also satisfies the pre-existing "improves on eval set" acceptance language in
   Phases 5 and 6.)
2. **Amend Phase 6 (Reranking).** Default reranker MUST be multilingual
   (`bge-reranker-v2-m3`); RRF becomes a recall gatherer feeding the reranker. *Acceptance
   (added):* measurable recall@k/MRR gain on `en→it` and `it→en`, no regression on `same`.
3. **Optional interim — `WeightedFuser`** (YAGNI; build only for a quick pre-rerank win).
4. **Phase 8 backlog — BGE-M3 learned multilingual sparse**, conditional on a remaining
   post-rerank cross-lingual gap. Query-translation CLIR explicitly rejected.
5. **Update §19** to reference this design as the resolution of the RRF single-channel /
   cross-lingual finding.

## 9. Out of scope

- CJK / no-space-script tokenization (the `trigram` FTS knob) — stays in Phase 8 backlog;
  the user's priority is EN/IT (Latin, space-separated).
- Monolingual same-language quality beyond not regressing it.
