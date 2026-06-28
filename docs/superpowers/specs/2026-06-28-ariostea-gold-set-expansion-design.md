# Gold-Set Expansion + Per-Channel Eval — Design

**Status:** Approved (design phase)
**Date:** 2026-06-28
**Depends on:** the eval harness (`src/ariostea/eval/`), the FTS diacritic fix (`unicode61 remove_diacritics 2` + Unicode-aware query regex, merged 2026-06-28).

## 1. Motivation

The eval harness exists but its fixture is too small and too entangled to answer the questions we now care about:

1. **Cross-lingual is inconclusive.** Every topic in the corpus is bilingual (pasta×2, dice×2). A cross-lingual case like "cooking spaghetti al dente" → `pasta_it.md` is confounded: `pasta_en.md` is equally relevant *and* same-language, so we can't tell whether a cross-lingual miss is a real failure or just the model reasonably preferring the same-language twin.
2. **No channel attribution.** The runner only measures the blended pipeline (dense + sparse + RRF + rerank). We just shipped an accent fix on the **sparse** channel but cannot prove it in isolation — dense could mask a sparse miss (or vice versa).
3. **No stemming evidence.** The Phase 8 "multilingual FTS stemming" backlog item is gated on eval evidence that doesn't exist yet.

This work expands the corpus/gold set and adds per-channel measurement so the harness can answer all three.

## 2. Scope decisions (settled)

- **Languages:** en / it / es (the existing en/it plus Spanish). Not de/pt for now.
- **Per-channel eval:** yes — measure dense, sparse, and hybrid separately.
- **Inflection cases:** included now, to settle the stemming backlog.

## 3. Corpus design

**Guiding principle: each *new* topic exists in exactly one language.** A cross-lingual query then has exactly one relevant note (in the other language) with no same-language sibling to compete — making the measurement conclusive.

Keep the 4 existing bilingual notes (`pasta_en`, `pasta_it`, `dice_en`, `dadi_it`) for **same-language baseline only** (they also act as English/Italian same-language distractors for the cross-lingual cases). Add **5 single-language notes**:

| File | Lang | Topic | Doubles as |
|------|------|-------|-----------|
| `astronomia_it.md` | it | stargazing / telescope | **accent** target (`città`) |
| `cucito_it.md` | it | sewing | **inflection** target (note contains `bottone`) |
| `beekeeping_en.md` | en | beekeeping | cross-lingual target (it→en, es→en) |
| `ciclismo_es.md` | es | cycling | **accent** target (`montaña`) |
| `alfareria_es.md` | es | pottery | **inflection** target (note contains `vasija`) |

One single-language topic per language (astronomia=it, beekeeping=en, ciclismo=es) is enough to cover all six ordered cross-lingual pairs as conclusive targets; the two inflection notes (it, es) cover the morphology test. English needs no inflection note — English morphology is too weak to demonstrate the stemming gap, so inflection cases are it/es only.

**Content constraints (enforced when writing the notes):**

- **Distinct topics.** No new topic may overlap another language's topic strongly enough to recreate a twin. (Stargazing, sewing, beekeeping, woodworking, cycling, pottery are mutually distinct.)
- **Accent term present and distinctive.** `città` appears in `astronomia_it.md`; `montaña` appears in `ciclismo_es.md`; each is reasonably unique to its note so the sparse keyword match is unambiguous.
- **Inflection notes contain ONLY the base form.** `cucito_it.md` contains `bottone` but never `bottoni`; `alfareria_es.md` contains `vasija` but never `vasijas`. Otherwise the sparse channel would match the queried plural directly and the inflection test would be dead.

## 4. Gold set (~17 cases)

The `scenario` field (see §6) groups cases. Planned distribution:

- **same** (7): the 4 existing + `astronomia` (it), `beekeeping` (en), `ciclismo` (es).
- **cross-lingual, conclusive** (6): all six ordered pairs among en/it/es, each targeting a single-language topic:
  - `en→it` stars/telescope → `astronomia_it`
  - `es→it` stars/telescope → `astronomia_it`
  - `it→en` beekeeping → `beekeeping_en`
  - `es→en` beekeeping → `beekeeping_en`
  - `en→es` cycling → `ciclismo_es`
  - `it→es` cycling → `ciclismo_es`
- **accent** (2): `città` → `astronomia_it`; `montaña` → `ciclismo_es`.
- **inflection** (2): `bottoni` → `cucito_it`; `vasijas` → `alfareria_es`.

Each gold case keeps the single-correct-note assumption (one expected note), under which recall@k and MRR hold as defined.

## 5. Per-channel eval

`evaluate(cases, search_fn, k)` already takes a `search_fn`, so per-channel measurement is **just calling it three times** — no change to `evaluate`/metrics. The **runner** builds three `SearchFn`s:

- **dense:** `vec = embeddings.embed_query(q)` → `store.dense(vec=vec, k=POOL)` → dedupe note paths → top-k.
- **sparse:** `store.sparse(query=q, k=POOL)` → dedupe note paths → top-k.
- **hybrid:** existing `search_payload(container, ...)` — the full production pipeline, kept faithful.

**Accessing raw channels without leaking adapters.** The production `Container` deliberately exposes only config, use cases, and the `IndexAdmin` port (the `ChunkRetriever`/`EmbeddingProvider` are wiring internals). To respect that boundary, the runner opens a **second, read-only handle** — its own `FastEmbedEmbeddings` + `SqliteStore` pointed at the same indexed DB path — purely for dense/sparse access. Production wiring is untouched. (Cost: the embedding model is constructed twice in the eval process; acceptable for an occasional script.)

The runner runs `evaluate` once per channel and prints three labeled reports (DENSE / SPARSE / HYBRID), each with the per-scenario breakdown.

The two new search-fn factories are unit-tested with a fake retriever + fake embeddings (assert they call `dense`/`sparse` correctly and dedupe to note paths). The end-to-end run remains an integration check, not a unit test.

## 6. Rename: `direction` → `scenario`

The grouping field no longer means only "language direction" — it now also carries `accent` and `inflection`, which are same-language test purposes. Rename for honesty:

| Before | After |
|--------|-------|
| `GoldCase.direction` | `GoldCase.scenario` |
| `DirectionScore` (field `.direction`) | `ScenarioScore` (field `.scenario`) |
| `EvalReport.by_direction` | `EvalReport.by_scenario` |
| `gold.json` key `"direction"` | `"scenario"` |
| `format_report` header `direction` | `scenario` |

`scenario` is a free-form label; the harness still just buckets by its string value and reports overall + per-scenario. A case carries exactly one scenario (accent/inflection cases are same-language by construction, so no second axis is lost). If a future need to cross-tab scenario × language ever appears, promoting it to an independent `category` field is a clean isolated refactor at that point — YAGNI until then.

## 7. Expected outcome

A 3-channel × scenario matrix. Hypotheses to confirm:

- **accent** cases hit on **sparse** → proves the diacritic fix end-to-end (regression guard going forward).
- **inflection** cases **miss on sparse** but **hit on dense** → confirms dense already absorbs morphology; records the evidence that keeps multilingual FTS stemming as YAGNI.
- **cross-lingual** scenarios are now interpretable because each target is conclusive.

**Measured results** (2026-06-28, k=5, `paraphrase-multilingual-mpnet-base-v2`, 17 gold cases):

```
=== DENSE ===
scenario       n  recall@5    mrr
accent         2     1.000  1.000
en→es          1     1.000  1.000
en→it          1     1.000  1.000
es→en          1     1.000  1.000
es→it          1     1.000  1.000
inflection     2     1.000  1.000
it→en          1     1.000  1.000
it→es          1     1.000  1.000
same           7     1.000  0.929
overall       17     1.000  0.971

=== SPARSE ===
scenario       n  recall@5    mrr
accent         2     1.000  1.000
en→es          1     0.000  0.000
en→it          1     0.000  0.000
es→en          1     0.000  0.000
es→it          1     1.000  1.000
inflection     2     0.000  0.000
it→en          1     0.000  0.000
it→es          1     1.000  1.000
same           7     1.000  1.000
overall       17     0.647  0.647

=== HYBRID ===
scenario       n  recall@5    mrr
accent         2     1.000  1.000
en→es          1     1.000  1.000
en→it          1     1.000  1.000
es→en          1     1.000  1.000
es→it          1     1.000  1.000
inflection     2     1.000  1.000
it→en          1     1.000  1.000
it→es          1     1.000  1.000
same           7     1.000  0.929
overall       17     1.000  0.971
```

Sparse `accent` recall@5 = 1.000 confirms the FTS diacritic fix is verified end-to-end. Sparse `inflection` recall@5 = 0.000 confirms FTS has no stemming, as expected; dense `inflection` recall@5 = 1.000 shows multilingual embeddings already recover inflected forms — this is the evidence that keeps multilingual FTS stemming as a YAGNI backlog item. The sparse `es→it` and `it→es` hits (1.000) are a fixture quirk: Spanish and Italian share enough cognate tokens (e.g. "telescopio", "montaña"/"montagna") that BM25 fires across those two language pairs — this is not a general sparse cross-lingual capability. Hybrid overall recall@5 = 1.000: dense fully compensates for sparse's blind spots; the `same` MRR of 0.929 reflects one same-language note ranking second rather than first.

## 8. Out of scope

- German / Portuguese corpus (en/it/es only for now).
- Any stemming implementation (this only gathers the evidence for the decision).
- A separate orthogonal `category` field (reusing renamed `scenario` instead).
- Changes to production `Container` exposure (eval wires its own raw-channel handle).
