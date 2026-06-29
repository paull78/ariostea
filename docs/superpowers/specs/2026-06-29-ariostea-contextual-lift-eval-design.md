# Contextual-Lift Eval — Design

**Status:** Approved (design phase)
**Date:** 2026-06-29
**Roadmap:** Closes the open thread left by PRD Phase 5 (Contextual Retrieval, merged 2026-06-28). Phase 5 shipped with the lift proven only *mechanically* (`tests/indexing/test_contextual_lift.py`, recall 0→1 via a stub on the sparse channel) and **never measured on a gold set**, because the existing eval corpus is single-sentence, single-chunk notes where contextualization cannot help. This eval measures the real lift.

## 1. Goal

Answer one question with real numbers: **does enabling Contextual Retrieval improve retrieval quality, and by how much?** A real-LLM, one-off measurement — index a context-dependent corpus twice (contextualization OFF vs ON against a local Ollama) and report the per-channel recall@k / MRR delta.

This is deliberately **not** a committed CI gate. A non-deterministic real-LLM result pinned to a pass/fail threshold would be flaky; the deterministic regression guard already exists (`test_contextual_lift.py`). This is a measurement tool, run manually.

## 2. Settled decisions

- **Mode:** real-LLM one-off against a local Ollama endpoint. Non-deterministic, manual.
- **Corpus:** a new, dedicated, **English-only** context-dependent fixture vault — isolates the variable under test (only contextualization changes; no cross-lingual confound).
- **Structure:** a standalone runner reusing the existing harness; the cross-lingual `run_eval.py` is left functionally unchanged (one small shared-helper extraction, §3).
- **Channels:** dense, sparse, hybrid — the same three the cross-lingual eval measures.

## 3. Architecture & files

All new, except one DRY extraction:

- **`eval/contextual_corpus/`** — new English-only, context-dependent fixture vault (§4).
- **`eval/contextual_gold.json`** — gold queries (same schema as `eval/gold.json`: `query`, `query_lang`, `expected`, `scenario`).
- **`eval/run_contextual_eval.py`** — the standalone runner (§5).
- **`src/ariostea/eval/channels.py`** — extract the `make_hybrid_search_fn` helper currently inlined in `eval/run_eval.py` into here so both runners share it. `run_eval.py` is updated to import it (its behavior is unchanged).

Reused with no behavior change:
- `src/ariostea/eval/harness.py` — `evaluate`, `format_report`, `load_gold`, `dedupe`, and the `EvalReport` / `ScenarioScore` types (no edits).
- `src/ariostea/eval/channels.py` — existing `make_dense_search_fn` and `make_sparse_search_fn` are unchanged; the file gains the extracted `make_hybrid_search_fn` (above).

## 4. Corpus & gold design (the crux)

The harness scores **note-level** recall@k / MRR (chunk hits deduped to note paths). For contextualization to move that number, a target note must be *hard to retrieve from its bare chunks but easy once a topic blurb is prepended*. Three ingredients:

### 4.1 Target notes (~5)
Short, single-topic, multi-section notes (multiple `#` headings, so `HeadingAwareChunker` yields ≥2 chunks) where one section describes a distinctive fact using **anaphora** — pronouns / "the instrument" / "it" — and **never names the topic**.

Example `eval/contextual_corpus/piano.md`:
```markdown
# Daily Routine
Spend twenty minutes on scales before anything else each morning.

# The Instrument
It has 88 of them, black and white, struck by felt hammers when a key is pressed.

# Maintenance
Have a technician tune it twice a year to keep the pitch true.
```
The "The Instrument" section never says "piano".

### 4.2 Distractor notes (~5)
Adjacent topics sharing surface vocabulary (other instruments; unrelated hobbies that share words like "press", "string", "tune") so the candidate pool is genuinely contested. Without distractors every query trivially scores recall@5 = 1.0 and no lift is observable.

### 4.3 Two gold scenarios
- **`buried`** — the query names the topic ("how many keys does a piano have"); the only matching chunk is the anaphoric one that does *not* name it. **Lift expected here.** OFF: the blurb-less chunk has no topic words → distractors outrank it, and the sparse channel misses entirely (the topic token is literally absent). ON: the note-level blurb injects "piano" into every chunk's `embedding_text` and FTS text → the chunk surfaces.
- **`direct`** (control) — the query targets a section that already names its own topic. **≈ no lift expected.** This confirms the eval measures a real, localized effect rather than globally inflating every score.

### 4.4 Honest expectation
Note-level blurbs on short notes may show only a **modest** lift, concentrated on the **sparse** channel (where the topic word is wholly absent OFF; the dense channel already absorbs some implicit similarity). A small or sparse-only lift is a valid, informative outcome — it is exactly the hypothesis recorded in the [contextual-retrieval design §10](2026-06-28-ariostea-contextual-retrieval-design.md) (note-level chosen over per-chunk for short Obsidian notes). The eval's job is to produce the number, not to guarantee a large one.

## 5. Runner behavior

`uv run python eval/run_contextual_eval.py [k]` (k defaults to 5).

1. **Config from environment** (point it at a running Ollama without editing code):
   - `ARIOSTEA_CTX_BASE_URL` — default `http://localhost:11434/v1`
   - `ARIOSTEA_CTX_MODEL` — default `llama3.1`
   - `ARIOSTEA_CTX_API_KEY` — default `""`
2. **Index twice** into separate throwaway temp DBs, both using the multilingual embedding model `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (same as `run_eval.py`):
   - **OFF:** `ContextualCfg(enabled=False)`
   - **ON:** `ContextualCfg(enabled=True, base_url=…, model=…, api_key=…)` from env.
3. **Loud integrity guard.** `LLMContextualizer` silently degrades to Noop when the endpoint is unreachable — which would make ON == OFF and quietly invalidate the measurement. After the ON index, assert at least one stored chunk has a non-null `context_blurb`; if every blurb is null, **abort** with a clear error: `contextualization produced no blurbs — is Ollama running at <base_url>?`. This guard is what protects the result from being a false null.
4. **Evaluate** all three channels (dense / sparse / hybrid) over both the OFF and ON indexes via the existing `evaluate`, reading each throwaway DB through a second `SqliteStore` handle (same pattern as `run_eval.py`, keeping the production `Container` ports-only).
5. **Output:** print the OFF per-channel report, the ON per-channel report, and a compact **delta** view.

### 5.1 `format_delta(off: EvalReport, on: EvalReport) -> str`
A pure presentation helper (lives in the runner script). Given two reports for the same channel and gold set, render one row per scenario (plus overall) showing recall@k and MRR as `OFF → ON (Δ±x.xxx)`. It pairs scenarios by `ScenarioScore.scenario`; both reports are produced from the same gold file so their scenario sets match.

## 6. Testing

The real-LLM `main()` is run manually — not a CI gate (per §1). The **pure pieces are TDD'd**:

- **`format_delta`** — deterministic over two hand-built `EvalReport`s → asserts the Δ math and that scenarios are paired correctly.
- **The blurb-presence guard** — a fake store/reader whose chunks all have `context_blurb=None` makes the guard raise; one non-null blurb makes it pass. (The guard is extracted as a small pure function over the store's chunk rows so it is testable without a live model.)
- **Corpus/gold integrity** (fast) — every `expected` path in `contextual_gold.json` exists in `eval/contextual_corpus/`, and each `buried`-scenario target note chunks into ≥2 chunks via `HeadingAwareChunker` (so the "buried section" premise actually holds).

No threshold assertion on the lift itself (flaky against a live model).

## 7. Out of scope

- Committing measured numbers as a regression gate (the deterministic `test_contextual_lift.py` already guards the mechanism).
- Cross-lingual contextual lift (English-only here to isolate the variable; a future eval could add `en+it` once the single-variable number is known).
- Per-chunk contextualization (still a recorded Phase 5 future enhancement; this eval measures the shipped note-level behavior).
- Any change to production code under `src/ariostea/` beyond the `make_hybrid_search_fn` extraction into `channels.py`.

## 8. Acceptance

- Running the script against a live Ollama prints OFF, ON, and delta tables per channel without error, and the integrity guard aborts cleanly when no endpoint is reachable.
- `format_delta`, the blurb-presence guard, and the corpus/gold integrity checks pass in the fast suite.
- The measured lift (whatever its size) is recorded back into the contextual-retrieval design as evidence.
