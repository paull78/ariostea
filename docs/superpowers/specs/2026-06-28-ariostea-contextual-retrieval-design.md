# Contextual Retrieval (Phase 5) — Design

**Status:** Approved (design phase)
**Date:** 2026-06-28
**Roadmap:** PRD Phase 5 ("Contextual Retrieval — contextualizer + prompt caching"). This design refines the PRD §15 sketch with the decisions below.

## 1. Motivation

Today every chunk is embedded and FTS-indexed as its bare text (`index_vault.py` hard-codes `context_blurb=None, embedding_text=c.text`). A chunk taken out of its note loses the entities/topic that make it findable — "It has 88 of them" is unsearchable for "piano keys". Anthropic's **Contextual Retrieval** fixes this by prepending an LLM-written context blurb to each chunk *before* embedding and BM25 indexing, so both retrieval channels become context-aware. This phase builds that ingestion-time contextualization, fully optional and gracefully degrading to today's behavior when no LLM is configured.

## 2. Settled scope decisions

- **Provider:** one **OpenAI-compatible HTTP** chat adapter (configurable `base_url`/`model`/`api_key`) covering OpenAI, Ollama, LM Studio, vLLM, llama.cpp, etc., plus a **Noop** contextualizer for the zero-key default. No native Anthropic adapter.
- **Blurb granularity: note-level.** One LLM call per note produces a single summary blurb, prepended to every chunk of that note. (Rationale and the per-chunk alternative are in §8.)
- **Prompt caching:** none needed. Note-level means the document is sent once per note, so there is no repeated-prefix to cache. The PRD's `cache_prefix` port param is **dropped** (YAGNI; a future Anthropic adapter would own caching internally).
- **Acceptance:** deterministic behavioral tests + a reproducible demonstrated lift via a stub contextualizer + one skippable real-LLM integration test (§7).
- **Default `enabled = false`:** the single-command, zero-key install stays Noop; contextualization is opt-in.

## 3. Ports (2 new, in `src/ariostea/ports/`)

```python
# ports/chat.py
@runtime_checkable
class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...

# ports/pipeline.py (append)
@runtime_checkable
class Contextualizer(Protocol):
    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]: ...
    @property
    def fingerprint(self) -> str: ...
```

`ChatProvider.complete(system, user)` is deliberately minimal — a single system+user turn is all contextualization needs. A general message-list interface is deferred (YAGNI); if a future feature needs multi-turn, it's an additive port change.

The `Contextualizer.fingerprint` exists for re-embed correctness (§6).

## 4. Adapters (3 new)

**`adapters/chat/openai_compat.py` — `OpenAICompatChat(ChatProvider)`**
HTTP `POST {base_url}/chat/completions` via **httpx** (new dependency), body `{model, messages:[{role:"system",...},{role:"user",...}], max_tokens, temperature:0}`, `Authorization: Bearer {api_key}` when a key is set, configurable `timeout`. Returns `choices[0].message.content.strip()`. Raises a port-level error on non-2xx / network failure (the contextualizer catches it — §5).

**`adapters/contextualize/llm.py` — `LLMContextualizer(Contextualizer)`**
Holds a `ChatProvider`, a `model_name` (for the fingerprint), and `max_tokens`. `contextualize`:
1. One call: `blurb = chat.complete(system=BLURB_INSTRUCTIONS, user=full_doc)` where `BLURB_INSTRUCTIONS` asks for a 1–2 sentence (~50-token) note summary that situates the note's content for search retrieval, output as plain text only.
2. For every chunk: `embedding_text = f"{blurb}\n\n{chunk.text}"`, `context_blurb = blurb` (the same blurb for all chunks of the note).
3. `fingerprint` → `f"llm:{model_name}"`.

**`adapters/contextualize/noop.py` — `NoopContextualizer(Contextualizer)`**
Returns `[ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text) for c in chunks]`; `fingerprint` → `"noop"`.

## 5. Graceful degradation (never block ingestion)

- **Disabled or unreachable at startup:** the container wires `NoopContextualizer` (warning logged), identical to today's behavior.
- **LLM call fails mid-index:** `LLMContextualizer.contextualize` wraps the single `complete` call in try/except; on failure it returns the **Noop result for that note** (all chunks text-only, `context_blurb=None`) and logs a warning. Indexing continues; the index is never corrupted or half-contextualized within a note.

## 6. Re-embed correctness (the fingerprint point)

Contextualization changes `embedding_text`, so stored vectors and FTS rows go stale when contextualization is toggled or the chat model changes. `IndexVault` currently re-embeds when `store.fingerprint() != embeddings.fingerprint`. Change it to compare a **combined** fingerprint:

```python
combined = f"{self._embeddings.fingerprint}|{self._contextualizer.fingerprint}"
# e.g. "fastembed:...|noop"  vs  "fastembed:...|llm:gpt-4o-mini"
```

Toggling `[contextual].enabled` (Noop↔LLM) or switching chat model changes the combined fingerprint, so the existing fingerprint guard forces a full re-embed — no silent staleness. `set_fingerprint` stores the combined value.

## 7. Config & store

**`config/schema.py` — new `ContextualCfg`:**
```python
class ContextualCfg(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"   # Ollama default endpoint
    api_key: str = ""
    model: str = "llama3.1"
    timeout: float = 30.0
    max_tokens: int = 128
```
Added to `Config` as `contextual: ContextualCfg = ContextualCfg()`. Documented in `ariostea.example.toml` under `[contextual]`.

**Store:** add a `context_blurb TEXT` column to the `chunks` table (transparency/inspection; `upsert_note` writes `cc.context_blurb`). Pre-release, so existing DBs simply rebuild — no migration code.

## 8. Container & indexing wiring

- `container.py`: `_build_contextualizer(cfg.contextual) -> Contextualizer` — Noop if disabled; else construct `OpenAICompatChat` + `LLMContextualizer`, degrading to Noop + warning on construction error. Inject into `IndexVault`.
- `index_vault.py`: replace the inline `ContextualizedChunk(...)` list comprehension with `cchunks = self._contextualizer.contextualize(note, body, chunks)`; adopt the combined fingerprint (§6).
- The `Container` continues to expose only config/use-cases/ports (the contextualizer and chat provider are wiring internals, like embeddings).

## 9. Acceptance & testing

**Deterministic unit tests (fast suite):**
- `NoopContextualizer` → `embedding_text == chunk.text`, `context_blurb is None`, `fingerprint == "noop"`.
- `LLMContextualizer` with a **fake `ChatProvider`** returning a fixed blurb → `embedding_text == "<blurb>\n\n<chunk.text>"` for every chunk, `context_blurb == blurb`, `fingerprint == "llm:<model>"`.
- `LLMContextualizer` with a fake provider that raises → degrades to text-only for the whole note (no exception escapes).
- `OpenAICompatChat` against a fake transport (httpx `MockTransport`) → builds the right request body/headers and parses `choices[0].message.content`; non-2xx raises.
- `ContextualCfg` defaults + config load.
- `IndexVault` combined-fingerprint: a fake contextualizer whose fingerprint differs forces re-embed (extend existing index tests).

**Demonstrated lift (reproducible, no real LLM — `integration` marked for the embedding model):**
- `tests/eval/test_contextual_lift.py`: a self-contained two-chunk ambiguous fixture note written to `tmp_path` (e.g. `# Piano` / `## Keys` … "It has 88 of them, struck by felt hammers."). Index it twice via `IndexVault` — once with `NoopContextualizer`, once with a test-only `StubContextualizer` that prepends the note title as the blurb — then run a dense search for a context-dependent query and assert recall goes **0 → 1**. This proves the *mechanism* (prepended context improves retrieval) deterministically, using the per-channel machinery from the gold-set work. The stub is a test helper, not shipped.

**Skippable real-LLM integration test:**
- `tests/adapters/chat/test_openai_compat_integration.py` (`integration`): if an endpoint is configured via env (e.g. `ARIOSTEA_TEST_CHAT_BASE_URL`), call a real OpenAI-compatible model and assert a non-empty blurb; otherwise `pytest.skip`.

## 10. Future enhancements (recorded, not in scope)

- **Per-chunk Contextual Retrieval (Anthropic's canonical method).** Generate a *chunk-specific* blurb (full doc + that chunk → blurb situating that chunk) instead of one note-level summary. Higher fidelity on **long, multi-topic documents** where chunks are diverse, at the cost of N calls per note (mitigated by automatic prompt-prefix caching when the doc is the stable `system` prefix). Note-level was chosen because Obsidian skews toward short, single-topic notes (many single-chunk), where a note summary captures most of the benefit far more cheaply, and a shared blurb risks **homogenizing** a note's chunks in embedding space. This swap is purely an `LLMContextualizer` internal change — the `Contextualizer` port, store, wiring, and fingerprint scheme are unchanged. **Gate:** pursue only if eval on a richer, multi-chunk vault shows note-level leaving measurable recall on the table.
- **Native Anthropic chat adapter** with explicit `cache_control` prompt caching — only if per-chunk is adopted and Claude is a target provider; caching would live inside that adapter, not in the `ChatProvider` port.
- **General multi-turn `ChatProvider` message-list interface** — only if a future feature needs more than a single system+user turn.

## 11. Out of scope

- Embeddings over HTTP (this phase adds only the *chat* HTTP path; embeddings stay local fastembed).
- Per-chunk blurbs, Anthropic adapter, multi-turn chat (all §10).
- Re-contextualizing only changed notes beyond what the existing content-hash + combined-fingerprint guard already provides.
