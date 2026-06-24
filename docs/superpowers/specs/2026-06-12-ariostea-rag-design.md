# Ariostea — Obsidian RAG MCP Server

**PRD & Architecture Specification**
Status: Draft for review · Date: 2026-06-12 · Owner: paolo

---

## 1. Summary

Ariostea is a **local-first, Obsidian-aware Retrieval-Augmented Generation system** exposed as a **Model Context Protocol (MCP) server**, installable and runnable with a single command (`uvx ariostea`). It indexes any Obsidian vault incrementally and lets an LLM agent (or the user) search the vault two ways:

1. **Knowledge search** — semantic + keyword retrieval of the most relevant passages, with their source notes.
2. **Source search** — provenance rollup answering "this concept appears in notes X, Y, Z," with hit counts and snippets.

Retrieval quality follows Anthropic's **Contextual Retrieval** method (contextual embeddings + contextual BM25 + reranking). Every external capability — embeddings, the contextualization LLM, the reranker, the vector store — sits behind a narrow adapter interface. The **OpenAI-compatible HTTP API is the universal default** for embeddings and chat, so the provider is chosen by swapping a base URL (OpenAI, Voyage, Ollama, LM Studio, vLLM, llama.cpp, TEI). A **bundled local stack** (fastembed + sqlite-vec) guarantees the single command works with zero API keys, fully offline.

### 1.1 Goals

- One command to install and run as an MCP server for a personal Obsidian vault.
- Incremental indexing: only changed files are re-processed; live updates via a file watcher.
- Top-tier retrieval: hybrid dense+sparse, contextual chunks, reranking.
- Strict modularity: db, embedding, LLM, reranker, and algorithm stages are all independently swappable.
- Provider transparency: OpenAI-compatible endpoints everywhere a standard exists; local fallback otherwise.
- Privacy by default: nothing leaves the machine unless the user configures a cloud endpoint.

### 1.2 Non-goals (initial release)

- No multi-user / hosted SaaS deployment; this is a single-user local server.
- No editing/writing back to the vault — read-only indexing.
- No dedicated entity/concept extraction index (deferred; provenance rollup covers "source search" for v1).
- No GUI — MCP tools + CLI only.

### 1.3 Naming

`ariostea` (Ludovico Ariosto, *Orlando Furioso* — a vast, cross-referenced work). Python package `ariostea`; CLI `ariostea`; PyPI/uvx target `ariostea`.

---

## 2. Users & primary use cases

**Actor: the user**, via an MCP-capable client (Claude Desktop, Cline, etc.), asking an agent questions answerable from their vault.

| # | Use case | Tool |
|---|----------|------|
| U1 | "What did I write about X?" → relevant passages + sources | `search_knowledge` |
| U2 | "Where does concept X appear?" → list of notes with snippets | `search_sources` |
| U3 | Retrieve a full note by path | `get_note` |
| U4 | Force a (re)index of the vault or specific paths | `reindex` |
| U5 | Inspect index health / config / counts | `status` |
| U6 | Keep the index fresh automatically while editing | watcher (no tool call) |

---

## 3. Architectural principles (Clean Architecture)

Ariostea is organized as concentric layers; **source-code dependencies point strictly inward**. No inner-layer name appears in an outer layer.

```
        ┌───────────────────────────────────────────────────────┐
        │ Frameworks & Drivers (outermost)                      │
        │  fastembed · sqlite-vec/FTS5 · httpx · watchfiles ·   │
        │  MCP SDK · typer/click                                 │
        │   ┌───────────────────────────────────────────────┐   │
        │   │ Interface Adapters                            │   │
        │   │  adapters/{embedding,chat,rerank,store} ·     │   │
        │   │  mcp/ · cli.py · config/ (composition root)   │   │
        │   │   ┌───────────────────────────────────────┐   │   │
        │   │   │ Use Cases (application policy)        │   │   │
        │   │   │  indexing/  ·  search/                │   │   │
        │   │   │   ┌───────────────────────────────┐   │   │   │
        │   │   │   │ Entities (domain)             │   │   │   │
        │   │   │   │  Note, Chunk, SearchResult,   │   │   │   │
        │   │   │   │  SourceHit, Query             │   │   │   │
        │   │   │   └───────────────────────────────┘   │   │   │
        │   │   └───────────────────────────────────────┘   │   │
        │   └───────────────────────────────────────────────┘   │
        └───────────────────────────────────────────────────────┘

  Dependencies point inward ▲.  ports/ (Protocols) live at the use-case
  boundary; outer layers implement them, inner layers depend on them.
```

### 3.1 The dependency rule, made concrete

- `domain/` imports **nothing** from the project (only stdlib / dataclasses / pydantic types).
- `indexing/` and `search/` (use cases) import from `domain/` and `ports/` only. They never import `fastembed`, `sqlite3`, `httpx`, `watchfiles`, or the MCP SDK.
- `adapters/*` implement `ports/` Protocols and may import frameworks freely.
- `mcp/` and `cli.py` are **delivery mechanisms** — thin translators that invoke use cases. Use cases are unaware they exist.
- `config/` is the **composition root**: the only place that imports both ports and concrete adapters, instantiates them from config, and injects them. (Permitted "main" component — it knows everything and is depended on by nothing.)

### 3.2 SOLID application

- **SRP (one reason to change per module / one actor):** the chunker changes only when chunking strategy changes; `SqliteStore` only when persistence changes; `mcp/` only when the tool contract changes. Each port has a single responsibility.
- **OCP:** new providers/algorithms = new adapter classes registered in the composition root. No existing use-case or adapter code is modified to add a Qdrant store or a Cohere reranker.
- **LSP:** every adapter is fully substitutable for its port; e.g. `NoopReranker` and `FastEmbedReranker` are interchangeable wherever a `Reranker` is expected. Contract tests (one suite per port) run against every adapter to enforce this.
- **ISP:** ports are narrow and role-specific. The fat "vector store" is split (see §6) so the **search** use case depends only on read roles and the **indexing** use case only on the write role.
- **DIP:** high-level policy (use cases) and low-level detail (adapters) both depend on the `ports/` abstractions; the abstractions own the interface, not the implementations.

### 3.3 Component principles

- **Cohesion by reason-to-change**, not by technical kind: `indexing/` groups everything that changes when the ingestion pipeline changes; `search/` everything that changes when retrieval changes.
- **Stable-dependencies:** `domain/` and `ports/` are the most stable (most depended-on, most abstract); frameworks are the most volatile (depend on nothing internal). Dependencies run from volatile → stable.
- **Acyclic:** layer graph is a DAG; the composition root is a sink that depends on all, depended on by none.

---

## 4. Module structure

```
ariostea/
  domain/                 # Entities — pure, dependency-free
    models.py             #   Note, Chunk, ContextualizedChunk, Query,
                          #   RetrievedChunk, SearchResult, SourceHit, IndexStats
  ports/                  # Protocol interfaces (the swappable boundaries)
    embedding.py          #   EmbeddingProvider
    chat.py               #   ChatProvider
    rerank.py             #   Reranker
    store.py              #   DocumentWriter, ChunkRetriever, SourceRollup, IndexAdmin
    pipeline.py           #   MarkdownParser, Chunker, Contextualizer, Fuser
  adapters/
    embedding/
      openai_compat.py    #   OpenAICompatEmbeddings  (default)
      fastembed_local.py  #   FastEmbedEmbeddings     (bundled offline fallback)
    chat/
      openai_compat.py    #   OpenAICompatChat        (default)
      ollama.py           #   OllamaChat
      noop.py             #   NoopContextualizer
    rerank/
      fastembed.py        #   FastEmbedReranker       (bundled default)
      cohere.py voyage.py jina.py tei.py noop.py
    store/
      sqlite/             #   SqliteStore (sqlite-vec + FTS5)  (default)
      qdrant.py lancedb.py pgvector.py
    parse/
      obsidian.py         #   ObsidianMarkdownParser  (frontmatter, headings, tags, wikilinks)
    chunk/
      heading_aware.py    #   HeadingAwareChunker     (default)
    fuse/
      rrf.py weighted.py
  indexing/               # Use case: IndexVault, IndexPaths (orchestrate stages via ports)
  search/                 # Use case: SearchKnowledge, SearchSources (retrieve→fuse→rerank→rollup)
  mcp/                    # Delivery: MCP server + tool schemas → call use cases
  watch/                  # Debounced file watcher → IndexPaths
  config/                 # Config schema + composition root (wiring)
  cli.py                  # ariostea init | serve | reindex | status
```

---

## 5. Domain model (`domain/models.py`)

Pure value objects. No persistence, no framework types.

```python
@dataclass(frozen=True)
class Note:
    path: str                      # vault-relative
    title: str
    frontmatter: dict[str, Any]
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]     # outgoing [[links]]
    content_hash: str              # sha256 of raw bytes
    mtime: float

@dataclass(frozen=True)
class Chunk:
    note_path: str
    ordinal: int
    heading_path: tuple[str, ...]  # e.g. ("Projects", "Ariostea")
    text: str
    token_count: int

@dataclass(frozen=True)
class ContextualizedChunk:
    chunk: Chunk
    context_blurb: str | None      # None when contextualization disabled/unavailable
    embedding_text: str            # blurb + "\n\n" + text  (or just text)

@dataclass(frozen=True)
class Query:
    text: str
    k: int = 10
    filters: QueryFilters | None = None   # tags, frontmatter, path globs

@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None

@dataclass(frozen=True)
class SearchResult:               # output of search_knowledge
    chunks: tuple[RetrievedChunk, ...]

@dataclass(frozen=True)
class SourceHit:                  # one note in a provenance rollup
    note_path: str
    title: str
    hit_count: int
    best_score: float
    snippets: tuple[str, ...]

@dataclass(frozen=True)
class IndexStats:
    notes: int; chunks: int; last_indexed: float; config_fingerprint: str
```

---

## 6. Ports (`ports/`)

Narrow, role-specific Protocols. Each is the abstraction both sides depend on.

```python
# embedding.py
class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def fingerprint(self) -> str: ...        # provider+model id; triggers reindex on change

# chat.py
class ChatProvider(Protocol):
    def complete(self, *, system: str, user: str,
                 cache_prefix: str | None = None) -> str: ...   # cache_prefix → prompt caching

# rerank.py
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[RetrievedChunk],
               top_n: int) -> list[RetrievedChunk]: ...

# store.py  — ISP: split by role so use cases depend only on what they use
class DocumentWriter(Protocol):                 # used by indexing
    def upsert_note(self, note: Note, chunks: Sequence[ContextualizedChunk],
                    embeddings: Sequence[list[float]]) -> None: ...
    def delete_note(self, path: str) -> None: ...

class ChunkRetriever(Protocol):                 # used by search
    def dense(self, vec: list[float], k: int, filters) -> list[RetrievedChunk]: ...
    def sparse(self, query: str, k: int, filters) -> list[RetrievedChunk]: ...

class SourceRollup(Protocol):                   # used by search_sources
    def rollup(self, chunk_ids: Sequence[str]) -> list[SourceHit]: ...

class IndexAdmin(Protocol):                     # used by indexing/status
    def known_hashes(self) -> dict[str, str]:   # path -> content_hash
        ...
    def stats(self) -> IndexStats: ...
    def fingerprint(self) -> str: ...

# pipeline.py
class MarkdownParser(Protocol):
    def parse(self, path: str, raw: str) -> Note: ...

class Chunker(Protocol):
    def chunk(self, note: Note, body: str) -> list[Chunk]: ...

class Contextualizer(Protocol):
    def contextualize(self, note: Note, full_doc: str,
                      chunks: Sequence[Chunk]) -> list[ContextualizedChunk]: ...

class Fuser(Protocol):
    def fuse(self, dense: list[RetrievedChunk],
             sparse: list[RetrievedChunk], k: int) -> list[RetrievedChunk]: ...
```

`SqliteStore` implements `DocumentWriter`, `ChunkRetriever`, `SourceRollup`, and `IndexAdmin` in one class, but callers receive only the role they need.

---

## 7. Adapters

| Port | Default | Alternatives |
|------|---------|--------------|
| EmbeddingProvider | `OpenAICompatEmbeddings` (uses `base_url`/`api_key`/`model`); **bundled offline fallback** `FastEmbedEmbeddings` (ONNX, e.g. `bge-small-en-v1.5`) | any OpenAI-compatible endpoint (OpenAI, Voyage, Ollama, LM Studio, vLLM, TEI) |
| ChatProvider (contextualizer LLM) | `OpenAICompatChat`; → `OllamaChat` if present; → `NoopContextualizer` if neither | any OpenAI-compatible chat endpoint; Anthropic via its own adapter (optional) |
| Reranker | `FastEmbedReranker` (bundled local cross-encoder, ONNX) | `CohereReranker`, `VoyageReranker`, `JinaReranker`, `TeiReranker`, `NoopReranker` |
| Vector store | `SqliteStore` (sqlite-vec + FTS5, single file) | `QdrantStore`, `LanceDBStore`, `PgVectorStore` |
| MarkdownParser | `ObsidianMarkdownParser` | generic markdown parser |
| Chunker | `HeadingAwareChunker` | fixed-size, semantic |
| Fuser | `RRFFuser` (Reciprocal Rank Fusion) | `WeightedFuser` |

**Reranking note:** there is no OpenAI-standard `/v1/rerank`, so the reranker is its own adapter family; the bundled local cross-encoder is the default so quality reranking needs no key.

---

## 8. Persistence schema (`adapters/store/sqlite`)

The schema is an **implementation detail of the store adapter** — it does not appear in `domain/` or use cases.

| Table | Purpose |
|-------|---------|
| `notes`(id, path, title, frontmatter_json, content_hash, mtime) | one row per note |
| `chunks`(id, note_id, ordinal, heading_path, text, context_blurb, token_count) | one row per chunk |
| `chunks_vec` (sqlite-vec virtual table: chunk_id, embedding[dim]) | dense KNN |
| `chunks_fts` (FTS5: chunk_id, embedding_text) | sparse BM25 |
| `links`(src_note_id, dst_title) | wikilink graph + backlinks |
| `tags`(note_id, tag) | tag filters |
| `meta`(key, value) | index version, **config fingerprint** (embedding model + chunker + dim) |

Hybrid retrieval, metadata filtering, and the source rollup are all single SQL queries over this one file. Incremental upserts and deletes are transactional. A change in `config fingerprint` (e.g. different embedding model → different dimension) triggers a guarded full reindex.

---

## 9. Indexing use case (`indexing/`)

`IndexVault` / `IndexPaths` orchestrate stages **through ports**:

```
scan(paths)                         # walk vault, apply ignore rules
  → for each file:
      raw = read(); hash = sha256(raw)
      if hash == IndexAdmin.known_hashes()[path]: skip   # incremental
      note = MarkdownParser.parse(path, raw)
      chunks = Chunker.chunk(note, body)                 # heading-aware + overlap
      ctx = Contextualizer.contextualize(note, raw, chunks)   # optional blurbs, doc prompt-cached
      vecs = EmbeddingProvider.embed_documents([c.embedding_text for c in ctx])
      DocumentWriter.upsert_note(note, ctx, vecs)        # atomic vec+fts+graph write
  → for each deleted path: DocumentWriter.delete_note(path)
```

- **Incremental by construction:** content-hash + mtime gate skips unchanged files; deletions detected by diffing known paths vs scanned paths.
- **Contextualization** degrades gracefully (Noop when no LLM), never blocking ingestion.
- No framework types cross into this layer; fully unit-testable with fake ports.

---

## 10. Search use case (`search/`)

`SearchKnowledge` and `SearchSources` share one retrieval core:

```
vec = EmbeddingProvider.embed_query(query.text)
dense  = ChunkRetriever.dense(vec, k=K_DENSE, filters)
sparse = ChunkRetriever.sparse(query.text, k=K_SPARSE, filters)
fused  = Fuser.fuse(dense, sparse, k=K_FUSE)            # RRF
ranked = Reranker.rerank(query.text, fused, top_n=query.k)
```

- **`SearchKnowledge`** returns `SearchResult` (ranked chunks + source note refs).
- **`SearchSources`** passes `ranked` chunk ids to `SourceRollup.rollup(...)` → `list[SourceHit]` grouped by note (hit count, best score, snippets) — the "appears in notes X, Y, Z" answer.

Defaults follow the Contextual Retrieval findings: dense+sparse hybrid, rerank top-~150 → top-~20.

---

## 11. Delivery: MCP server (`mcp/`)

Thin adapter over the use cases — **the only layer that knows MCP exists.**

| Tool | Signature | Use case |
|------|-----------|----------|
| `search_knowledge` | `(query, k=10, filters?)` → ranked passages + sources | `SearchKnowledge` |
| `search_sources` | `(query, k=10, filters?)` → notes with hit counts + snippets | `SearchSources` |
| `get_note` | `(path)` → full note text + metadata | store read |
| `reindex` | `(paths?, full=false)` → `IndexStats` | `IndexPaths`/`IndexVault` |
| `status` | `()` → `IndexStats` + config summary | `IndexAdmin.stats` |

Runs as a stdio MCP server. Swapping delivery to a CLI/HTTP transport touches only this layer.

---

## 12. Incremental & watcher (`watch/`)

`watchfiles`-based debounced watcher → calls `IndexPaths` for changed/created/deleted paths. The watcher is an outer driver; it depends on the indexing use case, not vice versa. On-demand reindex via the `reindex` MCP tool uses the same use case. Either trigger is sufficient; both are enabled by default.

---

## 13. Configuration & composition root (`config/`)

Single TOML config (plus env overrides). The composition root reads it and wires concrete adapters into use cases — the one place that imports both ports and adapters.

```toml
[vault]
path = "~/Documents/MyVault"
ignore = [".obsidian/", "templates/"]

[embedding]               # provider = "openai_compat" | "local"
provider  = "openai_compat"
base_url  = "http://localhost:11434/v1"   # Ollama/LM Studio/OpenAI/Voyage/TEI...
api_key   = "env:ARIOSTEA_EMBED_KEY"
model     = "nomic-embed-text"
# fallback to bundled fastembed model if unreachable / provider = "local"

[contextual]              # contextual retrieval
enabled   = true
provider  = "openai_compat"   # | "ollama" | "off"
base_url  = "http://localhost:11434/v1"
model     = "qwen2.5:3b"

[rerank]
provider  = "local"       # | "cohere" | "voyage" | "jina" | "tei" | "off"

[store]
backend   = "sqlite"      # | "qdrant" | "lancedb" | "pgvector"
path      = "~/.ariostea/index.db"

[search]
k_dense = 50; k_sparse = 50; k_fuse = 150; top_k = 20; fusion = "rrf"

[watch]
enabled = true; debounce_ms = 750
```

`ariostea init` is an interactive wizard that writes this file, detects a local Ollama, and prints the MCP-server registration snippet for the user's client.

---

## 14. Deployment & packaging

- Distributed on PyPI; run via `uvx ariostea <cmd>` (no Docker, no global install).
- `ariostea init` → config + MCP snippet. `ariostea serve --vault PATH` → stdio MCP server + watcher. `ariostea reindex` / `status` → maintenance.
- Bundled offline stack (fastembed ONNX + sqlite-vec) keeps `uvx ariostea serve` working with **zero keys**.
- Optional `docker compose` file (later phase) for users who want the DB/model runtime containerized.

---

## 15. Contextual Retrieval details

Per Anthropic's method: for each chunk, an LLM produces a 50–100-token blurb situating it in its document; the blurb is prepended to the chunk text **before both embedding and BM25 indexing**. The full document is sent once with `cache_prefix` so prompt caching amortizes its cost across the document's chunks (when the chat provider supports caching). Stored in `chunks.context_blurb`; the `embedding_text` (blurb + chunk) feeds both `chunks_vec` and `chunks_fts`. Toggle: `[contextual].enabled`. When off or no LLM is available, `embedding_text == chunk.text` and quality degrades gracefully to plain hybrid search.

---

## 16. Cross-cutting concerns

- **Errors:** adapters raise port-level exceptions (`EmbeddingUnavailable`, `StoreError`); use cases translate to domain outcomes; the MCP layer maps to tool errors. A provider outage during indexing degrades (skip contextualization, or queue file for retry) rather than corrupting the index.
- **Logging/metrics:** structured logs at the adapter boundary; `status` surfaces counts and last-index time.
- **Testing strategy (clean-arch payoff):**
  - *Domain & use cases:* pure unit tests with fake ports — no sqlite/ONNX/network.
  - *Port contract tests:* one shared suite per port, run against every adapter (enforces LSP).
  - *Adapter integration tests:* real sqlite-vec, real fastembed, recorded HTTP for OpenAI-compat.
  - *End-to-end:* index a fixture vault → assert tool outputs.

---

## 17. Build sequence (tracer-bullet phases)

Each phase is independently shippable and demonstrable end-to-end.

| Phase | Deliverable | Acceptance |
|-------|-------------|------------|
| 0 | Scaffold + domain + ports + composition root + MCP server with a stub tool | `uvx ariostea serve` starts; stub tool responds |
| 1 | **Walking skeleton:** scan→parse→naive chunk→fastembed→sqlite-vec dense→`search_knowledge` | Query a fixture vault, get relevant passages |
| 2 | FTS5 sparse + RRF hybrid | Hybrid beats dense-only on fixture queries |
| 3 | Provenance rollup + `search_sources` | "appears in notes X, Y, Z" returns correct notes |
| 4 | Incremental (hash/mtime diff) + watcher | Editing a note updates only that note's chunks |
| 5 | Contextual Retrieval (contextualizer + prompt caching) | Blurbs stored; retrieval quality improves on eval set |
| 6 | Reranking stage | Rerank reorders top-N measurably |
| 7 | Deep Obsidian structure (link graph, backlinks, tag/frontmatter filters, heading-aware refinement) | Filters + graph signals usable in search |
| 8 | Packaging polish (`init` wizard, docs) + extra store/rerank adapters + configurable FTS tokenizer | One-command onboarding documented; alt adapters pass contract tests |

> **Phase 8 backlog — configurable FTS5 tokenizer.** The sparse side currently uses FTS5's default `unicode61` (word) tokenizer, which is correct for space-separated languages (English, Italian, …). Expose the tokenizer as a `[store]` config knob so users with **CJK / no-space scripts** (Chinese, Japanese, Thai) or who want **substring/fuzzy matching** can opt into the `trigram` tokenizer. Trade-off: larger index, noisier BM25 relevance. Note this is purely a *lexical* concern — it never makes BM25 cross-lingual (that remains the embedding layer's job); it only changes the unit and granularity of literal matching. Lives entirely inside `SqliteStore` (the `CREATE VIRTUAL TABLE ... USING fts5(tokenize=...)` clause), so it's a localized, swappable change.

---

## 18. Decided defaults (open to change)

- **Store:** sqlite-vec + FTS5 single file (best fit for a personal vault; one engine for vectors, BM25, filters, provenance).
- **Bundled local stack:** fastembed (ONNX, no PyTorch) for embeddings + reranker.
- **Fusion:** Reciprocal Rank Fusion.
- **Universal provider interface:** OpenAI-compatible API for embeddings + chat.

## 19. Risks & open questions

- **Local contextualization cost/latency** on large vaults — mitigated by incremental indexing, prompt caching, and the off-by-config escape hatch.
- **Embedding-model swap** changes vector dimension → guarded full reindex on fingerprint change (handled via `meta`).
- **fastembed model footprint** in a `uvx` one-shot — pick a small default model; document the first-run download.
- **Reranker fragmentation** (no OpenAI standard) — accepted; isolated in its own adapter family with a local default.
- Deferred: dedicated entity/concept index (a second "source search" mode) — revisit after v1.

---

## 20. Clean-architecture conformance checklist

- [x] Dependencies point inward; `domain/` imports nothing internal.
- [x] Use cases unaware of delivery (MCP/CLI) and of frameworks.
- [x] DB schema does not shape the domain model.
- [x] Business rules testable without frameworks running.
- [x] Ports segregated by role (ISP); adapters substitutable (LSP); new providers add code, don't modify it (OCP).
- [x] Composition root is the only multi-layer importer; dependency graph is acyclic.
