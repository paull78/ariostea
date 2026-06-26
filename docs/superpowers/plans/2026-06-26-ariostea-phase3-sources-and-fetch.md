# Ariostea — Phase 3 (Source Provenance + Note Fetch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the second search mode — `search_sources` (provenance rollup: "this concept appears in notes X, Y, Z") — and a `get_note` fetch tool, completing the agentic **search + fetch** pattern over the hybrid retrieval built in Phase 2.

**Architecture:** Ports & adapters, unchanged dependency rule. A new `DocumentReader` role-port on the store exposes note titles (for grouping) and full-note reconstruction (for fetch). A new `SearchSources` use case **composes** the existing `SearchKnowledge` hybrid retriever, then groups the fused chunks by source note into `SourceHit`s. Two new MCP tools (`search_sources`, `get_note`) and the composition-root wiring complete the delivery layer.

**Tech Stack:** Python 3.12, `uv`, `pytest` (TDD), stdlib `sqlite3`, `mcp` (FastMCP). No new dependencies.

---

## How we work through this plan

Each task starts with **Why this shape** (the design reasoning — read it, ask me anything before we write code), then runs the TDD loop: failing test → confirm it fails → minimal implementation → confirm it passes → commit. We do **not** advance to the next task until you're satisfied. Stop me with questions at any checkbox.

## Scope of this plan

In scope: a `NoteDocument` domain model; a `DocumentReader` port (`note_titles`, `read_note`) implemented by `SqliteStore`; a `SearchSources` use case that groups fused chunks into `SourceHit`s; MCP `search_sources` and `get_note` tools + handlers; composition-root wiring exposing `sources` (use case) and `reader` (port); an end-to-end test for both new tools.

**Deferred to later phase plans:** incremental hash-diff + watcher (Phase 4); contextual blurbs (Phase 5); reranking (Phase 6); deep Obsidian filters/graph (Phase 7); section/neighbor context-expansion of search results, snippet term-highlighting via FTS5 `snippet()`, and richer `get_note` metadata (tags/frontmatter) (Phase 8 / spec §11 presentation ladder).

## Conventions

- Run one test: `uv run pytest tests/path::test_name -v`
- Run all unit tests: `uv run pytest -q -m "not integration"`
- Run everything (incl. model-loading integration tests): `uv run pytest -q`
- Before each commit: `uv run ruff check --fix . && uv run ruff format . && uv run ruff check .`
- Commit after every green step. Conventional Commits.

## Design decisions locked in this plan

- **Rollup is grouping over *fused* chunks, in the use case — not a store re-query.** `SearchSources` reuses `SearchKnowledge.search()` so the provenance reflects the *hybrid (RRF) relevance* the user actually gets. `best_score` is the max fused score per note; `hit_count` is how many of that note's chunks surfaced. (Spec §10.) This is also the structural fix for the "one long note floods flat results" issue (§19) — sources collapses a note's many chunks into one ranked entry.
- **Titles come from a thin `DocumentReader.note_titles` batch lookup**, not threaded through retrieval. `Chunk` carries `note_path` but not `title` (title is a `Note` property); a single batched `SELECT ... WHERE path IN (...)` keeps the use case clean and avoids N queries.
- **`get_note` reconstructs from stored chunks** (title + chunk texts joined in `ordinal` order), not from disk. This keeps the reader self-contained (no vault-root coupling, returns exactly what was indexed). Trade-off: not byte-exact (frontmatter/blank-line fidelity lost, no tags — the store doesn't persist them). Disk-read + full metadata is a Phase 8 enhancement, noted in scope.
- **Candidate pool ≥ returned sources.** `SearchSources` retrieves a wide chunk pool (`pool=50`) to find *all* notes a concept touches, then returns the top `query.k` notes. `Query.k` means "number of source notes" here; the chunk breadth is internal policy.
- **Container exposes `sources` (use case) + `reader` (port-typed), consistent with `admin`** — never the concrete store. (Honors the existing container-exposes-ports convention.)

---

## Task 3.1: `DocumentReader` port + `NoteDocument` model + `SqliteStore` implementation

**Files:**
- Modify: `src/ariostea/domain/models.py` (add `NoteDocument`)
- Modify: `src/ariostea/ports/store.py` (add `DocumentReader` Protocol)
- Modify: `src/ariostea/adapters/store/sqlite_store.py` (implement it)
- Modify: `tests/adapters/store/test_sqlite_store.py`
- Modify: `tests/ports/test_protocols.py`

**Why this shape:** Both new features need to read note-level data the chunk-retrieval path doesn't carry: `search_sources` needs **titles** to label grouped notes; `get_note` needs the **full note text**. These are a distinct *role* from writing (`DocumentWriter`) and from chunk retrieval (`ChunkRetriever`), so by Interface Segregation they get their own port — `DocumentReader`. The store already holds everything: titles in the `notes` table, ordered text in `chunks`. `note_titles` is a batched lookup (one query for many paths). `read_note` reconstructs the note by concatenating its chunks in `ordinal` order — self-contained, returns the indexed view. `NoteDocument` is the small frozen return type so the use case/handler speak in domain terms, not raw tuples.

- [ ] **Step 1: Write the failing tests**

Append to `tests/adapters/store/test_sqlite_store.py` (reusing the existing `_note`/`_cchunk` helpers; note `_note` defaults to path `a.md`, title `A`):

```python
def test_note_titles_batch_lookup(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    store.upsert_note(_note("a.md"), [_cchunk(_note("a.md"), 0, "x")], [[1.0, 0.0, 0.0]])
    store.upsert_note(_note("b.md"), [_cchunk(_note("b.md"), 0, "y")], [[0.0, 1.0, 0.0]])

    titles = store.note_titles(["a.md", "b.md", "missing.md"])
    assert titles == {"a.md": "A", "b.md": "A"}  # _note() always titles "A"
    assert store.note_titles([]) == {}


def test_read_note_reconstructs_in_ordinal_order(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note("a.md")
    chunks = [_cchunk(note, 0, "first part"), _cchunk(note, 1, "second part")]
    store.upsert_note(note, chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    doc = store.read_note("a.md")
    assert doc is not None
    assert doc.note_path == "a.md"
    assert doc.title == "A"
    assert doc.text == "first part\n\nsecond part"


def test_read_note_returns_none_when_absent(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    assert store.read_note("nope.md") is None
```

Append to `tests/ports/test_protocols.py`:

```python
def test_sqlite_store_conforms_to_document_reader():
    from ariostea.adapters.store.sqlite_store import SqliteStore
    from ariostea.ports.store import DocumentReader

    assert issubclass(SqliteStore, DocumentReader)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py tests/ports/test_protocols.py -v`
Expected: FAIL — `AttributeError: 'SqliteStore' object has no attribute 'note_titles'` / `read_note`, and the protocol import of `DocumentReader` fails (`ImportError`/`AttributeError`).

- [ ] **Step 3: Write minimal implementation**

**(a)** Add the model to `src/ariostea/domain/models.py` (after `SourceHit`):

```python
@dataclass(frozen=True)
class NoteDocument:
    note_path: str
    title: str
    text: str
```

**(b)** Add the port to `src/ariostea/ports/store.py`. First extend the domain import to include `NoteDocument`:

```python
from ariostea.domain.models import (
    ContextualizedChunk,
    IndexStats,
    Note,
    NoteDocument,
    QueryFilters,
    RetrievedChunk,
)
```

Then add the Protocol (place it after `ChunkRetriever`):

```python
@runtime_checkable
class DocumentReader(Protocol):
    def note_titles(self, paths: Sequence[str]) -> dict[str, str]: ...
    def read_note(self, path: str) -> NoteDocument | None: ...
```

**(c)** Implement in `src/ariostea/adapters/store/sqlite_store.py`:

Add `NoteDocument` to the domain import block, and `DocumentReader` to the ports import:

```python
from ariostea.domain.models import (
    Chunk,
    ContextualizedChunk,
    IndexStats,
    Note,
    NoteDocument,
    QueryFilters,
    RetrievedChunk,
)
from ariostea.ports.store import ChunkRetriever, DocumentReader, DocumentWriter, IndexAdmin
```

Add `DocumentReader` to the class bases:

```python
class SqliteStore(DocumentWriter, ChunkRetriever, IndexAdmin, DocumentReader):
```

Add the two methods (place them after the `sparse` method, before `# --- IndexAdmin ---`):

```python
    # --- DocumentReader ---
    def note_titles(self, paths: Sequence[str]) -> dict[str, str]:
        paths = list(paths)
        if not paths:
            return {}
        placeholders = ",".join("?" for _ in paths)
        rows = self.db.execute(
            f"SELECT path, title FROM notes WHERE path IN ({placeholders})", paths
        ).fetchall()
        return {r["path"]: r["title"] for r in rows}

    def read_note(self, path: str) -> NoteDocument | None:
        note = self.db.execute("SELECT title FROM notes WHERE path = ?", (path,)).fetchone()
        if note is None:
            return None
        rows = self.db.execute(
            "SELECT c.text FROM chunks c JOIN notes n ON n.id = c.note_id "
            "WHERE n.path = ? ORDER BY c.ordinal",
            (path,),
        ).fetchall()
        body = "\n\n".join(r["text"] for r in rows)
        return NoteDocument(note_path=path, title=note["title"], text=body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py tests/ports/test_protocols.py -v`
Expected: PASS (the three new store tests + the new protocol test, plus all prior ones).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . >/dev/null; uv run ruff format . >/dev/null && uv run ruff check .
git add src/ariostea/domain/models.py src/ariostea/ports/store.py src/ariostea/adapters/store/sqlite_store.py tests/adapters/store/test_sqlite_store.py tests/ports/test_protocols.py
git commit -m "feat(store): DocumentReader port (note titles + full-note read)"
```

---

## Task 3.2: `SearchSources` use case (provenance rollup)

**Files:**
- Create: `src/ariostea/search/search_sources.py`
- Create: `tests/search/test_search_sources.py`

**Why this shape:** `SearchSources` answers "where does concept X appear?" It is application policy, so it lives in `search/` and depends only on abstractions: the existing `SearchKnowledge` use case (composition — DRY reuse of the hybrid dense+sparse+RRF core) and the `DocumentReader` port (for titles). It retrieves a **wide** chunk pool (`pool=50`) so it can discover *every* note a concept touches, groups those fused chunks by `note_path`, and emits one `SourceHit` per note (`hit_count`, `best_score` = max fused score, a few `snippets`), sorted by best score and truncated to `query.k` notes. No store schema knowledge, no framework types — unit-testable with fakes.

- [ ] **Step 1: Write the failing test**

```python
# tests/search/test_search_sources.py
from ariostea.domain.models import Chunk, Query, RetrievedChunk, SearchResult
from ariostea.search.search_sources import SearchSources


def _rc(path, ordinal, score, text):
    chunk = Chunk(
        note_path=path, ordinal=ordinal, heading_path=("H",), text=text, token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=score)


class FakeSearcher:
    """Stands in for SearchKnowledge — records the Query it was given."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.last_query = None

    def search(self, query):
        self.last_query = query
        return SearchResult(chunks=tuple(self._chunks))


class FakeReader:
    def note_titles(self, paths):
        return {p: p.upper() for p in paths}

    def read_note(self, path):
        return None


def test_groups_chunks_by_note_with_counts_scores_snippets():
    chunks = [
        _rc("a.md", 0, 0.9, "alpha one"),
        _rc("a.md", 1, 0.4, "alpha two"),
        _rc("b.md", 0, 0.7, "beta one"),
    ]
    uc = SearchSources(searcher=FakeSearcher(chunks), reader=FakeReader())
    hits = uc.search(Query(text="x", k=10))

    by_path = {h.note_path: h for h in hits}
    assert by_path["a.md"].hit_count == 2
    assert by_path["a.md"].best_score == 0.9  # max fused score in the note
    assert by_path["a.md"].title == "A.MD"  # from reader.note_titles
    assert "alpha one" in by_path["a.md"].snippets
    assert by_path["b.md"].hit_count == 1


def test_sources_sorted_by_best_score_and_truncated_to_k():
    chunks = [
        _rc("a.md", 0, 0.3, "a"),
        _rc("b.md", 0, 0.9, "b"),
        _rc("c.md", 0, 0.6, "c"),
    ]
    uc = SearchSources(searcher=FakeSearcher(chunks), reader=FakeReader())
    hits = uc.search(Query(text="x", k=2))
    assert [h.note_path for h in hits] == ["b.md", "c.md"]  # best first, top 2


def test_retrieves_wide_pool_not_just_query_k():
    uc = SearchSources(searcher=(fs := FakeSearcher([])), reader=FakeReader(), pool=50)
    uc.search(Query(text="x", k=3))
    assert fs.last_query.k == 50  # broad chunk pool, independent of returned-note count
    assert fs.last_query.text == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_search_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.search.search_sources'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/search/search_sources.py
from __future__ import annotations

from ariostea.domain.models import Query, RetrievedChunk, SourceHit
from ariostea.ports.store import DocumentReader
from ariostea.search.search_knowledge import SearchKnowledge

_SNIPPET_CHARS = 160
_MAX_SNIPPETS = 3


class SearchSources:
    def __init__(
        self, searcher: SearchKnowledge, reader: DocumentReader, pool: int = 50
    ) -> None:
        self._searcher = searcher
        self._reader = reader
        self._pool = pool

    def search(self, query: Query) -> list[SourceHit]:
        broad = Query(text=query.text, k=self._pool, filters=query.filters)
        chunks = self._searcher.search(broad).chunks

        groups: dict[str, list[RetrievedChunk]] = {}
        for rc in chunks:
            groups.setdefault(rc.chunk.note_path, []).append(rc)

        titles = self._reader.note_titles(list(groups))
        hits = [
            SourceHit(
                note_path=path,
                title=titles.get(path, path),
                hit_count=len(rcs),
                best_score=max(rc.score for rc in rcs),
                snippets=tuple(rc.chunk.text[:_SNIPPET_CHARS] for rc in rcs[:_MAX_SNIPPETS]),
            )
            for path, rcs in groups.items()
        ]
        hits.sort(key=lambda h: h.best_score, reverse=True)
        return hits[: query.k]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_search_sources.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . >/dev/null; uv run ruff format . >/dev/null && uv run ruff check .
git add src/ariostea/search/search_sources.py tests/search/test_search_sources.py
git commit -m "feat(search): SearchSources use case (provenance rollup by note)"
```

---

## Task 3.3: MCP delivery — `search_sources` + `get_note` tools + wiring

**Files:**
- Modify: `src/ariostea/mcp/handlers.py`
- Modify: `src/ariostea/mcp/server.py`
- Modify: `src/ariostea/config/container.py`
- Modify: `tests/mcp/test_handlers.py`
- Modify: `tests/test_end_to_end.py`

**Why this shape:** The delivery layer stays a thin translator. Two pure handler functions convert use-case/port output to plain dicts (testable without a running server); `server.py` wraps them as FastMCP tools. The composition root constructs `SearchSources` and exposes it as `sources`, plus the store's `DocumentReader` face as `reader` — keeping the container to use-cases-and-ports (never the concrete store), consistent with `admin`. The `get_note` tool completes the agentic **search + fetch** pair: cheap broad search, then targeted full-note fetch on the agent's judgment.

- [ ] **Step 1: Write the failing tests**

Append to `tests/mcp/test_handlers.py`:

```python
def test_search_sources_payload_shapes_hits():
    from types import SimpleNamespace

    from ariostea.domain.models import SourceHit
    from ariostea.mcp.handlers import search_sources_payload

    class FakeSources:
        def search(self, query):
            return [
                SourceHit(
                    note_path="a.md",
                    title="A",
                    hit_count=2,
                    best_score=0.9,
                    snippets=("s1", "s2"),
                )
            ]

    container = SimpleNamespace(sources=FakeSources())
    payload = search_sources_payload(container, query="x", k=5)
    assert payload["sources"][0] == {
        "note_path": "a.md",
        "title": "A",
        "hit_count": 2,
        "best_score": 0.9,
        "snippets": ["s1", "s2"],
    }


def test_get_note_payload_found_and_missing():
    from types import SimpleNamespace

    from ariostea.domain.models import NoteDocument
    from ariostea.mcp.handlers import get_note_payload

    class FakeReader:
        def __init__(self, doc):
            self._doc = doc

        def read_note(self, path):
            return self._doc

    doc = NoteDocument(note_path="a.md", title="A", text="hello body")
    found = get_note_payload(SimpleNamespace(reader=FakeReader(doc)), path="a.md")
    assert found == {"found": True, "note_path": "a.md", "title": "A", "text": "hello body"}

    missing = get_note_payload(SimpleNamespace(reader=FakeReader(None)), path="x.md")
    assert missing == {"found": False, "note_path": "x.md"}
```

Append to `tests/test_end_to_end.py`:

```python
@pytest.mark.integration
def test_search_sources_and_get_note_end_to_end(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# Alpha\nVector databases store embeddings for search.")
    (vault / "b.md").write_text("# Beta\nEmbeddings power semantic vector search too.")

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    from ariostea.mcp.handlers import get_note_payload, search_sources_payload

    sources = search_sources_payload(container, query="embeddings vector search", k=5)
    paths = {s["note_path"] for s in sources["sources"]}
    assert {"a.md", "b.md"} <= paths  # the concept appears in both notes
    assert all(s["hit_count"] >= 1 and s["title"] for s in sources["sources"])

    note = get_note_payload(container, path="a.md")
    assert note["found"] is True
    assert "Vector databases" in note["text"]

    assert get_note_payload(container, path="does-not-exist.md")["found"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/mcp/test_handlers.py -v`
Expected: FAIL — `ImportError: cannot import name 'search_sources_payload'` / `get_note_payload`.

- [ ] **Step 3: Write minimal implementation**

**(a)** Add handlers to `src/ariostea/mcp/handlers.py`:

```python
def search_sources_payload(container: "Container", query: str, k: int = 10) -> dict:
    hits = container.sources.search(Query(text=query, k=k))
    return {
        "sources": [
            {
                "note_path": h.note_path,
                "title": h.title,
                "hit_count": h.hit_count,
                "best_score": h.best_score,
                "snippets": list(h.snippets),
            }
            for h in hits
        ]
    }


def get_note_payload(container: "Container", path: str) -> dict:
    doc = container.reader.read_note(path)
    if doc is None:
        return {"found": False, "note_path": path}
    return {"found": True, "note_path": doc.note_path, "title": doc.title, "text": doc.text}
```

**(b)** Register the tools in `src/ariostea/mcp/server.py`. Extend the handler import:

```python
from ariostea.mcp.handlers import (
    get_note_payload,
    reindex_payload,
    search_payload,
    search_sources_payload,
    status_payload,
)
```

Add the two tools inside `build_server` (before `return mcp`):

```python
    @mcp.tool()
    def search_sources(query: str, k: int = 10) -> dict:
        """Find which notes a concept appears in. Returns notes with hit counts, best score, and snippets."""
        return search_sources_payload(container, query=query, k=k)

    @mcp.tool()
    def get_note(path: str) -> dict:
        """Fetch a full note's reconstructed text and title by its vault-relative path."""
        return get_note_payload(container, path=path)
```

**(c)** Wire the composition root `src/ariostea/config/container.py`. Add imports:

```python
from ariostea.ports.store import DocumentReader, IndexAdmin
from ariostea.search.search_sources import SearchSources
```

(Replace the existing `from ariostea.ports.store import IndexAdmin` line with the combined one above.)

Add two fields to the `Container` dataclass (after `searcher`):

```python
    sources: SearchSources
    reader: DocumentReader
```

Build and pass them in `build_container` — after the `searcher = SearchKnowledge(...)` block, add:

```python
    sources = SearchSources(searcher=searcher, reader=store)
```

and update the return:

```python
    return Container(
        config=config,
        indexer=indexer,
        searcher=searcher,
        sources=sources,
        admin=store,
        reader=store,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/mcp/test_handlers.py -v` → PASS (2 new + existing).
Then the full suite: `uv run pytest -q` → PASS (all unit + integration, including the new e2e test; first run may load the model).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . >/dev/null; uv run ruff format . >/dev/null && uv run ruff check .
git add src/ariostea/mcp/handlers.py src/ariostea/mcp/server.py src/ariostea/config/container.py tests/mcp/test_handlers.py tests/test_end_to_end.py
git commit -m "feat(mcp): search_sources + get_note tools (search+fetch pattern)"
```

---

> **Phase 3 complete:** both search modes are live — `search_knowledge` (passages) and `search_sources` (provenance) — plus `get_note` for full-note fetch. The agentic search+fetch loop is in place.

---

## Self-review (completed by plan author)

- **Spec coverage:** §1 use cases U2 (`search_sources`) → Tasks 3.2 + 3.3; U3 (`get_note`) → Tasks 3.1 + 3.3. §6 `SourceRollup`/reader role → Task 3.1 `DocumentReader` (adapted: rollup logic lives in the `SearchSources` use case over fused chunks, titles via the reader — chosen over a store `chunk_ids` rollup so scores reflect RRF; documented in Design decisions). §10 `SearchSources` retrieval core (reuses hybrid) → Task 3.2. §11 delivery table `search_sources`/`get_note` rows → Task 3.3. §11 "group by source note" presentation note → Task 3.2. §17 Phase 3 acceptance ("appears in notes X, Y, Z"; fetch a full note) → e2e test in Task 3.3.
- **Type consistency:** `note_titles(paths: Sequence[str]) -> dict[str, str]` and `read_note(path: str) -> NoteDocument | None` identical in port (3.1), adapter (3.1), and call sites (3.2 reader, 3.3 handler). `SearchSources(searcher, reader, pool=50).search(Query) -> list[SourceHit]` consistent across use case (3.2), container (3.3), and handler (3.3). `NoteDocument(note_path, title, text)` fields match across model, reader, handler, and tests. `SourceHit` fields (`note_path, title, hit_count, best_score, snippets`) match the existing domain model.
- **Placeholders:** none — every code step shows complete code; every command shows expected output.
- **Deviation noted:** `get_note` reconstructs from chunks rather than reading disk; `SourceHit.title` comes from a reader lookup rather than the spec's `chunk_ids`-based `SourceRollup` — both rationalized in Design decisions, both swappable later.
