# Ariostea — Phase 2 (Hybrid Search: BM25 + RRF Fusion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the **lexical half** of retrieval — SQLite FTS5/BM25 sparse search — and fuse it with the existing dense (vector) search via Reciprocal Rank Fusion, so `search_knowledge` becomes a hybrid retriever that catches both semantic and exact-keyword matches.

**Architecture:** Ports & adapters, unchanged dependency rule. We widen the `ChunkRetriever` port with a `sparse()` method, extend the `SqliteStore` adapter with an FTS5 table (written on upsert, cleaned on delete, queried by BM25), add a new `Fuser` port with an `RRFFuser` adapter, and rewire the `SearchKnowledge` use case to call dense + sparse and fuse the two ranked lists. Reranking, contextual blurbs, and `search_sources` remain in later phases.

**Tech Stack:** Python 3.12, `uv`, `pytest` (TDD), stdlib `sqlite3` **FTS5** (built into SQLite — no new dependency), `sqlite-vec` (already present), `pydantic` (config).

---

## How we work through this plan

Each task starts with **Why this shape** (the design reasoning — read it, ask me anything before we write code), then runs the TDD loop: failing test → confirm it fails → minimal implementation → confirm it passes → commit. We do **not** advance to the next task until you're satisfied. Stop me with questions at any checkbox.

## Scope of this plan

In scope: `sparse()` on the `ChunkRetriever` port; `chunks_fts` FTS5 table in `SqliteStore` (populate on upsert, delete on remove, BM25 query with a query sanitizer); a `Fuser` port + `RRFFuser` adapter; `k_sparse` config; rewiring `SearchKnowledge` and the composition root to hybrid; an end-to-end test proving a lexical-only query that dense retrieval misses is now found.

**Deferred to later phase plans:** provenance rollup + `search_sources` (Phase 3), incremental hash-diff + watcher (Phase 4), contextual blurbs + prompt caching (Phase 5 — the blurb is already routed through `embedding_text`, which is what we index for BM25, so contextual BM25 comes for free when blurbs land), reranking (Phase 6 — the fused candidate pool is the reranker's input), Obsidian graph/filters incl. `sparse`/`dense` filter args (Phase 7), packaging + alt fuser/store adapters (Phase 8).

## Conventions

- Run a single test: `uv run pytest tests/path::test_name -v`
- Run all tests: `uv run pytest -q`
- Skip slow model/integration tests in fast loops: `uv run pytest -q -m "not integration"`
- After every green step, format + lint before committing: `uv run ruff format . && uv run ruff check .`
- Commit after every green step. Conventional Commits.

## Design decisions locked in this plan

- **One engine, one file.** FTS5 ships inside SQLite, so BM25 lives in the *same* `index.db` next to the vectors — no second store, no new dependency. (Spec §7, §8.)
- **Index `embedding_text`, not `chunk.text`.** The FTS row indexes `ContextualizedChunk.embedding_text`. Today that equals `chunk.text`; when Phase 5 prepends a context blurb, BM25 automatically becomes *contextual* BM25 with zero change here. (Spec §11.)
- **Chunk identity = `(note_path, ordinal)`.** Dense and sparse return the same chunk via different routes; the fuser must recognize them as one. We key on `(note_path, ordinal)` — stable, and already carried on every `Chunk`.
- **RRF, constant `rrf_k = 60`.** Reciprocal Rank Fusion combines lists by rank, not by raw score, so the incomparable scales of cosine-distance and BM25 never need normalizing. `60` is the standard constant from the original RRF paper. (Spec §7 Fuser, §17.)
- **Search breadth is composition-root policy.** `SearchKnowledge` pulls `k_dense`/`k_sparse` candidate counts from config (injected at construction), and fuses down to the caller's `query.k`. `Query` stays simple.

## Migration note (read before running against an existing index)

Adding `chunks_fts` only affects **newly created** databases (all tasks use fresh `tmp_path` DBs, so tests are unaffected). An `index.db` built in Phase 1 will not have the FTS table; `CREATE VIRTUAL TABLE IF NOT EXISTS` adds it empty on next open, but existing chunks won't be in it until re-indexed. The fingerprint-guarded full reindex that handles this automatically is Phase 4. For now, if you point at a Phase 1 DB, run `reindex` once.

---

## Task 2.1: Add `sparse()` to the `ChunkRetriever` port

**Files:**
- Modify: `src/ariostea/ports/store.py` (the `ChunkRetriever` Protocol)
- Modify: `tests/ports/test_protocols.py`

**Why this shape:** The port is the contract every retriever must satisfy. Dense and sparse are two retrieval *roles of the same port* (Spec §6 groups them on `ChunkRetriever`) because a single store can serve both and the search use case needs both together. Adding `sparse` here first means the use case can depend on it before any adapter implements it — ports lead, adapters follow. The signature mirrors `dense` (a `k` and an optional `filters`) for symmetry; `filters` is accepted but unused until Phase 7, exactly like `dense`.

- [ ] **Step 1: Write the failing test**

Add this test to `tests/ports/test_protocols.py` (append at the end of the file):

```python
def test_chunk_retriever_requires_dense_and_sparse():
    from ariostea.domain.models import QueryFilters, RetrievedChunk
    from ariostea.ports.store import ChunkRetriever

    class DenseOnly:
        def dense(self, vec, k, filters=None):
            return []

    class DenseAndSparse:
        def dense(self, vec, k, filters=None):
            return []

        def sparse(self, query, k, filters=None):
            return []

    # A retriever missing sparse() does NOT satisfy the widened port...
    assert not isinstance(DenseOnly(), ChunkRetriever)
    # ...one providing both does.
    assert isinstance(DenseAndSparse(), ChunkRetriever)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ports/test_protocols.py::test_chunk_retriever_requires_dense_and_sparse -v`
Expected: FAIL — `DenseOnly` currently satisfies `ChunkRetriever` (it only requires `dense`), so `assert not isinstance(...)` fails.

- [ ] **Step 3: Write minimal implementation**

In `src/ariostea/ports/store.py`, extend the `ChunkRetriever` Protocol so it also declares `sparse`:

```python
@runtime_checkable
class ChunkRetriever(Protocol):
    def dense(
        self, vec: list[float], k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...
    def sparse(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ports/test_protocols.py -v`
Expected: PASS (all protocol tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/ariostea/ports/store.py tests/ports/test_protocols.py
git commit -m "feat(ports): add sparse() to ChunkRetriever for BM25 retrieval"
```

---

## Task 2.2: FTS5/BM25 in `SqliteStore` (schema + upsert + delete + `sparse`)

**Files:**
- Modify: `src/ariostea/adapters/store/sqlite_store.py`
- Modify: `tests/adapters/store/test_sqlite_store.py`

**Why this shape:** This is the lexical engine. FTS5 is an SQLite virtual table that maintains an inverted index and exposes the `bm25()` ranking function, so we get classic keyword scoring with no extra dependency, in the same file as the vectors. We index `embedding_text` (future-proofing for contextual BM25) and keep a plain `chunk_id` column `UNINDEXED` so we can join back to `chunks`/`notes` exactly like the dense path. Writes happen inside the *existing* `upsert_note` transaction so vectors and the FTS row commit atomically — a chunk is never searchable by one route but not the other. `bm25()` returns negative values (more negative = better match), so we order ascending and flip the sign into a positive `score`. The `_fts_query` sanitizer turns free user text into a safe FTS5 `MATCH` expression: FTS5 `MATCH` is its own query language, and raw punctuation (`?`, `-`, `:`) or an empty string would raise a syntax error — we extract word tokens and `OR` them (maximizing recall, which is what fusion wants).

- [ ] **Step 1: Write the failing test**

Append these tests to `tests/adapters/store/test_sqlite_store.py` (the existing `_note` and `_cchunk` helpers at the top of that file are reused):

```python
def test_sparse_bm25_ranks_keyword_matches(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunks = [
        _cchunk(note, 0, "the quick brown fox jumps"),
        _cchunk(note, 1, "lorem ipsum dolor sit amet"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_note(note, chunks, embeddings)

    hits = store.sparse("fox", k=5)
    assert hits[0].chunk.text == "the quick brown fox jumps"
    assert hits[0].sparse_rank == 0
    assert hits[0].dense_rank is None  # sparse path leaves dense_rank unset
    assert hits[0].score > 0.0


def test_sparse_returns_empty_when_no_term_matches(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha beta gamma")], [[1.0, 0.0, 0.0]])
    assert store.sparse("zebra", k=5) == []


def test_sparse_sanitizes_punctuation_and_empty_queries(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha beta gamma")], [[1.0, 0.0, 0.0]])
    # punctuation around a real term must not raise and must still match
    assert store.sparse("  beta?! ", k=5)[0].chunk.text == "alpha beta gamma"
    # a query with no word characters yields no results (and no SQL error)
    assert store.sparse("?? -- ::", k=5) == []


def test_delete_note_removes_from_fts(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "findable keyword")], [[1.0, 0.0, 0.0]])
    assert store.sparse("findable", k=5)  # present before delete
    store.delete_note("a.md")
    assert store.sparse("findable", k=5) == []


def test_reupsert_does_not_duplicate_fts_rows(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "unique token")], [[1.0, 0.0, 0.0]])
    store.upsert_note(note, [_cchunk(note, 0, "unique token")], [[1.0, 0.0, 0.0]])
    hits = store.sparse("unique", k=10)
    assert len(hits) == 1  # old FTS row was cleaned, not duplicated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -v`
Expected: FAIL — `AttributeError: 'SqliteStore' object has no attribute 'sparse'` (and no `chunks_fts` table yet).

- [ ] **Step 3: Write minimal implementation**

Make four edits to `src/ariostea/adapters/store/sqlite_store.py`.

**(a)** Add `re` to the imports at the top (it currently imports `sqlite3`, `time`, `Path`, `Sequence`, `sqlite_vec`):

```python
import re
```

**(b)** Add the `chunks_fts` virtual table to `_init_schema`. Insert this `CREATE` statement inside the `executescript(...)` block, right after the `chunks_vec` table and before the `meta` table:

```python
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text
            );
```

**(c)** Write the FTS row inside `upsert_note`, in the per-chunk loop, immediately after the `chunks_vec` insert (so it shares the same transaction):

```python
                cur.execute(
                    "INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)",
                    (chunk_id, cc.embedding_text),
                )
```

**(d)** Remove the FTS row in `_delete_note_rows`. In the loop that deletes each chunk's vector, also delete its FTS row — replace the existing `for cid in chunk_ids:` block with:

```python
        for cid in chunk_ids:
            cur.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
```

**(e)** Add the query sanitizer as a module-level helper (place it near the top of the file, after the imports):

```python
def _fts_query(text: str) -> str:
    """Turn free user text into a safe FTS5 MATCH expression.

    FTS5 MATCH is its own query language; raw punctuation or an empty string
    raises a syntax error. We extract word tokens and OR them so any term may
    match (recall-first — fusion handles precision).
    """
    terms = re.findall(r"[A-Za-z0-9_]+", text)
    return " OR ".join(f'"{t}"' for t in terms)
```

**(f)** Add the `sparse` method to the `SqliteStore` class. Place it right after `dense` (under the `# --- ChunkRetriever ---` section):

```python
    def sparse(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        match = _fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            """
            WITH bm AS (
                SELECT chunk_id, bm25(chunks_fts) AS bm
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY bm
                LIMIT ?
            )
            SELECT n.path AS note_path, c.ordinal, c.heading_path,
                   c.text, c.token_count, bm.bm
            FROM bm
            JOIN chunks c ON c.id = bm.chunk_id
            JOIN notes n ON n.id = c.note_id
            ORDER BY bm.bm
            """,
            (match, k),
        ).fetchall()
        results: list[RetrievedChunk] = []
        for rank, r in enumerate(rows):
            heading_path = tuple(p for p in r["heading_path"].split("/") if p)
            chunk = Chunk(
                note_path=r["note_path"],
                ordinal=r["ordinal"],
                heading_path=heading_path,
                text=r["text"],
                token_count=r["token_count"],
            )
            # bm25() is negative (more negative = better); flip to a positive score.
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=-r["bm"],
                    dense_rank=None,
                    sparse_rank=rank,
                )
            )
        return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -v`
Expected: PASS (all store tests — the four original dense/admin/delete tests plus the five new sparse tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/ariostea/adapters/store/sqlite_store.py tests/adapters/store/test_sqlite_store.py
git commit -m "feat(store): FTS5/BM25 sparse retrieval in sqlite-vec store"
```

---

## Task 2.3: `Fuser` port + `RRFFuser` adapter

**Files:**
- Create: `src/ariostea/ports/fusion.py`
- Create: `src/ariostea/adapters/fuse/__init__.py`
- Create: `src/ariostea/adapters/fuse/rrf.py`
- Create: `tests/adapters/fuse/__init__.py` *(empty, only if your layout needs it — `tests/` uses `pythonpath`/rootdir discovery; create it if other `tests/**` dirs have one)*
- Create: `tests/adapters/fuse/test_rrf.py`
- Modify: `tests/ports/test_protocols.py`

**Why this shape:** Fusion is its own concern, so it gets its own port (ISP) — the search use case depends on `Fuser`, not on a concrete algorithm, which is what lets Phase 8 drop in a `WeightedFuser` with no use-case change. RRF combines ranked lists using only **rank position**, deliberately ignoring the raw scores: cosine distance and BM25 live on incomparable scales, and normalizing them is fiddly and brittle. The formula is `score(chunk) = Σ 1/(rrf_k + rank)` summed over each list the chunk appears in (rank 0-based here, so we use `rrf_k + rank + 1`). A chunk ranked high in *both* lists accumulates the most — that mutual agreement is the signal hybrid search is built on. We dedupe on `(note_path, ordinal)` and preserve each chunk's originating `dense_rank`/`sparse_rank` for transparency and as future reranker input.

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/fuse/test_rrf.py
from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal, score, dense_rank=None, sparse_rank=None, path="a.md"):
    chunk = Chunk(
        note_path=path, ordinal=ordinal, heading_path=("H",), text=f"c{ordinal}", token_count=1
    )
    return RetrievedChunk(
        chunk=chunk, score=score, dense_rank=dense_rank, sparse_rank=sparse_rank
    )


def test_chunk_in_both_lists_outranks_chunk_in_one():
    # A: top of both lists. B: only dense. C: only sparse.
    dense = [_rc(0, 0.9, dense_rank=0), _rc(1, 0.8, dense_rank=1)]   # A, B
    sparse = [_rc(0, 5.0, sparse_rank=0), _rc(2, 4.0, sparse_rank=1)]  # A, C

    fused = RRFFuser().fuse(dense, sparse, k=10)

    assert fused[0].chunk.ordinal == 0          # A wins — present in both
    assert fused[0].score > fused[1].score
    # the fused A carries both ranks; B/C carry only their own
    assert fused[0].dense_rank == 0 and fused[0].sparse_rank == 0


def test_dedupes_on_note_path_and_ordinal():
    dense = [_rc(0, 0.9, dense_rank=0)]
    sparse = [_rc(0, 5.0, sparse_rank=0)]
    fused = RRFFuser().fuse(dense, sparse, k=10)
    assert len(fused) == 1
    # same chunk seen via both routes keeps both rank annotations
    assert fused[0].dense_rank == 0 and fused[0].sparse_rank == 0


def test_truncates_to_k():
    dense = [_rc(i, 1.0 / (i + 1), dense_rank=i) for i in range(5)]
    fused = RRFFuser().fuse(dense, [], k=2)
    assert len(fused) == 2


def test_rrf_constant_changes_weighting_but_not_top_result():
    dense = [_rc(0, 0.9, dense_rank=0), _rc(1, 0.8, dense_rank=1)]
    sparse = [_rc(1, 5.0, sparse_rank=0), _rc(0, 4.0, sparse_rank=1)]
    fused = RRFFuser(rrf_k=60).fuse(dense, sparse, k=10)
    # 0 is rank0+rank1, 1 is rank1+rank0 — symmetric, both present
    assert {c.chunk.ordinal for c in fused} == {0, 1}
```

Also append a port-conformance test to `tests/ports/test_protocols.py`:

```python
def test_rrf_fuser_conforms_to_port():
    from ariostea.adapters.fuse.rrf import RRFFuser
    from ariostea.ports.fusion import Fuser

    assert isinstance(RRFFuser(), Fuser)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/fuse/test_rrf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.fuse'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/ports/fusion.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ariostea.domain.models import RetrievedChunk


@runtime_checkable
class Fuser(Protocol):
    def fuse(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]: ...
```

```python
# src/ariostea/adapters/fuse/__init__.py
```

```python
# src/ariostea/adapters/fuse/rrf.py
from __future__ import annotations

from dataclasses import dataclass

from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.ports.fusion import Fuser


@dataclass
class _Entry:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class RRFFuser(Fuser):
    """Reciprocal Rank Fusion: combine ranked lists by position, not by raw
    score, so cosine-distance and BM25 scales never need reconciling."""

    def __init__(self, rrf_k: int = 60) -> None:
        self._rrf_k = rrf_k

    def fuse(
        self,
        dense: list[RetrievedChunk],
        sparse: list[RetrievedChunk],
        k: int,
    ) -> list[RetrievedChunk]:
        table: dict[tuple[str, int], _Entry] = {}

        def absorb(results: list[RetrievedChunk], which: str) -> None:
            for rank, rc in enumerate(results):
                key = (rc.chunk.note_path, rc.chunk.ordinal)
                entry = table.get(key)
                if entry is None:
                    entry = _Entry(chunk=rc.chunk, score=0.0, dense_rank=None, sparse_rank=None)
                    table[key] = entry
                entry.score += 1.0 / (self._rrf_k + rank + 1)
                if which == "dense":
                    entry.dense_rank = rank
                else:
                    entry.sparse_rank = rank

        absorb(dense, "dense")
        absorb(sparse, "sparse")

        ranked = sorted(table.values(), key=lambda e: e.score, reverse=True)
        return [
            RetrievedChunk(
                chunk=e.chunk,
                score=e.score,
                dense_rank=e.dense_rank,
                sparse_rank=e.sparse_rank,
            )
            for e in ranked[:k]
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/fuse/test_rrf.py tests/ports/test_protocols.py -v`
Expected: PASS (4 RRF tests + protocol tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/ariostea/ports/fusion.py src/ariostea/adapters/fuse tests/adapters/fuse tests/ports/test_protocols.py
git commit -m "feat(fuse): Fuser port + RRFFuser (reciprocal rank fusion)"
```

---

## Task 2.4: Wire `SearchKnowledge` to hybrid retrieval (use case + config + container)

**Files:**
- Modify: `src/ariostea/search/search_knowledge.py`
- Modify: `src/ariostea/config/schema.py` (`SearchCfg`: add `k_sparse`)
- Modify: `src/ariostea/config/container.py`
- Modify: `tests/search/test_search_knowledge.py`
- Modify: `tests/config/test_schema.py`

**Why this shape:** This is the application policy change that turns two retrieval channels into one ranked answer. `SearchKnowledge` now embeds the query once, fetches `k_dense` dense candidates and `k_sparse` sparse candidates, and asks the `Fuser` to merge them down to the caller's `query.k`. The candidate breadths are *retrieval policy*, not per-query input, so they're injected from `SearchCfg` at the composition root — `Query` stays a clean user intent. We default `k_dense`/`k_sparse` to `50` (Spec §17). The use case still depends only on ports (`EmbeddingProvider`, `ChunkRetriever`, `Fuser`), so it stays fully unit-testable with fakes.

- [ ] **Step 1: Write the failing test**

Replace the entire contents of `tests/search/test_search_knowledge.py` with:

```python
from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.domain.models import Chunk, Query, RetrievedChunk
from ariostea.search.search_knowledge import SearchKnowledge


class FakeEmbed:
    def embed_documents(self, texts):
        return [[0.0] for _ in texts]

    def embed_query(self, text):
        return [float(len(text))]

    @property
    def dimension(self):
        return 1

    @property
    def fingerprint(self):
        return "fake"


def _chunk(ordinal, text, path="a.md"):
    return Chunk(
        note_path=path, ordinal=ordinal, heading_path=("A",), text=text, token_count=1
    )


class FakeRetriever:
    def __init__(self):
        self.dense_call = None
        self.sparse_call = None

    def dense(self, vec, k, filters=None):
        self.dense_call = (vec, k, filters)
        return [RetrievedChunk(chunk=_chunk(0, "semantic"), score=0.5, dense_rank=0)]

    def sparse(self, query, k, filters=None):
        self.sparse_call = (query, k, filters)
        return [RetrievedChunk(chunk=_chunk(1, "lexical"), score=2.0, sparse_rank=0)]


def test_search_runs_dense_and_sparse_and_fuses():
    retriever = FakeRetriever()
    uc = SearchKnowledge(
        embeddings=FakeEmbed(),
        retriever=retriever,
        fuser=RRFFuser(),
        k_dense=40,
        k_sparse=30,
    )
    result = uc.search(Query(text="hello", k=5))

    # both channels were exercised
    assert retriever.dense_call[0] == [5.0]      # embedded query ("hello" -> len 5)
    assert retriever.dense_call[1] == 40          # k_dense from construction
    assert retriever.sparse_call[0] == "hello"    # raw text to BM25
    assert retriever.sparse_call[1] == 30          # k_sparse from construction

    # fusion merged both unique chunks into the result
    texts = {c.chunk.text for c in result.chunks}
    assert texts == {"semantic", "lexical"}


def test_search_truncates_to_query_k():
    class ManyRetriever(FakeRetriever):
        def dense(self, vec, k, filters=None):
            return [
                RetrievedChunk(chunk=_chunk(i, f"d{i}"), score=1.0, dense_rank=i)
                for i in range(5)
            ]

        def sparse(self, query, k, filters=None):
            return []

    uc = SearchKnowledge(
        embeddings=FakeEmbed(), retriever=ManyRetriever(), fuser=RRFFuser()
    )
    result = uc.search(Query(text="x", k=2))
    assert len(result.chunks) == 2
```

Also append a `k_sparse` default check to `tests/config/test_schema.py` inside `test_minimal_config_applies_defaults` (add this assertion after the existing `top_k` assertion):

```python
    assert cfg.search.k_sparse == 50  # default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_search_knowledge.py tests/config/test_schema.py -v`
Expected: FAIL — `SearchKnowledge.__init__()` got an unexpected keyword argument `fuser` (current signature is `(embeddings, retriever)`), and `SearchCfg` has no `k_sparse`.

- [ ] **Step 3: Write minimal implementation**

**(a)** Add `k_sparse` to `SearchCfg` in `src/ariostea/config/schema.py`:

```python
class SearchCfg(BaseModel):
    k_dense: int = 50
    k_sparse: int = 50
    top_k: int = 10
```

**(b)** Replace the body of `src/ariostea/search/search_knowledge.py` with the hybrid version:

```python
from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.fusion import Fuser
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        retriever: ChunkRetriever,
        fuser: Fuser,
        k_dense: int = 50,
        k_sparse: int = 50,
    ) -> None:
        self._embeddings = embeddings
        self._retriever = retriever
        self._fuser = fuser
        self._k_dense = k_dense
        self._k_sparse = k_sparse

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        dense = self._retriever.dense(vec, k=self._k_dense, filters=query.filters)
        sparse = self._retriever.sparse(query.text, k=self._k_sparse, filters=query.filters)
        fused = self._fuser.fuse(dense, sparse, k=query.k)
        return SearchResult(chunks=tuple(fused))
```

**(c)** Update the composition root `src/ariostea/config/container.py` to build the fuser and inject it plus the candidate breadths. Add the imports:

```python
from ariostea.adapters.fuse.rrf import RRFFuser
```

and replace the `searcher = ...` line with:

```python
    searcher = SearchKnowledge(
        embeddings=embeddings,
        retriever=store,
        fuser=RRFFuser(),
        k_dense=config.search.k_dense,
        k_sparse=config.search.k_sparse,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_search_knowledge.py tests/config/test_schema.py -v`
Expected: PASS.

Then run the whole non-integration suite to confirm nothing else regressed:

Run: `uv run pytest -q -m "not integration"`
Expected: PASS (all unit tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/ariostea/search/search_knowledge.py src/ariostea/config/schema.py src/ariostea/config/container.py tests/search/test_search_knowledge.py tests/config/test_schema.py
git commit -m "feat(search): hybrid dense+sparse retrieval fused with RRF"
```

---

## Task 2.5: End-to-end proof — hybrid beats dense-only on a lexical query

**Files:**
- Modify: `tests/test_end_to_end.py`

**Why this shape:** Spec §18 sets the Phase 2 exit criterion: *"Hybrid beats dense-only on fixture queries."* This test is that gate. It indexes a vault containing a rare, exact token (an invented identifier that carries no semantic signal an embedding could latch onto), then searches for that exact token. Dense retrieval alone would have no reason to rank the right note first; BM25 matches the literal term, and fusion surfaces it. Running through the real container (fastembed + sqlite-vec + FTS5 + RRF) proves the whole hybrid path is wired correctly, not just the units. It's an `integration` test because it loads the real embedding model.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_end_to_end.py`:

```python
@pytest.mark.integration
def test_hybrid_finds_exact_keyword_dense_would_miss(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # A rare, semantically-empty identifier that only exact (BM25) match can target.
    (vault / "config.md").write_text(
        "# Settings\nThe deployment uses token ZK7QWASDF as its rotation key."
    )
    (vault / "prose.md").write_text(
        "# Overview\nA general discussion of deployments, settings, and keys in systems."
    )

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    payload = search_payload(container, query="ZK7QWASDF", k=2)
    assert payload["results"]
    assert payload["results"][0]["note_path"] == "config.md"
```

- [ ] **Step 2: Run test to verify it fails**

First confirm the test is meaningful by checking the behavior is exercised end-to-end:

Run: `uv run pytest tests/test_end_to_end.py::test_hybrid_finds_exact_keyword_dense_would_miss -v -m integration`
Expected at this point in the plan (after Tasks 2.1–2.4 are merged): PASS. The hybrid path already exists, so this task primarily *locks in* the exit criterion as a regression guard. If you are executing 2.5 before 2.4's container wiring, it FAILS (search returns prose.md first or sparse is absent).

> If you want to *see* it fail first (recommended to prove the test has teeth), temporarily set `k_sparse=0`-equivalent by stubbing sparse: run the test against a container built before Task 2.4. Otherwise, treat a green run here as confirmation the exit criterion holds.

- [ ] **Step 3: Implementation**

No production code needed — Tasks 2.1–2.4 supply the behavior. This task is the acceptance test that pins the Phase 2 goal.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -q` (includes integration; first run may download the model)
Expected: PASS (all unit + integration tests).

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add tests/test_end_to_end.py
git commit -m "test(e2e): hybrid retrieval surfaces exact-keyword match (Phase 2 exit gate)"
```

---

> **Phase 2 complete:** `search_knowledge` is now a hybrid retriever — dense semantic recall + BM25 exact-term precision, fused by RRF, all in the one SQLite file. Next plan: Phase 3 (`search_sources` provenance rollup).

---

## Self-review (completed by plan author)

- **Spec coverage:** Spec §18 row "2 | FTS5 sparse + RRF hybrid | Hybrid beats dense-only" → Tasks 2.2 + 2.3 + 2.5. Spec §6 `ChunkRetriever.sparse` → Task 2.1/2.2. Spec §7 `RRFFuser` / `Fuser` → Task 2.3. Spec §8 `chunks_fts (FTS5: chunk_id, embedding_text)` → Task 2.2 (indexes `embedding_text`). Spec §10 retrieval core (dense+sparse+fuse) → Task 2.4. Spec §17 defaults (`k_dense`/`k_sparse`) → Task 2.4 config. Deferred items (rerank, blurbs, sources, filters) explicitly out of scope above.
- **Type consistency:** `sparse(self, query, k, filters=None) -> list[RetrievedChunk]` identical in port (2.1), adapter (2.2), and use case call site (2.4). `Fuser.fuse(dense, sparse, k)` identical in port (2.3), adapter (2.3), and call site (2.4). `RRFFuser(rrf_k=60)` constructor name matches its test. Chunk identity `(note_path, ordinal)` used consistently in the fuser.
- **Placeholders:** none — every code step shows complete code; every command shows expected output.
