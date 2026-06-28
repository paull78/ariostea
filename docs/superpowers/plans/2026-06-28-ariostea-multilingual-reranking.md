# Multilingual Reranking (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multilingual cross-encoder reranking stage that re-scores the fused candidate pool by true query↔passage relevance, closing the cross-lingual ranking gap the eval harness measured (cross-lingual `recall@1 = 0.000`, `MRR@5 = 0.500`).

**Architecture:** New `Reranker` port with two adapters — `FastEmbedReranker` (a multilingual ONNX cross-encoder, default `jinaai/jina-reranker-v2-base-multilingual`) and `NoopReranker` (identity passthrough, the opt-out and test double). `RRFFuser` is demoted to a recall gatherer (fuses a large `pool`), and `SearchKnowledge` calls the reranker to pick the final `top_k`. The container builds the reranker from a new `[rerank]` config section and falls back to `NoopReranker` with a logged warning if the model is unavailable (LSP-safe degradation).

**Tech Stack:** Python 3.12, `fastembed` (`TextCrossEncoder`), pytest (`integration` marker for the model-loading test), pydantic config.

**Source spec:** [`docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md`](../specs/2026-06-27-ariostea-multilingual-retrieval-design.md) §4.2 — this plan implements **Component 2**.

---

## Notes that shaped this plan (verified against the code)

- The design named `bge-reranker-v2-m3`, but it is **not** in this fastembed version's supported list. The available multilingual cross-encoder is **`jinaai/jina-reranker-v2-base-multilingual`** (the design's named alternative). We use it as the default.
- `fastembed.rerank.cross_encoder.TextCrossEncoder` exposes `rerank(query: str, documents: Iterable[str]) -> Iterable[float]` — one relevance score per document (higher = better). The adapter sorts candidates by that score.
- `RetrievedChunk` is a frozen dataclass; use `dataclasses.replace(rc, score=...)` to produce a re-scored copy.
- Only `tests/test_end_to_end.py` and `tests/eval/test_harness_integration.py` build the real container, and **both are `@pytest.mark.integration`** — so changing `SearchKnowledge`'s constructor + the container does not affect the fast suite (we update `tests/search/test_search_knowledge.py` in the same task).

## File Structure

- Create: `src/ariostea/ports/rerank.py` — `Reranker` Protocol.
- Create: `src/ariostea/adapters/rerank/__init__.py` — package marker.
- Create: `src/ariostea/adapters/rerank/noop.py` — `NoopReranker`.
- Create: `src/ariostea/adapters/rerank/fastembed_rerank.py` — `FastEmbedReranker`.
- Modify: `src/ariostea/config/schema.py` — add `RerankCfg`, wire into `Config`.
- Modify: `src/ariostea/search/search_knowledge.py` — add `reranker` + `pool`, change the flow.
- Modify: `src/ariostea/config/container.py` — build the reranker with fallback; inject it.
- Modify: `docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md` — record the acceptance result.
- Test: `tests/ports/test_protocols.py` (append), `tests/adapters/rerank/test_noop.py`, `tests/adapters/rerank/test_fastembed_rerank.py`, `tests/search/test_search_knowledge.py` (rewrite), `tests/config/test_schema.py` (append).

---

### Task 1: Reranker port

**Files:**
- Create: `src/ariostea/ports/rerank.py`
- Test: `tests/ports/test_protocols.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/ports/test_protocols.py`:

```python
def test_reranker_protocol_is_runtime_checkable():
    from ariostea.domain.models import RetrievedChunk
    from ariostea.ports.rerank import Reranker

    class FakeReranker:
        def rerank(
            self, query: str, candidates: list[RetrievedChunk], top_n: int
        ) -> list[RetrievedChunk]:
            return list(candidates[:top_n])

    assert isinstance(FakeReranker(), Reranker)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ports/test_protocols.py::test_reranker_protocol_is_runtime_checkable -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.ports.rerank'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ariostea/ports/rerank.py`:

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ariostea.domain.models import RetrievedChunk


@runtime_checkable
class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ports/test_protocols.py::test_reranker_protocol_is_runtime_checkable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/ports/rerank.py tests/ports/test_protocols.py
git commit -m "feat(rerank): Reranker port"
```

---

### Task 2: NoopReranker adapter

**Files:**
- Create: `src/ariostea/adapters/rerank/__init__.py`
- Create: `src/ariostea/adapters/rerank/noop.py`
- Test: `tests/adapters/rerank/test_noop.py`

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/rerank/test_noop.py`:

```python
from ariostea.adapters.rerank.noop import NoopReranker
from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal):
    chunk = Chunk(
        note_path="a.md", ordinal=ordinal, heading_path=("H",), text=f"c{ordinal}", token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=0.0, dense_rank=ordinal)


def test_noop_preserves_order_and_truncates():
    candidates = [_rc(0), _rc(1), _rc(2)]
    out = NoopReranker().rerank("any query", candidates, top_n=2)
    assert [rc.chunk.ordinal for rc in out] == [0, 1]


def test_noop_handles_empty():
    assert NoopReranker().rerank("q", [], top_n=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/rerank/test_noop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.rerank'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ariostea/adapters/rerank/__init__.py` (empty):

```python
```

Create `src/ariostea/adapters/rerank/noop.py`:

```python
from __future__ import annotations

from ariostea.domain.models import RetrievedChunk
from ariostea.ports.rerank import Reranker


class NoopReranker(Reranker):
    """Identity reranker: keep the fused order, just truncate to top_n.

    Used when reranking is disabled or the model is unavailable, and as a
    deterministic test double.
    """

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:top_n])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/rerank/test_noop.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/adapters/rerank/__init__.py src/ariostea/adapters/rerank/noop.py tests/adapters/rerank/test_noop.py
git commit -m "feat(rerank): NoopReranker identity adapter"
```

---

### Task 3: FastEmbedReranker adapter

**Files:**
- Create: `src/ariostea/adapters/rerank/fastembed_rerank.py`
- Test: `tests/adapters/rerank/test_fastembed_rerank.py`

- [ ] **Step 1: Write the failing test**

Create `tests/adapters/rerank/test_fastembed_rerank.py`:

```python
import pytest

from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal, text):
    chunk = Chunk(
        note_path=f"{ordinal}.md",
        ordinal=ordinal,
        heading_path=("H",),
        text=text,
        token_count=1,
    )
    # Deliberately bad fused order: the relevant chunk starts last.
    return RetrievedChunk(chunk=chunk, score=0.0, dense_rank=ordinal)


@pytest.mark.integration
def test_fastembed_reranker_promotes_relevant_passage():
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    candidates = [
        _rc(0, "A recipe for boiling pasta with salt and water."),
        _rc(1, "The weather forecast predicts rain over the weekend."),
        _rc(2, "Rolling dice and moving tokens on a board game."),
    ]
    out = FastEmbedReranker().rerank("how do board games use dice", candidates, top_n=2)

    assert len(out) == 2
    # The dice passage must be promoted to the top despite starting last.
    assert out[0].chunk.text.startswith("Rolling dice")
    # Scores are reranker relevance scores in descending order.
    assert out[0].score >= out[1].score
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/rerank/test_fastembed_rerank.py -v -m integration`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.rerank.fastembed_rerank'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ariostea/adapters/rerank/fastembed_rerank.py`:

```python
from __future__ import annotations

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from ariostea.domain.models import RetrievedChunk
from ariostea.ports.rerank import Reranker


class FastEmbedReranker(Reranker):
    """Multilingual cross-encoder reranker (ONNX via fastembed).

    Scores each candidate passage against the query and returns the top_n by
    relevance. The default model is multilingual on purpose: an English-only
    cross-encoder would score cross-lingual passages low and defeat the point.
    """

    def __init__(self, model_name: str = "jinaai/jina-reranker-v2-base-multilingual") -> None:
        self._model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [rc.chunk.text for rc in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(rc, score=float(score)) for rc, score in ranked[:top_n]]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/rerank/test_fastembed_rerank.py -v -m integration`
Expected: PASS (1 passed). First run downloads the reranker model (~280MB).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/adapters/rerank/fastembed_rerank.py tests/adapters/rerank/test_fastembed_rerank.py
git commit -m "feat(rerank): FastEmbedReranker multilingual cross-encoder"
```

---

### Task 4: Rerank config section

**Files:**
- Modify: `src/ariostea/config/schema.py`
- Test: `tests/config/test_schema.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/config/test_schema.py`:

```python
def test_rerank_defaults(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.enabled is True
    assert cfg.rerank.model == "jinaai/jina-reranker-v2-base-multilingual"
    assert cfg.rerank.pool == 100


def test_rerank_can_be_disabled(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[rerank]\nenabled = false\npool = 40\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.enabled is False
    assert cfg.rerank.pool == 40
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_schema.py::test_rerank_defaults -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'rerank'`

- [ ] **Step 3: Write minimal implementation**

In `src/ariostea/config/schema.py`, add the `RerankCfg` class after `SearchCfg`:

```python
class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "jinaai/jina-reranker-v2-base-multilingual"
    pool: int = 100  # candidates fused before reranking selects the final top_k
```

Then add the field to `Config` (the class becomes):

```python
class Config(BaseModel):
    vault: VaultCfg
    embedding: EmbeddingCfg = EmbeddingCfg()
    store: StoreCfg = StoreCfg()
    search: SearchCfg = SearchCfg()
    rerank: RerankCfg = RerankCfg()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_schema.py -v`
Expected: PASS (all schema tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/config/schema.py tests/config/test_schema.py
git commit -m "feat(rerank): [rerank] config section"
```

---

### Task 5: Wire reranking into the pipeline

This is the integration task: `SearchKnowledge` gains the reranker + pool, and the container builds and injects it with degradation fallback. Done together so the constructor change and its only caller move as one.

**Files:**
- Modify: `src/ariostea/search/search_knowledge.py`
- Modify: `src/ariostea/config/container.py`
- Test: `tests/search/test_search_knowledge.py` (rewrite)

- [ ] **Step 1: Rewrite the unit test (failing)**

Replace the entire contents of `tests/search/test_search_knowledge.py` with:

```python
from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.adapters.rerank.noop import NoopReranker
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
    return Chunk(note_path=path, ordinal=ordinal, heading_path=("A",), text=text, token_count=1)


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
        reranker=NoopReranker(),
        k_dense=40,
        k_sparse=30,
    )
    result = uc.search(Query(text="hello", k=5))

    assert retriever.dense_call[0] == [5.0]  # embedded query ("hello" -> len 5)
    assert retriever.dense_call[1] == 40  # k_dense from construction
    assert retriever.sparse_call[0] == "hello"  # raw text to BM25
    assert retriever.sparse_call[1] == 30  # k_sparse from construction

    texts = {c.chunk.text for c in result.chunks}
    assert texts == {"semantic", "lexical"}


def test_search_truncates_to_query_k():
    class ManyRetriever(FakeRetriever):
        def dense(self, vec, k, filters=None):
            return [
                RetrievedChunk(chunk=_chunk(i, f"d{i}"), score=1.0, dense_rank=i) for i in range(5)
            ]

        def sparse(self, query, k, filters=None):
            return []

    uc = SearchKnowledge(
        embeddings=FakeEmbed(),
        retriever=ManyRetriever(),
        fuser=RRFFuser(),
        reranker=NoopReranker(),
    )
    result = uc.search(Query(text="x", k=2))
    assert len(result.chunks) == 2


class ReverseReranker:
    """Test double: reverses candidate order, then truncates — proves the use
    case actually applies the reranker rather than returning fused order."""

    def rerank(self, query, candidates, top_n):
        return list(reversed(candidates))[:top_n]


def test_search_applies_reranker_then_truncates():
    class ManyRetriever(FakeRetriever):
        def dense(self, vec, k, filters=None):
            return [
                RetrievedChunk(chunk=_chunk(i, f"d{i}"), score=1.0, dense_rank=i) for i in range(4)
            ]

        def sparse(self, query, k, filters=None):
            return []

    uc = SearchKnowledge(
        embeddings=FakeEmbed(),
        retriever=ManyRetriever(),
        fuser=RRFFuser(),
        reranker=ReverseReranker(),
        pool=10,
    )
    result = uc.search(Query(text="x", k=2))
    # Fused order by RRF is d0,d1,d2,d3; reversed is d3,d2,...; top_n=2 -> d3,d2.
    assert [c.chunk.text for c in result.chunks] == ["d3", "d2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_search_knowledge.py -v`
Expected: FAIL — `TypeError: SearchKnowledge.__init__() got an unexpected keyword argument 'reranker'`

- [ ] **Step 3: Modify SearchKnowledge**

Replace the entire contents of `src/ariostea/search/search_knowledge.py` with:

```python
from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.fusion import Fuser
from ariostea.ports.rerank import Reranker
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        retriever: ChunkRetriever,
        fuser: Fuser,
        reranker: Reranker,
        k_dense: int = 50,
        k_sparse: int = 50,
        pool: int = 100,
    ) -> None:
        self._embeddings = embeddings
        self._retriever = retriever
        self._fuser = fuser
        self._reranker = reranker
        self._k_dense = k_dense
        self._k_sparse = k_sparse
        self._pool = pool

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        dense = self._retriever.dense(vec=vec, k=self._k_dense, filters=query.filters)
        sparse = self._retriever.sparse(query=query.text, k=self._k_sparse, filters=query.filters)
        # RRF is a recall gatherer: fuse a large pool, then let the reranker
        # pick the final top_k by true query-passage relevance.
        fused = self._fuser.fuse(dense=dense, sparse=sparse, k=self._pool)
        ranked = self._reranker.rerank(query.text, fused, top_n=query.k)
        return SearchResult(chunks=tuple(ranked))
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run pytest tests/search/test_search_knowledge.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Modify the container to build and inject the reranker**

Replace the entire contents of `src/ariostea/config/container.py` with:

```python
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker
from ariostea.adapters.rerank.noop import NoopReranker
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.schema import Config, RerankCfg
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.rerank import Reranker
from ariostea.ports.store import DocumentReader, IndexAdmin
from ariostea.search.search_knowledge import SearchKnowledge
from ariostea.search.search_sources import SearchSources

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """Assembled application: config, the use cases consumers call, and the
    admin port for status. Concrete adapters (embeddings, store) are wiring
    internals of build_container and are deliberately not exposed here."""

    config: Config
    indexer: IndexVault
    searcher: SearchKnowledge
    admin: IndexAdmin
    sources: SearchSources
    reader: DocumentReader


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _build_reranker(cfg: RerankCfg) -> Reranker:
    """Build the configured reranker, degrading to NoopReranker (fused order)
    with a warning if the model cannot be loaded — a degraded ranking, never a
    failed search."""
    if not cfg.enabled:
        return NoopReranker()
    try:
        return FastEmbedReranker(model_name=cfg.model)
    except Exception as exc:  # model missing/offline/unsupported
        logger.warning("reranker unavailable (%s); falling back to fused order", exc)
        return NoopReranker()


def build_container(config: Config) -> Container:
    # Embedding provider — local fastembed for the walking skeleton.
    embeddings: EmbeddingProvider = FastEmbedEmbeddings(model_name=config.embedding.local_model)

    store_path = _expand(config.store.path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path=store_path, dim=embeddings.dimension)

    parser = ObsidianMarkdownParser()
    chunker = HeadingAwareChunker()

    # The store is injected into each use case as its narrow role
    # (DocumentWriter for indexing, ChunkRetriever for search); only its
    # IndexAdmin face is re-exposed on the Container for status.
    indexer = IndexVault(parser=parser, chunker=chunker, embeddings=embeddings, store=store)
    searcher = SearchKnowledge(
        embeddings=embeddings,
        retriever=store,
        fuser=RRFFuser(),
        reranker=_build_reranker(config.rerank),
        k_dense=config.search.k_dense,
        k_sparse=config.search.k_sparse,
        pool=config.rerank.pool,
    )
    sources = SearchSources(searcher=searcher, reader=store)

    return Container(
        config=config,
        indexer=indexer,
        searcher=searcher,
        admin=store,
        sources=sources,
        reader=store,
    )
```

- [ ] **Step 6: Run the fast suite to confirm nothing else broke**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS, integration tests deselected.

- [ ] **Step 7: Run the end-to-end integration tests (real models, incl. reranker)**

Run: `uv run pytest tests/test_end_to_end.py -m integration -q`
Expected: PASS. First run downloads the reranker model.

> **Watch the exact-keyword test.** `test_hybrid_finds_exact_keyword_dense_would_miss`
> asserts the `ZK7QWASDF` identifier note ranks #1. A cross-encoder judges *semantic*
> relevance and may not honor a literal-token match the way BM25 does. If reranking
> demotes that note, **do not silently weaken the assertion** — stop and surface it: it is
> a genuine reranking-vs-lexical tradeoff (rerankers can hurt identifier/code lookup) worth
> a decision (e.g. keep a lexical escape hatch, or accept the tradeoff). In practice modern
> rerankers attend to literal overlap and usually keep it #1, but verify.

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/search/search_knowledge.py src/ariostea/config/container.py tests/search/test_search_knowledge.py
git commit -m "feat(rerank): wire multilingual reranking into the search pipeline"
```

---

### Task 6: Acceptance gate — re-run the eval, record the result

The design's acceptance is "measurable recall@k / MRR gain on `en→it` and `it→en`, no regression on `same`." We now measure it.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md`

- [ ] **Step 1: Run the eval harness with reranking active**

The runner builds the container from default config, so reranking is now on automatically.

Run: `uv run python eval/run_eval.py 1`
Then run: `uv run python eval/run_eval.py`
Record both tables. Expected direction: cross-lingual `recall@1` rises above `0.000` and/or cross-lingual `MRR` rises above `0.500`, while `same` does not regress (recall@1 stays ≥ 0.750).

- [ ] **Step 2: Confirm the integration eval test still passes**

Run: `uv run pytest tests/eval/test_harness_integration.py -m integration -q`
Expected: PASS (`same` recall@3 still 1.0).

- [ ] **Step 3: Record the acceptance result in the design doc**

In `docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md`, find the end of section `### 4.2 Multilingual reranking` (the line ending "...no regression on `same`."). Append a new line immediately after it, filling in the **actual measured numbers** from Step 1:

```markdown

> **Result (measured 2026-06-28, `jina-reranker-v2-base-multilingual`):** cross-lingual
> `recall@1` rose from 0.000 to <MEASURED>, `MRR@5` from 0.500 to <MEASURED>; `same`
> held at recall@1 <MEASURED> (no regression). Default model is Jina (the design's named
> alternative) because `bge-reranker-v2-m3` is absent from the installed fastembed.
```

Replace each `<MEASURED>` with the real value observed in Step 1. (This step has no code; it records the outcome that gates the phase.)

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md
git commit -m "docs(rerank): record cross-lingual eval result for reranking"
```

---

## Notes for the implementer

- **Why Jina, not bge-reranker-v2-m3:** the design named bge-m3, but it isn't in the installed fastembed's supported models; `jinaai/jina-reranker-v2-base-multilingual` is, and it is genuinely multilingual (the design's stated fallback). If a future fastembed adds bge-m3, switching is a one-line config change — that's the whole point of the port.
- **Why the reranker is injected, not defaulted inside SearchKnowledge:** keeping the use case dependent on the `Reranker` *port* (never a concrete adapter) preserves the dependency rule and lets tests pass `NoopReranker`/fakes. The container is the only place that names concrete rerankers.
- **Why fallback lives in the container:** model-load failure is a wiring/environment concern, so the composition root handles it (try `FastEmbedReranker`, except → `NoopReranker` + warning). The use case stays oblivious — it just calls `reranker.rerank`.
- **Pool vs top_k:** `pool` (default 100) is how many fused candidates the reranker sees; `query.k` is the final count. On the tiny eval fixture the pool holds every chunk, so the reranker can freely reorder — which is exactly what lets cross-lingual numbers move.
```
