# Blurb-aware reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the cross-encoder reranker score a chunk's context blurb together with its text, then measure blurbs and blurb-aware reranking on the wiki gold set.

**Architecture:** The blurb already sits in `chunks.context_blurb`. It gets a field on `RetrievedChunk`, which the store fills and RRF fusion keeps. `FastEmbedReranker` prepends it to the passage when `RerankCfg.use_context` is set. A new eval runner indexes the wiki corpus once with blurbs, checks every note got one, and scores arms 2 and 3 over that one index.

**Tech Stack:** Python 3.12, sqlite + sqlite-vec + FTS5, fastembed (ONNX embedding and cross-encoder), pydantic config, pytest, LM Studio for blurb generation.

Implements `docs/design/2026-09-29-blurb-aware-reranking.md`. Branch
`feat/blurb-aware-rerank`.

**Fast suite:** `uv run pytest -m "not integration" -q`. Run it before every
commit. The full suite (about 12 minutes) runs once, in Task 8.

---

### Task 1: `RetrievedChunk.context_blurb`

**Files:**
- Modify: `src/ariostea/domain/models.py` (`RetrievedChunk`)
- Test: `tests/domain/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
def test_retrieved_chunk_blurb_defaults_to_none():
    chunk = Chunk(note_path="a.md", ordinal=0, heading_path=(), text="t", token_count=1)
    rc = RetrievedChunk(chunk=chunk, score=1.0)
    assert rc.context_blurb is None
    assert RetrievedChunk(chunk=chunk, score=1.0, context_blurb="b").context_blurb == "b"
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/domain/test_models.py -q`
Expected: FAIL, `unexpected keyword argument 'context_blurb'` or `AttributeError`.

- [ ] **Step 3: Add the field**

```python
@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    # The note-level blurb from contextual indexing, for a reranker that
    # should judge the chunk with the context the first stages saw.
    context_blurb: str | None = None
```

- [ ] **Step 4: Run it and see it pass**

Run: `uv run pytest tests/domain/test_models.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/domain/models.py tests/domain/test_models.py
git commit -m "feat(domain): a retrieved chunk can carry its context blurb"
```

### Task 2: the store returns the blurb

**Files:**
- Modify: `src/ariostea/adapters/store/sqlite_store.py` (`dense()` and `sparse()`)
- Test: `tests/adapters/store/test_sqlite_store.py`

- [ ] **Step 1: Write the failing tests**

The file's `_cchunk` helper always writes `context_blurb=None`. Give it an
optional blurb and keep existing callers unchanged:

```python
def _cchunk(note, ordinal, text, blurb=None):
    chunk = Chunk(
        note_path=note.path,
        ordinal=ordinal,
        heading_path=("A",),
        text=text,
        token_count=len(text.split()),
    )
    embedding_text = f"{blurb}\n\n{text}" if blurb else text
    return ContextualizedChunk(chunk=chunk, context_blurb=blurb, embedding_text=embedding_text)


def test_dense_and_sparse_return_the_stored_blurb(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha", blurb="About Greek letters.")], [[1.0, 0.0, 0.0]])

    dense = store.dense([1.0, 0.0, 0.0], k=1)
    sparse = store.sparse("alpha", k=1)

    assert dense[0].context_blurb == "About Greek letters."
    assert sparse[0].context_blurb == "About Greek letters."
    assert dense[0].chunk.text == "alpha"  # the raw text stays raw


def test_plain_index_returns_no_blurb(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "alpha")], [[1.0, 0.0, 0.0]])

    assert store.dense([1.0, 0.0, 0.0], k=1)[0].context_blurb is None
    assert store.sparse("alpha", k=1)[0].context_blurb is None
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -q`
Expected: the two new tests FAIL (`context_blurb` is `None` for the blurbed chunk).

- [ ] **Step 3: Select and set the blurb**

In `dense()`, extend the select list:

```sql
SELECT c.note_id, n.path AS note_path, c.ordinal, c.heading_path,
       c.text, c.token_count, c.context_blurb, knn.distance
```

In `sparse()`:

```sql
SELECT n.path as note_path, c.ordinal, c.heading_path, c.text, c.token_count,
       c.context_blurb, bm.bm
```

In both result loops, pass `context_blurb=r["context_blurb"]` to `RetrievedChunk(...)`.

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -q`
Expected: PASS.

- [ ] **Step 5: Fast suite, then commit**

```bash
uv run pytest -m "not integration" -q
git add src/ariostea/adapters/store/sqlite_store.py tests/adapters/store/test_sqlite_store.py
git commit -m "feat(store): return each chunk's context blurb with search results"
```

### Task 3: RRF keeps the blurb

**Files:**
- Modify: `src/ariostea/adapters/fuse/rrf.py` (`_Entry`, `fuse`)
- Test: `tests/adapters/fuse/test_rrf.py`

`fuse()` builds new `RetrievedChunk`s from an internal `_Entry`, so without this
task the blurb is dropped silently between the store and the reranker.

- [ ] **Step 1: Write the failing test**

```python
def test_fused_chunks_keep_their_blurb_from_either_channel():
    def with_blurb(rc, blurb):
        return RetrievedChunk(
            chunk=rc.chunk,
            score=rc.score,
            dense_rank=rc.dense_rank,
            sparse_rank=rc.sparse_rank,
            context_blurb=blurb,
        )

    dense = [with_blurb(_rc(0, 0.9, dense_rank=0), "note A")]  # A, dense only
    sparse = [with_blurb(_rc(1, 5.0, sparse_rank=0), "note B")]  # B, sparse only

    fused = {rc.chunk.ordinal: rc for rc in RRFFuser().fuse(dense, sparse, k=10)}

    assert fused[0].context_blurb == "note A"
    assert fused[1].context_blurb == "note B"
```

- [ ] **Step 2: Run it and see it fail**

Run: `uv run pytest tests/adapters/fuse/test_rrf.py -q`
Expected: FAIL, `None != 'note A'`.

- [ ] **Step 3: Carry it through**

Read `_Entry` at the top of `rrf.py` and add a `context_blurb: str | None`
field to it. In `absorb`, create the entry with it:

```python
entry = _Entry(
    chunk=rc.chunk,
    score=0.0,
    dense_rank=None,
    sparse_rank=None,
    context_blurb=rc.context_blurb,
)
```

The first result that creates the entry sets it. Dense and sparse read the
same `chunks` row, so there is nothing to reconcile. In the return
comprehension add `context_blurb=e.context_blurb`.

- [ ] **Step 4: Run it and see it pass**

Run: `uv run pytest tests/adapters/fuse/test_rrf.py -q`
Expected: PASS.

- [ ] **Step 5: Fast suite, then commit**

```bash
uv run pytest -m "not integration" -q
git add src/ariostea/adapters/fuse/rrf.py tests/adapters/fuse/test_rrf.py
git commit -m "fix(fuse): keep the context blurb through RRF fusion"
```

### Task 4: the reranker can score blurb plus text

**Files:**
- Modify: `src/ariostea/adapters/rerank/fastembed_rerank.py`
- Test: `tests/adapters/rerank/test_fastembed_rerank.py`

The tests swap `TextCrossEncoder` for a fake that records what it was asked to
score, so they need no model download and are not `integration`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/adapters/rerank/test_fastembed_rerank.py`:

```python
class _RecordingEncoder:
    """Stands in for fastembed's cross-encoder: scores passages by position
    (later is better) and remembers what it was given."""

    def __init__(self, model_name):
        self.seen: list[str] = []

    def rerank(self, query, documents):
        self.seen = list(documents)
        return [float(i) for i in range(len(documents))]


@pytest.fixture
def fake_encoder(monkeypatch):
    from ariostea.adapters.rerank import fastembed_rerank

    monkeypatch.setattr(fastembed_rerank, "TextCrossEncoder", _RecordingEncoder)


def _blurbed(ordinal, text, blurb):
    rc = _rc(ordinal, text)
    return RetrievedChunk(chunk=rc.chunk, score=0.0, dense_rank=ordinal, context_blurb=blurb)


def test_use_context_scores_blurb_then_text(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker(use_context=True)
    reranker.rerank("q", [_blurbed(0, "It has four strings.", "The double bass.")], top_n=1)

    assert reranker._model.seen == ["The double bass.\n\nIt has four strings."]


def test_use_context_falls_back_to_text_without_a_blurb(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker(use_context=True)
    reranker.rerank("q", [_blurbed(0, "plain", None), _blurbed(1, "empty", "")], top_n=2)

    assert reranker._model.seen == ["plain", "empty"]


def test_default_scores_raw_text_and_returns_it_unchanged(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker()
    out = reranker.rerank("q", [_blurbed(0, "first", "B"), _blurbed(1, "second", "B")], top_n=2)

    assert reranker._model.seen == ["first", "second"]
    assert [rc.chunk.text for rc in out] == ["second", "first"]  # fake prefers later
    assert all(rc.context_blurb == "B" for rc in out)
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/adapters/rerank/test_fastembed_rerank.py -q -m "not integration"`
Expected: FAIL, `unexpected keyword argument 'use_context'`.

- [ ] **Step 3: Implement**

```python
class FastEmbedReranker(Reranker):
    """Multilingual cross-encoder reranker (ONNX via fastembed).

    Scores each candidate passage against the query and returns the top_n by
    relevance. The default model is multilingual on purpose: an English-only
    cross-encoder would score cross-lingual passages low and defeat the point.

    With `use_context`, a candidate's context blurb is prepended to its text in
    the same format contextual indexing embeds, so the reranker judges what the
    dense and sparse stages matched rather than the bare chunk.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        use_context: bool = False,
    ) -> None:
        self._model_name = model_name
        self._use_context = use_context
        self._model = TextCrossEncoder(model_name=model_name)

    def _passage(self, rc: RetrievedChunk) -> str:
        if self._use_context and rc.context_blurb:
            return f"{rc.context_blurb}\n\n{rc.chunk.text}"
        return rc.chunk.text

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = list(self._model.rerank(query, [self._passage(rc) for rc in candidates]))
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [replace(rc, score=float(score)) for rc, score in ranked[:top_n]]
```

The `"{blurb}\n\n{text}"` format must match `LLMContextualizer`
(`src/ariostea/adapters/contextualize/llm.py:44`).

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/adapters/rerank/test_fastembed_rerank.py -q -m "not integration"`
Expected: PASS.

- [ ] **Step 5: Fast suite, then commit**

```bash
uv run pytest -m "not integration" -q
git add src/ariostea/adapters/rerank/fastembed_rerank.py tests/adapters/rerank/test_fastembed_rerank.py
git commit -m "feat(rerank): optionally score the context blurb with the chunk"
```

### Task 5: `RerankCfg.use_context`

**Files:**
- Modify: `src/ariostea/config/schema.py` (`RerankCfg`), `src/ariostea/config/container.py` (`_build_reranker`)
- Test: `tests/config/test_schema.py`, `tests/config/test_container.py`

- [ ] **Step 1: Write the failing tests**

In `tests/config/test_schema.py`, extend `test_rerank_defaults` with
`assert cfg.rerank.use_context is False`, and add:

```python
def test_rerank_use_context_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[rerank]\nuse_context = true\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.use_context is True
```

In `tests/config/test_container.py`:

```python
def test_build_reranker_passes_use_context(monkeypatch):
    from ariostea.config import container
    from ariostea.config.schema import RerankCfg

    built = {}

    class _Fake:
        def __init__(self, model_name, use_context=False):
            built.update(model_name=model_name, use_context=use_context)

    monkeypatch.setattr(container, "FastEmbedReranker", _Fake)
    container._build_reranker(RerankCfg(use_context=True))

    assert built["use_context"] is True
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/config -q -m "not integration"`
Expected: FAIL on `use_context`.

- [ ] **Step 3: Implement**

`schema.py`, in `RerankCfg` after `pool`:

```python
    # Score the context blurb with the chunk. Only matters when contextual
    # indexing is on; off until the wiki measurement says otherwise.
    use_context: bool = False
```

`container.py`, in `_build_reranker`:

```python
        return FastEmbedReranker(model_name=cfg.model, use_context=cfg.use_context)
```

- [ ] **Step 4: Run them and see them pass**

Run: `uv run pytest tests/config -q -m "not integration"`
Expected: PASS.

- [ ] **Step 5: Fast suite, then commit**

```bash
uv run pytest -m "not integration" -q
git add src/ariostea/config tests/config
git commit -m "feat(config): rerank.use_context, off by default"
```

### Task 6: eval helpers for the blurb experiment

**Files:**
- Modify: `src/ariostea/eval/wiki_index.py` (add `context_rerank_config`, `context_rerank_channel`)
- Modify: `src/ariostea/eval/results_log.py` (add `code_commit`), `eval/run_chunk_sweep.py` (use it)
- Create: `src/ariostea/eval/blurb_eval.py`
- Test: `tests/eval/test_wiki_index.py`, `tests/eval/test_blurb_eval.py`

- [ ] **Step 1: Write the failing tests**

In `tests/eval/test_wiki_index.py`, next to `test_fused_config_turns_off_only_the_reranker`:

```python
def test_context_rerank_config_changes_only_use_context(tmp_path):
    # Arm 3 is arm 2's index searched with the blurb-aware reranker: same
    # reranker model, same pool, same contextual config, nothing reindexed.
    from ariostea.config.schema import ContextualCfg
    from ariostea.eval.wiki_index import context_rerank_config, wiki_config

    indexed = wiki_config(tmp_path, "db", contextual=ContextualCfg(enabled=True, model="m"))
    arm3 = context_rerank_config(indexed)

    assert arm3.rerank.use_context is True
    assert indexed.rerank.use_context is False  # the original is untouched
    assert arm3.rerank.model_dump(exclude={"use_context"}) == indexed.rerank.model_dump(
        exclude={"use_context"}
    )
    assert arm3.model_dump(exclude={"rerank"}) == indexed.model_dump(exclude={"rerank"})
```

Create `tests/eval/test_blurb_eval.py`:

```python
import pytest

from ariostea.eval.blurb_eval import (
    ARMS,
    BlurbCoverageError,
    blurb_run_id,
    require_full_coverage,
)


def test_run_ids_are_distinct_per_arm_and_dated():
    ids = [blurb_run_id("2026-09-30", arm) for arm in ARMS]
    assert len(set(ids)) == len(ARMS)
    assert all(i.startswith("2026-09-30-blurbs-") for i in ids)


def test_each_arm_names_its_control_and_channels():
    by_key = {arm.key: arm for arm in ARMS}
    assert by_key["raw-rerank"].channels == ("DENSE", "SPARSE", "HYBRID")
    assert by_key["raw-rerank"].control == "2026-09-24-chunk-160t+40-dense-hybrid-sparse"
    assert by_key["fused"].channels == ("FUSED",)
    assert by_key["fused"].control == "2026-09-28-chunk-160t+40-fused"
    assert by_key["context-rerank"].channels == ("HYBRID",)
    assert by_key["context-rerank"].control == "2026-09-24-chunk-160t+40-dense-hybrid-sparse"


def test_full_coverage_passes_and_reports_the_count():
    rows = [("a.md", "blurb"), ("a.md", "blurb"), ("b.md", "other")]
    assert require_full_coverage(rows) == 2


def test_a_note_without_a_blurb_aborts_and_is_named():
    rows = [("a.md", "blurb"), ("b.md", None), ("c.md", "")]
    with pytest.raises(BlurbCoverageError, match="b.md.*c.md"):
        require_full_coverage(rows)


def test_an_empty_index_aborts():
    with pytest.raises(BlurbCoverageError):
        require_full_coverage([])
```

- [ ] **Step 2: Run them and see them fail**

Run: `uv run pytest tests/eval/test_wiki_index.py tests/eval/test_blurb_eval.py -q -m "not integration"`
Expected: FAIL with import errors.

- [ ] **Step 3: Implement the wiki_index helpers**

In `src/ariostea/eval/wiki_index.py`, after `fused_channel`:

```python
def context_rerank_config(config: Config) -> Config:
    """`config` with the reranker scoring blurb plus chunk, nothing else changed."""
    rerank = config.rerank.model_copy(update={"use_context": True})
    return config.model_copy(update={"rerank": rerank})


def context_rerank_channel(container: Container) -> SpanSearchFn:
    """Production search over `container`'s index with the blurb-aware
    reranker. Like `fused_channel`, a second container over the same database,
    not re-indexed, and not part of `wiki_channels` for the same reason."""
    return make_hybrid_chunk_fn(
        build_container(context_rerank_config(container.config)), CHUNK_POOL
    )
```

- [ ] **Step 4: Implement `blurb_eval.py`**

```python
"""Pure helpers for the blurb-aware reranking experiment (run_blurb_eval.py).

In the package rather than the runner so they are testable without a model,
an LLM or a database. See docs/design/2026-09-29-blurb-aware-reranking.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.contextual import find_uncontextualized_notes

HYBRID_CONTROL = "2026-09-24-chunk-160t+40-dense-hybrid-sparse"
FUSED_CONTROL = "2026-09-28-chunk-160t+40-fused"


@dataclass(frozen=True)
class Arm:
    """One log entry: which channels it records and the run it is compared to.

    `source` names the channel each logged channel is scored with, so arm 3
    can log its blurb-aware hybrid under "HYBRID" and be compared cell for
    cell with the control's HYBRID.
    """

    key: str
    label: str
    channels: tuple[str, ...]
    control: str
    source: dict[str, str]


ARMS = (
    Arm(
        key="raw-rerank",
        label="Blurbs, reranker scores raw text",
        channels=("DENSE", "SPARSE", "HYBRID"),
        control=HYBRID_CONTROL,
        source={"DENSE": "DENSE", "SPARSE": "SPARSE", "HYBRID": "HYBRID"},
    ),
    Arm(
        key="fused",
        label="Blurbs, no reranker",
        channels=("FUSED",),
        control=FUSED_CONTROL,
        source={"FUSED": "FUSED"},
    ),
    Arm(
        key="context-rerank",
        label="Blurbs, reranker scores blurb and text",
        channels=("HYBRID",),
        control=HYBRID_CONTROL,
        source={"HYBRID": "HYBRID+CONTEXT"},
    ),
)


def blurb_run_id(date: str, arm: Arm) -> str:
    return f"{date}-blurbs-{arm.key}"


class BlurbCoverageError(RuntimeError):
    """The blurbed index is missing blurbs; scoring it would understate them."""


def require_full_coverage(rows: list[tuple[str, str | None]]) -> int:
    """Return the number of blurbed notes, or raise naming every note that
    fell back to plain text. `rows` is (note_path, context_blurb) per chunk."""
    if not rows:
        raise BlurbCoverageError("the index has no chunks")
    missing = find_uncontextualized_notes(rows)
    if missing:
        raise BlurbCoverageError(
            f"{len(missing)} note(s) have no blurb: {', '.join(missing)}"
        )
    return len({path for path, _ in rows})
```

- [ ] **Step 5: Move the commit helper**

Move `_commit` from `eval/run_chunk_sweep.py` into
`src/ariostea/eval/results_log.py` as `code_commit()` (same body, same
docstring; add `import subprocess`). In `run_chunk_sweep.py`, delete `_commit`,
import `code_commit` from `ariostea.eval.results_log`, and call it where
`_commit()` was called. The git paths stay relative, so both runners must be
launched from the repo root, as they are today.

- [ ] **Step 6: Run the tests and see them pass**

Run: `uv run pytest tests/eval -q -m "not integration"`
Expected: PASS, including the existing sweep tests.

- [ ] **Step 7: Commit**

```bash
uv run pytest -m "not integration" -q
git add src/ariostea/eval tests/eval eval/run_chunk_sweep.py
git commit -m "feat(eval): helpers for the blurb-aware reranking experiment"
```

### Task 7: the runner `eval/run_blurb_eval.py`

**Files:**
- Create: `eval/run_blurb_eval.py`

The runner is wiring over tested helpers; it is verified by the manual run in
Task 9, like `run_chunk_sweep.py`.

- [ ] **Step 1: Write the runner**

```python
"""Measure note-level context blurbs, and blurb-aware reranking, on the wiki gold set.

Usage:
    uv run python eval/run_blurb_eval.py               # build the blurbed index, score, log
    uv run python eval/run_blurb_eval.py --reuse-index # score the index built last time

Indexes the corpus once at the production chunking default with contextual
indexing on, aborts unless every note got a blurb, then scores:
  arm 2  DENSE, SPARSE, HYBRID with the reranker on raw text, and FUSED
  arm 3  HYBRID with the reranker scoring blurb plus text
Arm 1 (no blurbs) is already logged; each entry names it as its control.
See docs/design/2026-09-29-blurb-aware-reranking.md.

The blurb LLM comes from:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_CTX_MODEL     (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_CTX_API_KEY   (default empty)
    ARIOSTEA_CTX_TIMEOUT   (seconds, default 300: whole articles, local model)
Load the model with a context window of at least 32k tokens first.

Blurbing takes 20 to 40 minutes and each HYBRID pass about an hour, so the
blurbed index is kept under eval/results/indexes/ for --reuse-index.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from ariostea.config.container import build_container
from ariostea.config.schema import ChunkingCfg, ContextualCfg
from ariostea.eval.blurb_eval import (
    ARMS,
    BlurbCoverageError,
    blurb_run_id,
    require_full_coverage,
)
from ariostea.eval.chunk_sweep import load_discriminated
from ariostea.eval.contextual import read_blurb_rows
from ariostea.eval.results_log import (
    append_run,
    code_commit,
    load_runs,
    make_run,
    render_html,
    report_to_dict,
)
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import load_wiki_gold
from ariostea.eval.wiki_index import (
    CHUNK_POOL,
    MULTILINGUAL_MODEL,
    context_rerank_channel,
    fused_channel,
    index_wiki_corpus,
    wiki_channels,
    wiki_config,
)

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
INDEX = RESULTS / "indexes" / "blurbs-160t+40.db"
K = 5


def _contextual() -> ContextualCfg:
    return ContextualCfg(
        enabled=True,
        base_url=os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:1234/v1"),
        model=os.environ.get("ARIOSTEA_CTX_MODEL", "qwen2.5-14b-instruct-mlx"),
        api_key=os.environ.get("ARIOSTEA_CTX_API_KEY", ""),
        timeout=float(os.environ.get("ARIOSTEA_CTX_TIMEOUT", "300")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reuse-index", action="store_true", help="score the kept index")
    parser.add_argument("--experiment", default="Contextual blurbs", help="group name in the log")
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    args = parser.parse_args(argv)

    today = dt.date.today().isoformat()
    runs = {run["id"]: run for run in load_runs(RUNS)}
    if not args.no_log:
        for arm in ARMS:
            if arm.control not in runs:
                parser.error(f"control run {arm.control!r} is not in {RUNS}")
            if blurb_run_id(today, arm) in runs:
                parser.error(f"already logged today: {blurb_run_id(today, arm)}")

    ctx = _contextual()
    if args.reuse_index:
        if not INDEX.exists():
            parser.error(f"no kept index at {INDEX}; run without --reuse-index first")
        print(f"reusing {INDEX}", flush=True)
        container = build_container(wiki_config(WIKI, str(INDEX), contextual=ctx))
    else:
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.unlink(missing_ok=True)
        print(f"indexing with blurbs from {ctx.model} ...", flush=True)
        container = index_wiki_corpus(WIKI, str(INDEX), contextual=ctx)

    try:
        blurbed = require_full_coverage(read_blurb_rows(str(INDEX)))
    except BlurbCoverageError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print(f"blurb coverage {blurbed}/{blurbed} notes", flush=True)

    cases = load_wiki_gold(WIKI / "gold.json")
    easy = load_discriminated(WIKI / "gold_rejected.json")
    channels = wiki_channels(str(INDEX), container)
    channels["FUSED"] = fused_channel(container)
    channels["HYBRID+CONTEXT"] = context_rerank_channel(container)

    commit = code_commit()
    for arm in ARMS:
        lines: list[str] = []

        def say(text: str) -> None:
            print(text, flush=True)
            lines.append(text)

        say(f"\n##### {arm.label}")
        scores: dict[str, dict] = {}
        dropped: dict[str, dict] = {}
        for name in arm.channels:
            fn = channels[arm.source[name]]
            say(f"  scoring {name} ({arm.source[name]}) ...")
            report = evaluate_spans(cases, fn, k=K, pool=CHUNK_POOL)
            scores[name] = report_to_dict(report)
            dropped[name] = report_to_dict(evaluate_spans(easy, fn, k=K, pool=CHUNK_POOL))[
                "overall"
            ]
            say(f"=== {name} ===\n{format_span_report(report)}")
            say(
                f"  discriminated-out cases ({len(easy)}): span recall "
                f"{dropped[name]['span_recall']:.3f}"
            )
        if args.no_log:
            continue

        run_id = blurb_run_id(today, arm)
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = make_run(
            run_id=run_id,
            experiment=args.experiment,
            label=arm.label,
            date=today,
            commit=commit,
            control=arm.control,
            config={
                "k": K,
                "embedding": MULTILINGUAL_MODEL,
                "chunking": ChunkingCfg().model_dump(),
                "contextual": {
                    "model": ctx.model,
                    "timeout": ctx.timeout,
                    "blurbed_notes": blurbed,
                },
                "rerank_use_context": arm.key == "context-rerank",
            },
            cases=len(cases),
            # Blurbs do not change chunking, so the control's ceiling holds.
            reachable=runs[arm.control]["reachable"],
            channels=scores,
            notes=f"Channels measured: {', '.join(arm.channels)}.",
        )
        run["discriminated"] = dropped
        append_run(RUNS, run)
        print(f"  logged {run_id}", flush=True)

    if not args.no_log:
        page = RESULTS / "experiment_log.html"
        template = (RESULTS / "template.html").read_text(encoding="utf-8")
        page.write_text(render_html(load_runs(RUNS), template), encoding="utf-8")
        print(f"\nrendered {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`build_container` on an existing database opens it without reindexing;
`fused_channel` already relies on that.

- [ ] **Step 2: Smoke-check the argument handling**

Run: `uv run python eval/run_blurb_eval.py --reuse-index --no-log`
Expected: exits with `no kept index at .../blurbs-160t+40.db` (no index yet). Nothing is indexed.

Run: `uv run ruff check eval/run_blurb_eval.py src/ariostea`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add eval/run_blurb_eval.py
git commit -m "feat(eval): runner for the blurb-aware reranking experiment"
```

### Task 8: full suite

- [ ] **Step 1: Run everything, integration included**

Run: `uv run pytest -q` (about 12 minutes; the reranker integration test downloads nothing new)
Expected: all pass, 1 skipped as before. Fix anything that fails before going on.

### Task 9: the measurement (manual, about 3 hours)

- [ ] **Step 1: Check that arm 1 still reproduces**

```bash
lms ps                        # nothing loaded
uv run python eval/run_chunk_sweep.py 160t+40 --channels DENSE,SPARSE --no-log
```

Expected: DENSE overall span recall 0.545, SPARSE 0.533. If either differs,
stop and report: arm 1 must be re-measured before any comparison.

- [ ] **Step 2: Load the blurb model**

```bash
lms unload --all
lms load qwen2.5-14b-instruct-mlx --context-length 32768
```

- [ ] **Step 3: Run detached**

The Claude Code memory watchdog kills background tasks, so detach:

```bash
nohup uv run python eval/run_blurb_eval.py > eval/results/logs/blurb-run.out 2>&1 & disown
```

`$!` is the `uv` wrapper; the Python worker is its child (`pgrep -P <pid>`).
Watch the saved PID with `kill -0`, not `pgrep -f`. Blurbing prints nothing
per note; progress is the log reaching `blurb coverage 79/79 notes`.

If it aborts on coverage, the output names the notes. Raise
`ARIOSTEA_CTX_TIMEOUT` or the context length and rerun without
`--reuse-index`. If it dies after the coverage line, rerun with
`--reuse-index`.

- [ ] **Step 4: Unload the model and commit the log**

```bash
lms unload --all
git add eval/results/runs.jsonl eval/results/experiment_log.html eval/results/logs/
git commit -m "chore(eval): log the contextual blurb runs"
```

- [ ] **Step 5: Republish the log artifact**

Republish `eval/results/experiment_log.html` to
https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid (pass it as `url`).

### Task 10: write it up and decide

- [ ] **Step 1: Apply the decision rule**

From the log, for each comparison compute overall and per-type deltas:
- arm 2 HYBRID and arm 3 HYBRID against arm 1 HYBRID (0.880);
- arm 2 FUSED against arm 1 FUSED (0.629);
- arm 3 HYBRID against arm 2 HYBRID;
- arm 2 DENSE and SPARSE against arm 1, per type.

A comparison passes at +0.03 overall with no type below -0.05.

- [ ] **Step 2: Add a "Contextual blurbs" section to `docs/retrieval-tuning.md`**

Cover the setup (arms, model, coverage), the results table, the verdict under
the rule, and what DENSE and SPARSE say about one blurb shared by dozens of
chunks. Add the reproduction commands to the "Reproducing" section.

- [ ] **Step 3: Update memory `contextual-lift-measured`** with the wiki result.

- [ ] **Step 4: Ask the owner about the `use_context` default**

Only if arm 3 passes against arm 2. Do not flip it without approval.

- [ ] **Step 5: Commit**

```bash
git add docs/retrieval-tuning.md
git commit -m "docs: measure contextual blurbs and blurb-aware reranking"
```
