# Ariostea — Phase 0 + 1 (Walking Skeleton) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a working, local, end-to-end RAG MCP server that indexes an Obsidian vault and answers `search_knowledge` queries with dense semantic retrieval — built on the clean-architecture skeleton (domain → ports → use cases → adapters → delivery).

**Architecture:** Ports & adapters. Pure `domain/` entities; use cases (`indexing/`, `search/`) depend only on `ports/` Protocols; concrete adapters (fastembed, sqlite-vec, MCP SDK) live at the edge and are wired by a single composition root. This phase implements the **dense-retrieval slice only** — sparse/BM25, fusion, reranking, contextual blurbs, provenance, and the watcher arrive in later phase plans.

**Tech Stack:** Python 3.12, `uv` (env + runner), `pytest` (TDD), `pydantic` (config), `tomllib` (stdlib, config parsing), `typer` (CLI), `fastembed` (local ONNX embeddings, no PyTorch), `sqlite-vec` + stdlib `sqlite3` (vector store), `mcp` (MCP Python SDK / `FastMCP`).

---

## How we work through this plan

Each task starts with **Why this shape** (the design reasoning — read it, ask me anything before we write code), then runs the TDD loop: failing test → confirm it fails → minimal implementation → confirm it passes → commit. We do **not** advance to the next task until you're satisfied. Stop me with questions at any checkbox.

## Scope of this plan

In scope: project scaffold; domain models; port Protocols; config + composition root; MCP server with `status`, `reindex`, `search_knowledge` tools; vault scanner; minimal Obsidian markdown parser; heading-aware chunker; fastembed embedding adapter; sqlite-vec store (schema, upsert, dense query, admin); the `IndexVault` and `SearchKnowledge` use cases; end-to-end test over a fixture vault.

**Deferred to later phase plans:** FTS5/BM25 sparse retrieval + RRF fusion (Phase 2), provenance rollup + `search_sources` (Phase 3), incremental hash-diff + file watcher (Phase 4 — Phase 1 already hashes, Phase 4 adds the watcher and delete-detection), contextual blurbs + prompt caching (Phase 5), reranking (Phase 6), deep Obsidian graph/filters (Phase 7), packaging wizard + alt adapters (Phase 8).

## Conventions

- Environment & deps: `uv sync` (creates `.venv`, installs deps).
- Run a test: `uv run pytest tests/path::test_name -v`
- Run all tests: `uv run pytest -q`
- Package source lives under `src/ariostea/`; tests under `tests/` mirroring the package.
- Commit after every green step. Commit messages use Conventional Commits.

---

## Task 0.1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `src/ariostea/__init__.py`
- Create: `tests/test_smoke.py`

**Why this shape:** A `src/` layout forces tests to import the *installed* package (not loose files), which catches packaging mistakes early. `uv` gives us a one-command reproducible env, matching the project's "single command" ethos. We pin the dependency surface now so every later task has its tools available.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
import ariostea


def test_package_exposes_version():
    assert isinstance(ariostea.__version__, str)
    assert ariostea.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea'` (package not yet created/installed).

- [ ] **Step 3: Write minimal implementation**

```toml
# pyproject.toml
[project]
name = "ariostea"
version = "0.0.1"
description = "Local-first, Obsidian-aware RAG MCP server"
requires-python = ">=3.12"
dependencies = [
  "mcp>=1.2",
  "fastembed>=0.4",
  "sqlite-vec>=0.1.6",
  "pydantic>=2.6",
  "typer>=0.12",
  "watchfiles>=0.24",
]

[project.scripts]
ariostea = "ariostea.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/ariostea"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[dependency-groups]
dev = ["pytest>=8"]
```

```python
# src/ariostea/__init__.py
__version__ = "0.0.1"
```

- [ ] **Step 4: Sync the environment, then run test to verify it passes**

Run: `uv sync && uv run pytest tests/test_smoke.py -v`
Expected: PASS. (First `uv sync` downloads deps; subsequent runs are fast.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ariostea/__init__.py tests/test_smoke.py
git commit -m "chore: scaffold ariostea package with uv + pytest"
```

---

## Task 0.2: Domain entities

**Files:**
- Create: `src/ariostea/domain/__init__.py`
- Create: `src/ariostea/domain/models.py`
- Create: `tests/domain/test_models.py`

**Why this shape:** These are the innermost circle — frozen, dependency-free value objects. Freezing them (`frozen=True`) makes them safe to pass across boundaries (no shared mutable state) and signals they're data, not behavior. Every other layer will speak in these types, so getting the field names right now keeps later tasks consistent.

- [ ] **Step 1: Write the failing test**

```python
# tests/domain/test_models.py
import dataclasses
import pytest

from ariostea.domain.models import Note, Chunk, RetrievedChunk, Query, IndexStats


def test_note_holds_metadata_and_is_frozen():
    note = Note(
        path="ideas/rag.md",
        title="RAG",
        frontmatter={"status": "draft"},
        tags=("ml", "search"),
        wikilinks=("Embeddings",),
        content_hash="abc123",
        mtime=1.0,
    )
    assert note.tags == ("ml", "search")
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.path = "other.md"


def test_chunk_and_retrieved_chunk_compose():
    chunk = Chunk(note_path="ideas/rag.md", ordinal=0, heading_path=("RAG",), text="hello", token_count=1)
    rc = RetrievedChunk(chunk=chunk, score=0.9, dense_rank=0, sparse_rank=None)
    assert rc.chunk.text == "hello"
    assert rc.score == 0.9


def test_query_defaults():
    q = Query(text="what is rag")
    assert q.k == 10 and q.filters is None


def test_index_stats_fields():
    s = IndexStats(notes=2, chunks=5, last_indexed=1.0, config_fingerprint="fp")
    assert s.notes == 2 and s.config_fingerprint == "fp"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/domain/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.domain'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/domain/__init__.py
```

```python
# src/ariostea/domain/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Note:
    path: str
    title: str
    frontmatter: dict[str, Any]
    tags: tuple[str, ...]
    wikilinks: tuple[str, ...]
    content_hash: str
    mtime: float


@dataclass(frozen=True)
class Chunk:
    note_path: str
    ordinal: int
    heading_path: tuple[str, ...]
    text: str
    token_count: int


@dataclass(frozen=True)
class ContextualizedChunk:
    chunk: Chunk
    context_blurb: str | None
    embedding_text: str


@dataclass(frozen=True)
class QueryFilters:
    tags: tuple[str, ...] = ()
    path_globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Query:
    text: str
    k: int = 10
    filters: QueryFilters | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


@dataclass(frozen=True)
class SearchResult:
    chunks: tuple[RetrievedChunk, ...]


@dataclass(frozen=True)
class SourceHit:
    note_path: str
    title: str
    hit_count: int
    best_score: float
    snippets: tuple[str, ...]


@dataclass(frozen=True)
class IndexStats:
    notes: int
    chunks: int
    last_indexed: float
    config_fingerprint: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/domain/test_models.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/domain tests/domain
git commit -m "feat(domain): add core entities (Note, Chunk, Query, results)"
```

---

## Task 0.3: Port Protocols

**Files:**
- Create: `src/ariostea/ports/__init__.py`
- Create: `src/ariostea/ports/embedding.py`
- Create: `src/ariostea/ports/store.py`
- Create: `src/ariostea/ports/pipeline.py`
- Create: `tests/ports/test_protocols.py`

**Why this shape:** Ports are the seams that make every part swappable. They're `Protocol`s (structural typing) so adapters don't need to inherit anything — they just match the shape. We split the store into **role ports** (`DocumentWriter`, `ChunkRetriever`, `IndexAdmin`) so the indexing use case can't accidentally call search methods, and vice versa (Interface Segregation). `@runtime_checkable` lets us assert conformance in a test. We define only the ports this phase uses; later phases add `Reranker`, `ChatProvider`, `SourceRollup`, `Fuser`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ports/test_protocols.py
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import DocumentWriter, ChunkRetriever, IndexAdmin
from ariostea.ports.pipeline import MarkdownParser, Chunker
from ariostea.domain.models import Note, Chunk, IndexStats


class FakeEmbed:
    def embed_documents(self, texts):
        return [[0.0, 1.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 1.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


def test_fake_embedder_conforms():
    assert isinstance(FakeEmbed(), EmbeddingProvider)


def test_store_role_ports_are_distinct():
    # A class can satisfy multiple role ports; the ports themselves are separate types.
    assert DocumentWriter is not ChunkRetriever is not IndexAdmin
    assert {DocumentWriter, ChunkRetriever, IndexAdmin}.__len__() == 3


def test_pipeline_ports_exist():
    assert MarkdownParser is not None and Chunker is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ports/test_protocols.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.ports'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/ports/__init__.py
```

```python
# src/ariostea/ports/embedding.py
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
    @property
    def dimension(self) -> int: ...
    @property
    def fingerprint(self) -> str: ...
```

```python
# src/ariostea/ports/store.py
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ariostea.domain.models import (
    Note,
    ContextualizedChunk,
    RetrievedChunk,
    QueryFilters,
    IndexStats,
)


@runtime_checkable
class DocumentWriter(Protocol):
    def upsert_note(
        self,
        note: Note,
        chunks: Sequence[ContextualizedChunk],
        embeddings: Sequence[list[float]],
    ) -> None: ...
    def delete_note(self, path: str) -> None: ...


@runtime_checkable
class ChunkRetriever(Protocol):
    def dense(
        self, vec: list[float], k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class IndexAdmin(Protocol):
    def known_hashes(self) -> dict[str, str]: ...
    def stats(self) -> IndexStats: ...
    def fingerprint(self) -> str: ...
    def set_fingerprint(self, value: str) -> None: ...
```

```python
# src/ariostea/ports/pipeline.py
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ariostea.domain.models import Note, Chunk


@runtime_checkable
class MarkdownParser(Protocol):
    def parse(self, path: str, raw: str, mtime: float) -> tuple[Note, str]:
        """Return (note metadata, body-without-frontmatter)."""
        ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, note: Note, body: str) -> list[Chunk]: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ports/test_protocols.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/ports tests/ports
git commit -m "feat(ports): add embedding, store role, and pipeline protocols"
```

---

## Task 0.4: Config schema + loader

**Files:**
- Create: `src/ariostea/config/__init__.py`
- Create: `src/ariostea/config/schema.py`
- Create: `tests/config/test_schema.py`

**Why this shape:** Config is the user-facing contract for "swap a base URL to change provider." Pydantic gives us typed validation and defaults so the single command works with an almost-empty file (only `vault.path` required). `tomllib` is stdlib — no extra dependency to read TOML. The loader is pure (path in, `Config` out) so it's trivially testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_schema.py
from ariostea.config.schema import Config, load_config


def test_minimal_config_applies_defaults(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.vault.path == "~/Vault"
    assert cfg.embedding.provider == "local"      # default
    assert cfg.store.backend == "sqlite"          # default
    assert cfg.search.top_k == 10                 # default


def test_full_config_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(
        """
[vault]
path = "/notes"
ignore = [".obsidian/"]

[embedding]
provider = "openai_compat"
base_url = "http://localhost:11434/v1"
model = "nomic-embed-text"

[store]
backend = "sqlite"
path = "/tmp/index.db"

[search]
k_dense = 40
top_k = 8
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.embedding.base_url == "http://localhost:11434/v1"
    assert cfg.embedding.model == "nomic-embed-text"
    assert cfg.vault.ignore == [".obsidian/"]
    assert cfg.search.k_dense == 40 and cfg.search.top_k == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/config/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/config/__init__.py
```

```python
# src/ariostea/config/schema.py
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


class VaultCfg(BaseModel):
    path: str
    ignore: list[str] = [".obsidian/"]


class EmbeddingCfg(BaseModel):
    provider: str = "local"  # "local" | "openai_compat"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    local_model: str = "BAAI/bge-small-en-v1.5"


class StoreCfg(BaseModel):
    backend: str = "sqlite"
    path: str = "~/.ariostea/index.db"


class SearchCfg(BaseModel):
    k_dense: int = 50
    top_k: int = 10


class Config(BaseModel):
    vault: VaultCfg
    embedding: EmbeddingCfg = EmbeddingCfg()
    store: StoreCfg = StoreCfg()
    search: SearchCfg = SearchCfg()


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return Config(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/config/test_schema.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/config tests/config
git commit -m "feat(config): typed TOML config schema and loader"
```

---

## Task 0.5: MCP server with `status` tool + CLI skeleton

**Files:**
- Create: `src/ariostea/mcp/__init__.py`
- Create: `src/ariostea/mcp/handlers.py`
- Create: `src/ariostea/mcp/server.py`
- Create: `src/ariostea/cli.py`
- Create: `tests/mcp/test_handlers.py`

**Why this shape:** The MCP layer is pure delivery. To keep it testable without spinning up a server, all logic lives in **plain handler functions** (`mcp/handlers.py`) that take a port and return plain data; `mcp/server.py` only wraps them as `FastMCP` tools. We test the handlers, not the framework. This is the clean-architecture payoff: business-facing behavior is verified without the framework running. The CLI is a thin `typer` entry so `ariostea` resolves; real wiring lands in Task 1.8.

- [ ] **Step 1: Write the failing test**

```python
# tests/mcp/test_handlers.py
from ariostea.mcp.handlers import status_payload
from ariostea.domain.models import IndexStats


class FakeAdmin:
    def known_hashes(self):
        return {}

    def stats(self):
        return IndexStats(notes=3, chunks=12, last_indexed=42.0, config_fingerprint="fp")

    def fingerprint(self):
        return "fp"

    def set_fingerprint(self, value):
        pass


def test_status_payload_reports_counts():
    payload = status_payload(FakeAdmin())
    assert payload == {
        "notes": 3,
        "chunks": 12,
        "last_indexed": 42.0,
        "config_fingerprint": "fp",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/mcp/test_handlers.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.mcp'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/mcp/__init__.py
```

```python
# src/ariostea/mcp/handlers.py
from __future__ import annotations

from ariostea.ports.store import IndexAdmin


def status_payload(admin: IndexAdmin) -> dict:
    s = admin.stats()
    return {
        "notes": s.notes,
        "chunks": s.chunks,
        "last_indexed": s.last_indexed,
        "config_fingerprint": s.config_fingerprint,
    }
```

```python
# src/ariostea/mcp/server.py
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ariostea.mcp.handlers import status_payload


def build_server(admin) -> FastMCP:
    mcp = FastMCP("ariostea")

    @mcp.tool()
    def status() -> dict:
        """Report index health: note/chunk counts, last index time, config fingerprint."""
        return status_payload(admin)

    return mcp
```

```python
# src/ariostea/cli.py
from __future__ import annotations

import typer

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


@app.command()
def serve(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Run the MCP server (full wiring lands in Task 1.8)."""
    typer.echo(f"ariostea serve — config={config} (not yet wired)")


@app.command()
def main_placeholder() -> None:  # keeps the module importable before wiring
    typer.echo("ok")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/mcp/test_handlers.py -v`
Expected: PASS (1 test). Also confirm the CLI resolves: `uv run ariostea --help` lists `serve`.

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/mcp src/ariostea/cli.py tests/mcp
git commit -m "feat(mcp): status handler + FastMCP server + CLI skeleton"
```

---

> **Phase 0 complete:** the skeleton is wired end-to-end (config → port → handler → MCP tool → CLI). Phase 1 fills it with real retrieval.

---

## Task 1.1: Vault scanner

**Files:**
- Create: `src/ariostea/indexing/__init__.py`
- Create: `src/ariostea/indexing/scanner.py`
- Create: `tests/indexing/test_scanner.py`

**Why this shape:** The scanner is the only component that touches the filesystem during indexing. It yields `(path, raw_text, mtime, content_hash)` so downstream stages stay pure. Content-hashing here is what makes incremental indexing possible later (Phase 4) — we compute it now even though we don't yet diff against the store. Ignore rules keep `.obsidian/` and templates out.

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_scanner.py
from ariostea.indexing.scanner import scan_vault


def test_scan_finds_markdown_and_hashes(tmp_path):
    (tmp_path / "a.md").write_text("# A\nhello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\nworld")
    (tmp_path / "ignore.txt").write_text("not markdown")

    found = {f.rel_path: f for f in scan_vault(tmp_path, ignore=[])}
    assert set(found) == {"a.md", "sub/b.md"}
    assert found["a.md"].raw.startswith("# A")
    assert len(found["a.md"].content_hash) == 64  # sha256 hex


def test_scan_respects_ignore(tmp_path):
    (tmp_path / "keep.md").write_text("k")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("x")
    found = {f.rel_path for f in scan_vault(tmp_path, ignore=[".obsidian/"])}
    assert found == {"keep.md"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_scanner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.indexing'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/indexing/__init__.py
```

```python
# src/ariostea/indexing/scanner.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True)
class ScannedFile:
    rel_path: str
    raw: str
    mtime: float
    content_hash: str


def _is_ignored(rel_path: str, ignore: Sequence[str]) -> bool:
    return any(rel_path == pat.rstrip("/") or rel_path.startswith(pat) for pat in ignore)


def scan_vault(root: str | Path, ignore: Sequence[str] = ()) -> Iterator[ScannedFile]:
    root = Path(root)
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if _is_ignored(rel, ignore):
            continue
        raw = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        yield ScannedFile(
            rel_path=rel,
            raw=raw,
            mtime=path.stat().st_mtime,
            content_hash=digest,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_scanner.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/indexing/__init__.py src/ariostea/indexing/scanner.py tests/indexing/test_scanner.py
git commit -m "feat(indexing): vault scanner with hashing and ignore rules"
```

---

## Task 1.2: Obsidian markdown parser

**Files:**
- Create: `src/ariostea/adapters/__init__.py`
- Create: `src/ariostea/adapters/parse/__init__.py`
- Create: `src/ariostea/adapters/parse/obsidian.py`
- Create: `tests/adapters/parse/test_obsidian.py`

**Why this shape:** This adapter implements the `MarkdownParser` port. For the walking skeleton we extract just enough Obsidian structure to be useful and to populate `Note`: YAML frontmatter, title, `#tags`, `[[wikilinks]]`, and the body with frontmatter stripped. We parse frontmatter by hand (a small fenced block) to avoid a YAML dependency for now; the deep graph work is Phase 7. Returning `(Note, body)` lets the chunker work on clean text.

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/parse/test_obsidian.py
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser


def test_parses_frontmatter_title_tags_links():
    raw = (
        "---\n"
        "status: draft\n"
        "---\n"
        "# Retrieval Augmented Generation\n\n"
        "We use [[Embeddings]] and #search techniques.\n"
    )
    parser = ObsidianMarkdownParser()
    note, body = parser.parse("ideas/rag.md", raw, mtime=1.0)

    assert note.title == "Retrieval Augmented Generation"
    assert note.frontmatter == {"status": "draft"}
    assert "search" in note.tags
    assert "Embeddings" in note.wikilinks
    assert body.startswith("# Retrieval Augmented Generation")
    assert "status: draft" not in body  # frontmatter stripped


def test_title_falls_back_to_filename_when_no_heading():
    parser = ObsidianMarkdownParser()
    note, _ = parser.parse("notes/loose-thought.md", "just text, no heading", mtime=2.0)
    assert note.title == "loose-thought"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/parse/test_obsidian.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/adapters/__init__.py
```

```python
# src/ariostea/adapters/parse/__init__.py
```

```python
# src/ariostea/adapters/parse/obsidian.py
from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from ariostea.domain.models import Note

_FRONTMATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
_TAG = re.compile(r"(?:^|\s)#([A-Za-z0-9_\-/]+)")
_WIKILINK = re.compile(r"\[\[([^\]|#]+)")
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    m = _FRONTMATTER.match(raw)
    if not m:
        return {}, raw
    block, body = m.group(1), raw[m.end():]
    fm: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm, body


class ObsidianMarkdownParser:
    def parse(self, path: str, raw: str, mtime: float) -> tuple[Note, str]:
        frontmatter, body = _parse_frontmatter(raw)
        heading = _H1.search(body)
        title = heading.group(1).strip() if heading else PurePosixPath(path).stem
        tags = tuple(sorted(set(_TAG.findall(body))))
        wikilinks = tuple(sorted(set(link.strip() for link in _WIKILINK.findall(body))))
        content_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        note = Note(
            path=path,
            title=title,
            frontmatter=frontmatter,
            tags=tags,
            wikilinks=wikilinks,
            content_hash=content_hash,
            mtime=mtime,
        )
        return note, body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/parse/test_obsidian.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/adapters/__init__.py src/ariostea/adapters/parse tests/adapters/parse
git commit -m "feat(parse): minimal Obsidian markdown parser (frontmatter/tags/links/title)"
```

---

## Task 1.3: Heading-aware chunker

**Files:**
- Create: `src/ariostea/adapters/chunk/__init__.py`
- Create: `src/ariostea/adapters/chunk/heading_aware.py`
- Create: `tests/adapters/chunk/test_heading_aware.py`

**Why this shape:** Implements the `Chunker` port. Chunk boundaries that respect headings keep semantically-coherent units together (a heading + its prose), which is what makes the `heading_path` metadata meaningful for later filtering and provenance. For the skeleton we split on headings, then soft-split oversized sections by paragraph up to a token budget. We approximate tokens as whitespace-words now (good enough; a real tokenizer is a later refinement) to avoid pulling a tokenizer dependency into this task.

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/chunk/test_heading_aware.py
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.domain.models import Note


def _note():
    return Note(path="n.md", title="N", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=0.0)


def test_splits_on_headings_and_tracks_heading_path():
    body = (
        "# Title\n"
        "Intro paragraph.\n\n"
        "## Section A\n"
        "Alpha content.\n\n"
        "## Section B\n"
        "Beta content.\n"
    )
    chunks = HeadingAwareChunker(max_tokens=200).chunk(_note(), body)
    headings = [c.heading_path for c in chunks]
    texts = [c.text for c in chunks]
    assert ("Title",) in headings
    assert ("Title", "Section A") in headings
    assert ("Title", "Section B") in headings
    assert any("Alpha content" in t for t in texts)
    # ordinals are sequential
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_oversized_section_is_split_by_token_budget():
    body = "# T\n" + " ".join(f"word{i}" for i in range(50))
    chunks = HeadingAwareChunker(max_tokens=20).chunk(_note(), body)
    assert len(chunks) >= 2
    assert all(c.token_count <= 20 for c in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/chunk/test_heading_aware.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.chunk'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/adapters/chunk/__init__.py
```

```python
# src/ariostea/adapters/chunk/heading_aware.py
from __future__ import annotations

import re
from dataclasses import dataclass

from ariostea.domain.models import Note, Chunk

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


@dataclass
class _Section:
    heading_path: tuple[str, ...]
    text: str


def _split_sections(body: str) -> list[_Section]:
    sections: list[_Section] = []
    stack: list[str] = []  # current heading path by level
    buffer: list[str] = []
    current_path: tuple[str, ...] = ()

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append(_Section(current_path, text))
        buffer.clear()

    for line in body.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()
            stack[:] = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")
            stack.append(title)
            current_path = tuple(s for s in stack if s)
            buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return sections


def _token_count(text: str) -> int:
    return len(text.split())


class HeadingAwareChunker:
    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def chunk(self, note: Note, body: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        ordinal = 0
        for section in _split_sections(body):
            for piece in self._fit(section.text):
                chunks.append(
                    Chunk(
                        note_path=note.path,
                        ordinal=ordinal,
                        heading_path=section.heading_path,
                        text=piece,
                        token_count=_token_count(piece),
                    )
                )
                ordinal += 1
        return chunks

    def _fit(self, text: str) -> list[str]:
        words = text.split()
        if len(words) <= self.max_tokens:
            return [text]
        pieces: list[str] = []
        for i in range(0, len(words), self.max_tokens):
            pieces.append(" ".join(words[i : i + self.max_tokens]))
        return pieces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/chunk/test_heading_aware.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/adapters/chunk tests/adapters/chunk
git commit -m "feat(chunk): heading-aware chunker with token-budget splitting"
```

---

## Task 1.4: FastEmbed embedding adapter

**Files:**
- Create: `src/ariostea/adapters/embedding/__init__.py`
- Create: `src/ariostea/adapters/embedding/fastembed_local.py`
- Create: `tests/adapters/embedding/test_fastembed_local.py`

**Why this shape:** Implements `EmbeddingProvider` with a bundled local model — the zero-key path. `fastembed` runs ONNX (no PyTorch), so it stays light enough for a `uvx` tool. `dimension` is discovered by probing once and cached; `fingerprint` encodes the model id so that changing the model later triggers a guarded reindex (Phase 4+). This test actually downloads/loads the model on first run, so it's an **integration test** (slower); mark it so it can be deselected in fast loops.

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/embedding/test_fastembed_local.py
import pytest

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings


@pytest.mark.integration
def test_embeds_documents_and_query_consistently():
    emb = FastEmbedEmbeddings()  # default BAAI/bge-small-en-v1.5
    docs = emb.embed_documents(["cats and dogs", "vector databases"])
    q = emb.embed_query("vector databases")

    assert len(docs) == 2
    assert len(docs[0]) == emb.dimension == len(q)
    assert emb.fingerprint.startswith("fastembed:")
    # query is closer to its matching doc than the unrelated one (cosine via dot on normalized vecs)
    import math

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert dot(q, docs[1]) > dot(q, docs[0])
```

Register the marker so pytest doesn't warn:

```toml
# add to pyproject.toml under [tool.pytest.ini_options]
markers = ["integration: tests that load real models or hit external services"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/embedding/test_fastembed_local.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.embedding'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/adapters/embedding/__init__.py
```

```python
# src/ariostea/adapters/embedding/fastembed_local.py
from __future__ import annotations

from typing import Sequence

from fastembed import TextEmbedding


class FastEmbedEmbeddings:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._dim: int | None = None

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    @property
    def fingerprint(self) -> str:
        return f"fastembed:{self._model_name}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/embedding/test_fastembed_local.py -v -m integration`
Expected: PASS (1 test; first run downloads the model — allow a minute).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/ariostea/adapters/embedding tests/adapters/embedding
git commit -m "feat(embedding): local fastembed adapter (ONNX, zero-key)"
```

---

## Task 1.5: SQLite vector store (schema + upsert + dense + admin)

**Files:**
- Create: `src/ariostea/adapters/store/__init__.py`
- Create: `src/ariostea/adapters/store/sqlite_store.py`
- Create: `tests/adapters/store/test_sqlite_store.py`

**Why this shape:** This adapter satisfies `DocumentWriter`, `ChunkRetriever`, and `IndexAdmin` in one class (the spec's single-file SQLite store) — but callers receive only the role they need. `sqlite-vec` provides the `vec0` virtual table for KNN; we serialize vectors with `sqlite_vec.serialize_float32`. The schema lives entirely here (it must not leak into the domain). `upsert_note` is transactional and replaces a note's chunks wholesale, which makes re-indexing a file idempotent. FTS/BM25, links, and tags tables are **deferred** to their phases — we add only `notes`, `chunks`, `chunks_vec`, and `meta` now.

> **Setup note:** loading a SQLite extension requires `conn.enable_load_extension(True)`. uv-managed CPython supports this. If you hit `AttributeError: enable_load_extension` on a system Python, run under `uv` (as all commands here do).

- [ ] **Step 1: Write the failing test**

```python
# tests/adapters/store/test_sqlite_store.py
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.domain.models import Note, Chunk, ContextualizedChunk


def _note(path="a.md"):
    return Note(path=path, title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h1", mtime=1.0)


def _cchunk(note, ordinal, text):
    chunk = Chunk(note_path=note.path, ordinal=ordinal, heading_path=("A",), text=text, token_count=len(text.split()))
    return ContextualizedChunk(chunk=chunk, context_blurb=None, embedding_text=text)


def test_upsert_then_dense_retrieves_nearest(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunks = [_cchunk(note, 0, "alpha"), _cchunk(note, 1, "beta")]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert_note(note, chunks, embeddings)

    hits = store.dense([0.9, 0.1, 0.0], k=2)
    assert hits[0].chunk.text == "alpha"          # nearest to [1,0,0]
    assert hits[0].chunk.note_path == "a.md"
    assert len(hits) == 2


def test_upsert_replaces_previous_chunks(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "old")], [[1.0, 0.0, 0.0]])
    store.upsert_note(note, [_cchunk(note, 0, "new")], [[1.0, 0.0, 0.0]])
    hits = store.dense([1.0, 0.0, 0.0], k=5)
    assert [h.chunk.text for h in hits] == ["new"]  # old chunk gone


def test_admin_reports_hashes_and_stats(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "x")], [[1.0, 0.0, 0.0]])
    store.set_fingerprint("fp-1")

    assert store.known_hashes() == {"a.md": "h1"}
    stats = store.stats()
    assert stats.notes == 1 and stats.chunks == 1 and stats.config_fingerprint == "fp-1"


def test_delete_note_removes_everything(tmp_path):
    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    store.upsert_note(note, [_cchunk(note, 0, "x")], [[1.0, 0.0, 0.0]])
    store.delete_note("a.md")
    assert store.known_hashes() == {}
    assert store.dense([1.0, 0.0, 0.0], k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.store'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/adapters/store/__init__.py
```

```python
# src/ariostea/adapters/store/sqlite_store.py
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Sequence

import sqlite_vec

from ariostea.domain.models import (
    Note,
    Chunk,
    ContextualizedChunk,
    RetrievedChunk,
    QueryFilters,
    IndexStats,
)


class SqliteStore:
    def __init__(self, path: str, dim: int) -> None:
        self._dim = dim
        Path(path).parent.mkdir(parents=True, exist_ok=True) if Path(path).parent != Path("") else None
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                mtime REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading_path TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dim}]
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self.db.commit()

    # --- DocumentWriter ---
    def upsert_note(
        self,
        note: Note,
        chunks: Sequence[ContextualizedChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._delete_note_rows(cur, note.path)
            cur.execute(
                "INSERT INTO notes(path, title, content_hash, mtime) VALUES (?,?,?,?)",
                (note.path, note.title, note.content_hash, note.mtime),
            )
            note_id = cur.lastrowid
            for cc, vec in zip(chunks, embeddings):
                ch = cc.chunk
                cur.execute(
                    "INSERT INTO chunks(note_id, ordinal, heading_path, text, token_count) "
                    "VALUES (?,?,?,?,?)",
                    (note_id, ch.ordinal, "/".join(ch.heading_path), ch.text, ch.token_count),
                )
                chunk_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vec)),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def delete_note(self, path: str) -> None:
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._delete_note_rows(cur, path)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _delete_note_rows(self, cur: sqlite3.Cursor, path: str) -> None:
        row = cur.execute("SELECT id FROM notes WHERE path = ?", (path,)).fetchone()
        if row is None:
            return
        note_id = row["id"]
        chunk_ids = [
            r["id"] for r in cur.execute("SELECT id FROM chunks WHERE note_id = ?", (note_id,)).fetchall()
        ]
        for cid in chunk_ids:
            cur.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
        cur.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    # --- ChunkRetriever ---
    def dense(self, vec: list[float], k: int, filters: QueryFilters | None = None) -> list[RetrievedChunk]:
        rows = self.db.execute(
            """
            WITH knn AS (
                SELECT chunk_id, distance
                FROM chunks_vec
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.note_id, n.path AS note_path, c.ordinal, c.heading_path,
                   c.text, c.token_count, knn.distance
            FROM knn
            JOIN chunks c ON c.id = knn.chunk_id
            JOIN notes n ON n.id = c.note_id
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(vec), k),
        ).fetchall()
        # NOTE: sqlite-vec requires the `k = ?` constraint on the vec0 scan itself;
        # a LIMIT applied after the JOIN is rejected ("k = ? constraint is required").
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
            results.append(
                RetrievedChunk(chunk=chunk, score=1.0 / (1.0 + r["distance"]), dense_rank=rank, sparse_rank=None)
            )
        return results

    # --- IndexAdmin ---
    def known_hashes(self) -> dict[str, str]:
        rows = self.db.execute("SELECT path, content_hash FROM notes").fetchall()
        return {r["path"]: r["content_hash"] for r in rows}

    def stats(self) -> IndexStats:
        notes = self.db.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
        chunks = self.db.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        return IndexStats(
            notes=notes, chunks=chunks, last_indexed=time.time(), config_fingerprint=self.fingerprint()
        )

    def fingerprint(self) -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        return row["value"] if row else ""

    def set_fingerprint(self, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES ('fingerprint', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (value,),
        )
        self.db.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/adapters/store/test_sqlite_store.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/adapters/store tests/adapters/store
git commit -m "feat(store): sqlite-vec store with upsert, dense KNN, admin"
```

---

## Task 1.6: `IndexVault` use case

**Files:**
- Create: `src/ariostea/indexing/index_vault.py`
- Create: `tests/indexing/test_index_vault.py`

**Why this shape:** This is application policy — it orchestrates scan → parse → chunk → embed → write **through ports only**. It never imports fastembed or sqlite. That's why the test can drive it entirely with fakes (fast, deterministic, no model download): the proof that our boundaries are real. For the skeleton there's no contextualization yet, so `embedding_text == chunk.text` and `context_blurb is None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_index_vault.py
from ariostea.indexing.index_vault import IndexVault
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker


class FakeEmbed:
    def __init__(self):
        self.seen = []

    def embed_documents(self, texts):
        self.seen.extend(texts)
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


class FakeStore:
    def __init__(self):
        self.notes = {}
        self._fp = ""

    def upsert_note(self, note, chunks, embeddings):
        self.notes[note.path] = (note, list(chunks), list(embeddings))

    def delete_note(self, path):
        self.notes.pop(path, None)

    def known_hashes(self):
        return {p: n.content_hash for p, (n, _, _) in self.notes.items()}

    def stats(self):
        from ariostea.domain.models import IndexStats
        return IndexStats(len(self.notes), 0, 0.0, self._fp)

    def fingerprint(self):
        return self._fp

    def set_fingerprint(self, value):
        self._fp = value


def test_index_vault_indexes_each_note(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")

    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        parser=ObsidianMarkdownParser(),
        chunker=HeadingAwareChunker(max_tokens=200),
        embeddings=embed,
        store=store,
    )
    stats = indexer.index(tmp_path, ignore=[])

    assert set(store.notes) == {"a.md", "b.md"}
    assert stats.notes == 2
    # embeddings were requested for the chunk text
    assert any("alpha content here" in t for t in embed.seen)
    # fingerprint recorded so later runs can detect model changes
    assert store.fingerprint() == "fake:v1"


def test_embedding_text_defaults_to_chunk_text(tmp_path):
    (tmp_path / "a.md").write_text("# A\nplain text")
    store = FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store).index(tmp_path, ignore=[])
    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb is None
    assert chunks[0].embedding_text == chunks[0].chunk.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.indexing.index_vault'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/indexing/index_vault.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ariostea.domain.models import ContextualizedChunk, IndexStats
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import MarkdownParser, Chunker
from ariostea.ports.store import DocumentWriter, IndexAdmin
from ariostea.indexing.scanner import scan_vault


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: DocumentWriter | IndexAdmin,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    def index(self, root: str | Path, ignore: Sequence[str] = ()) -> IndexStats:
        for scanned in scan_vault(root, ignore=ignore):
            note, body = self._parser.parse(scanned.rel_path, scanned.raw, scanned.mtime)
            chunks = self._chunker.chunk(note, body)
            if not chunks:
                continue
            cchunks = [
                ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text)
                for c in chunks
            ]
            vectors = self._embeddings.embed_documents([cc.embedding_text for cc in cchunks])
            self._store.upsert_note(note, cchunks, vectors)
        self._store.set_fingerprint(self._embeddings.fingerprint)
        return self._store.stats()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/indexing/index_vault.py tests/indexing/test_index_vault.py
git commit -m "feat(indexing): IndexVault use case (scan→parse→chunk→embed→write)"
```

---

## Task 1.7: `SearchKnowledge` use case

**Files:**
- Create: `src/ariostea/search/__init__.py`
- Create: `src/ariostea/search/search_knowledge.py`
- Create: `tests/search/test_search_knowledge.py`

**Why this shape:** The retrieval use case: embed the query, ask the `ChunkRetriever` for the nearest chunks, wrap them in a `SearchResult`. For the skeleton it's dense-only; Phase 2 inserts sparse + fusion *behind the same use case* without changing its callers. Driven by fakes in the test — again, no model needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/search/test_search_knowledge.py
from ariostea.search.search_knowledge import SearchKnowledge
from ariostea.domain.models import Query, Chunk, RetrievedChunk


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


class FakeRetriever:
    def __init__(self):
        self.last = None

    def dense(self, vec, k, filters=None):
        self.last = (vec, k, filters)
        chunk = Chunk(note_path="a.md", ordinal=0, heading_path=("A",), text="match", token_count=1)
        return [RetrievedChunk(chunk=chunk, score=0.5, dense_rank=0)]


def test_search_embeds_query_and_returns_results():
    retriever = FakeRetriever()
    uc = SearchKnowledge(embeddings=FakeEmbed(), retriever=retriever)
    result = uc.search(Query(text="hello", k=5))

    assert result.chunks[0].chunk.text == "match"
    # query was embedded and passed through with k
    assert retriever.last[0] == [5.0]  # len("hello")
    assert retriever.last[1] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/search/test_search_knowledge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.search'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/search/__init__.py
```

```python
# src/ariostea/search/search_knowledge.py
from __future__ import annotations

from ariostea.domain.models import Query, SearchResult
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever


class SearchKnowledge:
    def __init__(self, embeddings: EmbeddingProvider, retriever: ChunkRetriever) -> None:
        self._embeddings = embeddings
        self._retriever = retriever

    def search(self, query: Query) -> SearchResult:
        vec = self._embeddings.embed_query(query.text)
        hits = self._retriever.dense(vec, k=query.k, filters=query.filters)
        return SearchResult(chunks=tuple(hits))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/search/test_search_knowledge.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/search/__init__.py src/ariostea/search/search_knowledge.py tests/search/test_search_knowledge.py
git commit -m "feat(search): SearchKnowledge use case (dense retrieval)"
```

---

## Task 1.8: Composition root + MCP tools + end-to-end test

**Files:**
- Create: `src/ariostea/config/container.py`
- Modify: `src/ariostea/mcp/handlers.py` (add `search_payload`, `reindex_payload`)
- Modify: `src/ariostea/mcp/server.py` (add `search_knowledge`, `reindex` tools)
- Modify: `src/ariostea/cli.py` (wire `serve` to a real container)
- Create: `tests/test_end_to_end.py`

**Why this shape:** The composition root is the one place allowed to import both ports and concrete adapters; it reads config, builds the embedding adapter (which determines the vector dimension), builds the store with that dimension, and assembles the use cases. The MCP tools become thin wrappers over handler functions that call the use cases. The end-to-end test indexes a real fixture vault with the **real** adapters and asserts a query returns the right note — proof the whole slice works.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_end_to_end.py
import pytest

from ariostea.config.schema import Config, VaultCfg, StoreCfg
from ariostea.config.container import build_container
from ariostea.mcp.handlers import search_payload, reindex_payload, status_payload


@pytest.mark.integration
def test_index_and_search_end_to_end(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "rag.md").write_text("# Retrieval\nVector databases store embeddings for semantic search.")
    (vault / "cooking.md").write_text("# Pasta\nBoil water, add salt, cook the pasta al dente.")

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)

    reindex_payload(container)  # full index
    assert status_payload(container.admin)["notes"] == 2

    payload = search_payload(container, query="how are embeddings stored", k=1)
    assert payload["results"]
    assert payload["results"][0]["note_path"] == "rag.md"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.config.container'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/config/container.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ariostea.config.schema import Config
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.indexing.index_vault import IndexVault
from ariostea.search.search_knowledge import SearchKnowledge


@dataclass
class Container:
    config: Config
    embeddings: FastEmbedEmbeddings
    store: SqliteStore
    indexer: IndexVault
    searcher: SearchKnowledge

    @property
    def admin(self) -> SqliteStore:
        return self.store


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def build_container(config: Config) -> Container:
    # Embedding provider — local fastembed for the walking skeleton.
    embeddings = FastEmbedEmbeddings(model_name=config.embedding.local_model)

    store_path = _expand(config.store.path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path=store_path, dim=embeddings.dimension)

    parser = ObsidianMarkdownParser()
    chunker = HeadingAwareChunker()

    indexer = IndexVault(parser=parser, chunker=chunker, embeddings=embeddings, store=store)
    searcher = SearchKnowledge(embeddings=embeddings, retriever=store)

    return Container(
        config=config,
        embeddings=embeddings,
        store=store,
        indexer=indexer,
        searcher=searcher,
    )
```

```python
# src/ariostea/mcp/handlers.py   (REPLACE FILE CONTENTS)
from __future__ import annotations

import os

from ariostea.domain.models import Query
from ariostea.ports.store import IndexAdmin


def status_payload(admin: IndexAdmin) -> dict:
    s = admin.stats()
    return {
        "notes": s.notes,
        "chunks": s.chunks,
        "last_indexed": s.last_indexed,
        "config_fingerprint": s.config_fingerprint,
    }


def reindex_payload(container) -> dict:
    vault_path = os.path.expanduser(container.config.vault.path)
    stats = container.indexer.index(vault_path, ignore=container.config.vault.ignore)
    return {"notes": stats.notes, "chunks": stats.chunks}


def search_payload(container, query: str, k: int = 10) -> dict:
    result = container.searcher.search(Query(text=query, k=k))
    return {
        "results": [
            {
                "note_path": rc.chunk.note_path,
                "heading_path": list(rc.chunk.heading_path),
                "text": rc.chunk.text,
                "score": rc.score,
            }
            for rc in result.chunks
        ]
    }
```

```python
# src/ariostea/mcp/server.py   (REPLACE FILE CONTENTS)
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from ariostea.mcp.handlers import status_payload, search_payload, reindex_payload


def build_server(container) -> FastMCP:
    mcp = FastMCP("ariostea")

    @mcp.tool()
    def status() -> dict:
        """Report index health: note/chunk counts, last index time, config fingerprint."""
        return status_payload(container.admin)

    @mcp.tool()
    def reindex() -> dict:
        """Index (or re-index) the configured vault. Returns note/chunk counts."""
        return reindex_payload(container)

    @mcp.tool()
    def search_knowledge(query: str, k: int = 10) -> dict:
        """Semantic search over the vault. Returns the most relevant passages with their source notes."""
        return search_payload(container, query=query, k=k)

    return mcp
```

```python
# src/ariostea/cli.py   (REPLACE FILE CONTENTS)
from __future__ import annotations

import typer

from ariostea.config.schema import load_config
from ariostea.config.container import build_container
from ariostea.mcp.server import build_server
from ariostea.mcp.handlers import reindex_payload, status_payload

app = typer.Typer(help="Ariostea — Obsidian RAG MCP server")


@app.command()
def serve(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Run the stdio MCP server."""
    container = build_container(load_config(config))
    server = build_server(container)
    server.run()  # stdio transport


@app.command()
def reindex(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault once and exit."""
    container = build_container(load_config(config))
    result = reindex_payload(container)
    typer.echo(f"Indexed {result['notes']} notes, {result['chunks']} chunks.")


@app.command()
def status(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Print index status and exit."""
    container = build_container(load_config(config))
    typer.echo(status_payload(container.admin))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_end_to_end.py -v -m integration`
Expected: PASS (1 test — indexes the fixture vault with real fastembed + sqlite-vec and retrieves `rag.md`).

Then run the full suite (skipping slow integration tests): `uv run pytest -q -m "not integration"`
Expected: all unit tests PASS.

Then the full suite including integration: `uv run pytest -q`
Expected: all PASS.

- [ ] **Step 5: Update the `tests/mcp/test_handlers.py` import note and commit**

> The earlier `tests/mcp/test_handlers.py` still passes — `status_payload(admin)` is unchanged. No edit needed.

```bash
git add src/ariostea/config/container.py src/ariostea/mcp/handlers.py src/ariostea/mcp/server.py src/ariostea/cli.py tests/test_end_to_end.py
git commit -m "feat: wire composition root + MCP tools; end-to-end index+search"
```

---

## Definition of done (Phase 0 + 1)

- `uv run pytest -q` is green (unit + integration).
- `uv run ariostea reindex` indexes a real vault and `uv run ariostea status` reports counts.
- `uv run ariostea serve` starts a stdio MCP server exposing `status`, `reindex`, `search_knowledge`.
- Domain and use-case tests pass without loading any model (boundaries verified).

## Manual smoke test (optional, after Task 1.8)

```bash
mkdir -p /tmp/vault && printf '# Embeddings\nVectors capture meaning for semantic search.\n' > /tmp/vault/embeddings.md
printf '[vault]\npath = "/tmp/vault"\nignore = []\n\n[store]\npath = "/tmp/ariostea.db"\n' > /tmp/ariostea.toml
uv run ariostea reindex --config /tmp/ariostea.toml
uv run ariostea status  --config /tmp/ariostea.toml
```

---

## Self-review (plan vs spec)

- **Spec coverage (this slice):** Phase 0 (scaffold, domain, ports, composition root, MCP stub) → Tasks 0.1–0.5 ✓. Phase 1 (scan, parse, chunk, embed, sqlite-vec dense, `search_knowledge`) → Tasks 1.1–1.8 ✓. Deferred phases are explicitly listed and out of scope.
- **Clean-architecture conformance:** domain imports nothing internal (0.2); use cases depend only on ports (1.6, 1.7 tested with fakes, no frameworks); store schema confined to the adapter (1.5); composition root is the sole multi-layer importer (1.8); ports segregated by role (0.3).
- **Type consistency:** `EmbeddingProvider` (`embed_documents`/`embed_query`/`dimension`/`fingerprint`), `DocumentWriter.upsert_note(note, chunks, embeddings)`, `ChunkRetriever.dense(vec, k, filters)`, `IndexAdmin` (`known_hashes`/`stats`/`fingerprint`/`set_fingerprint`), `MarkdownParser.parse(path, raw, mtime) -> (Note, body)`, `Chunker.chunk(note, body)` are used identically across Tasks 0.3, 1.2–1.8. `ContextualizedChunk(chunk, context_blurb, embedding_text)` consistent in 0.2/1.5/1.6.
- **Placeholders:** none — every code step contains complete, runnable code.
