# Contextual Retrieval (Phase 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional ingestion-time contextualization — a note-level LLM blurb prepended to every chunk before embedding and FTS indexing — that gracefully degrades to today's plain-chunk behavior when no LLM is configured.

**Architecture:** Two new ports (`ChatProvider`, `Contextualizer`) with three adapters (`OpenAICompatChat`, `LLMContextualizer`, `NoopContextualizer`). `IndexVault` calls the injected contextualizer instead of hard-coding bare chunks, and re-embeds when a combined `embeddings|contextualizer` fingerprint changes. Contextualization is off by default.

**Tech Stack:** Python, httpx (new dep), pydantic config, sqlite-vec + FTS5, pytest (`integration` marker for model/network tests).

**Design:** `docs/superpowers/specs/2026-06-28-ariostea-contextual-retrieval-design.md`

---

## File Structure

- `src/ariostea/ports/chat.py` — **create**: `ChatProvider` port.
- `src/ariostea/ports/pipeline.py` — **modify**: append `Contextualizer` port.
- `src/ariostea/adapters/contextualize/__init__.py`, `noop.py`, `llm.py` — **create**: `NoopContextualizer`, `LLMContextualizer`.
- `src/ariostea/adapters/chat/__init__.py`, `openai_compat.py` — **create**: `OpenAICompatChat`.
- `src/ariostea/config/schema.py` — **modify**: `ContextualCfg` + `Config.contextual`.
- `src/ariostea/adapters/store/sqlite_store.py` — **modify**: `context_blurb` column + write it.
- `src/ariostea/indexing/index_vault.py` — **modify**: inject contextualizer, combined fingerprint.
- `src/ariostea/config/container.py` — **modify**: `_build_contextualizer`, inject into `IndexVault`.
- `ariostea.example.toml` — **modify**: document `[contextual]`.
- `pyproject.toml` — **modify**: add `httpx`.
- Tests: `tests/ports/test_protocols.py` (+2 ports), `tests/adapters/contextualize/test_noop.py`, `test_llm.py`, `tests/adapters/chat/test_openai_compat.py`, `test_openai_compat_integration.py`, `tests/config/test_schema.py` (+contextual), `tests/adapters/store/test_sqlite_store.py` (+blurb), `tests/indexing/test_index_vault.py` (modify), `tests/config/test_container.py` (+contextualizer build), `tests/indexing/test_contextual_lift.py` (create).
- `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md` — **modify**: mark Phase 5 done.

---

### Task 1: Ports + NoopContextualizer

**Files:**
- Create: `src/ariostea/ports/chat.py`
- Modify: `src/ariostea/ports/pipeline.py`
- Create: `src/ariostea/adapters/contextualize/__init__.py`, `src/ariostea/adapters/contextualize/noop.py`
- Test: `tests/adapters/contextualize/test_noop.py`, `tests/ports/test_protocols.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/contextualize/test_noop.py`:

```python
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.domain.models import Chunk, Note


def _note():
    return Note(path="a.md", title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=1.0)


def _chunk(ordinal, text):
    return Chunk(note_path="a.md", ordinal=ordinal, heading_path=("A",), text=text, token_count=len(text.split()))


def test_noop_leaves_chunk_text_unchanged():
    note = _note()
    chunks = [_chunk(0, "alpha"), _chunk(1, "beta")]

    out = NoopContextualizer().contextualize(note, "full doc", chunks)

    assert [c.embedding_text for c in out] == ["alpha", "beta"]
    assert all(c.context_blurb is None for c in out)
    assert [c.chunk for c in out] == chunks


def test_noop_fingerprint_is_stable():
    assert NoopContextualizer().fingerprint == "noop"
```

Append to `tests/ports/test_protocols.py` (add at the end of the file):

```python
def test_chat_provider_protocol():
    from ariostea.ports.chat import ChatProvider

    class Chat:
        def complete(self, system, user):
            return "ok"

    assert isinstance(Chat(), ChatProvider)
    assert not isinstance(object(), ChatProvider)


def test_contextualizer_protocol():
    from ariostea.ports.pipeline import Contextualizer

    class Ctx:
        def contextualize(self, note, full_doc, chunks):
            return []

        @property
        def fingerprint(self):
            return "x"

    assert isinstance(Ctx(), Contextualizer)
    assert not isinstance(object(), Contextualizer)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/adapters/contextualize/test_noop.py tests/ports/test_protocols.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.ports.chat'` / `cannot import name 'Contextualizer'` / `NoopContextualizer`.

- [ ] **Step 3: Create `src/ariostea/ports/chat.py`**

```python
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...
```

- [ ] **Step 4: Append `Contextualizer` to `src/ariostea/ports/pipeline.py`**

Change the imports at the top from:

```python
from typing import Protocol, runtime_checkable

from ariostea.domain.models import Chunk, Note
```

to:

```python
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ariostea.domain.models import Chunk, ContextualizedChunk, Note
```

Then append at the end of the file:

```python
@runtime_checkable
class Contextualizer(Protocol):
    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]: ...
    @property
    def fingerprint(self) -> str: ...
```

- [ ] **Step 5: Create `src/ariostea/adapters/contextualize/__init__.py`** (empty file).

- [ ] **Step 6: Create `src/ariostea/adapters/contextualize/noop.py`**

```python
from __future__ import annotations

from collections.abc import Sequence

from ariostea.domain.models import Chunk, ContextualizedChunk, Note
from ariostea.ports.pipeline import Contextualizer


class NoopContextualizer(Contextualizer):
    """No LLM: every chunk is embedded/indexed as its bare text."""

    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]:
        return [
            ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text) for c in chunks
        ]

    @property
    def fingerprint(self) -> str:
        return "noop"
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/adapters/contextualize/test_noop.py tests/ports/test_protocols.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/ariostea/ports/chat.py src/ariostea/ports/pipeline.py src/ariostea/adapters/contextualize/ tests/adapters/contextualize/test_noop.py tests/ports/test_protocols.py
git commit -m "feat(contextual): ChatProvider + Contextualizer ports + NoopContextualizer"
```

---

### Task 2: LLMContextualizer

**Files:**
- Create: `src/ariostea/adapters/contextualize/llm.py`
- Test: `tests/adapters/contextualize/test_llm.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/adapters/contextualize/test_llm.py`:

```python
from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.domain.models import Chunk, Note


def _note():
    return Note(path="a.md", title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=1.0)


def _chunk(ordinal, text):
    return Chunk(note_path="a.md", ordinal=ordinal, heading_path=("A",), text=text, token_count=len(text.split()))


class FakeChat:
    def __init__(self, reply="a situating blurb"):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


class BrokenChat:
    def complete(self, system, user):
        raise RuntimeError("provider down")


def test_blurb_is_prepended_to_every_chunk():
    chat = FakeChat("ACME Q2 report")
    ctx = LLMContextualizer(chat, model_name="m")

    out = ctx.contextualize(_note(), "the full document", [_chunk(0, "alpha"), _chunk(1, "beta")])

    assert [c.embedding_text for c in out] == ["ACME Q2 report\n\nalpha", "ACME Q2 report\n\nbeta"]
    assert all(c.context_blurb == "ACME Q2 report" for c in out)
    # the full document (not the chunk) is sent as the user content, once
    assert len(chat.calls) == 1
    assert chat.calls[0][1] == "the full document"


def test_empty_blurb_degrades_to_plain_text():
    ctx = LLMContextualizer(FakeChat("   "), model_name="m")
    out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha")])
    assert out[0].embedding_text == "alpha"
    assert out[0].context_blurb is None


def test_provider_failure_degrades_to_plain_text():
    ctx = LLMContextualizer(BrokenChat(), model_name="m")
    out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha"), _chunk(1, "beta")])
    assert [c.embedding_text for c in out] == ["alpha", "beta"]
    assert all(c.context_blurb is None for c in out)


def test_fingerprint_includes_model():
    assert LLMContextualizer(FakeChat(), model_name="gpt-4o-mini").fingerprint == "llm:gpt-4o-mini"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/adapters/contextualize/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.contextualize.llm'`.

- [ ] **Step 3: Create `src/ariostea/adapters/contextualize/llm.py`**

```python
from __future__ import annotations

import logging
from collections.abc import Sequence

from ariostea.domain.models import Chunk, ContextualizedChunk, Note
from ariostea.ports.chat import ChatProvider
from ariostea.ports.pipeline import Contextualizer

logger = logging.getLogger(__name__)

_INSTRUCTIONS = (
    "You write a short context blurb (one or two sentences, about 50 words) that situates "
    "the following note for search retrieval: state its main topic and key entities so a "
    "fragment of it can be found out of context. Output only the blurb, with no preamble."
)


class LLMContextualizer(Contextualizer):
    """Generate one note-level blurb via a ChatProvider and prepend it to every
    chunk. Any failure (or an empty blurb) degrades the whole note to plain text
    so indexing is never blocked."""

    def __init__(self, chat: ChatProvider, model_name: str) -> None:
        self._chat = chat
        self._model_name = model_name

    def contextualize(
        self, note: Note, full_doc: str, chunks: Sequence[Chunk]
    ) -> list[ContextualizedChunk]:
        try:
            blurb = self._chat.complete(system=_INSTRUCTIONS, user=full_doc).strip()
        except Exception as exc:  # provider down / timeout / bad response
            logger.warning("contextualization failed for %s (%s); indexing plain", note.path, exc)
            blurb = ""
        if not blurb:
            return [
                ContextualizedChunk(chunk=c, context_blurb=None, embedding_text=c.text)
                for c in chunks
            ]
        return [
            ContextualizedChunk(
                chunk=c, context_blurb=blurb, embedding_text=f"{blurb}\n\n{c.text}"
            )
            for c in chunks
        ]

    @property
    def fingerprint(self) -> str:
        return f"llm:{self._model_name}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/adapters/contextualize/test_llm.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/adapters/contextualize/llm.py tests/adapters/contextualize/test_llm.py
git commit -m "feat(contextual): LLMContextualizer (note-level blurb, graceful degradation)"
```

---

### Task 3: OpenAICompatChat adapter

**Files:**
- Modify: `pyproject.toml` (add `httpx`)
- Create: `src/ariostea/adapters/chat/__init__.py`, `src/ariostea/adapters/chat/openai_compat.py`
- Test: `tests/adapters/chat/test_openai_compat.py`, `tests/adapters/chat/test_openai_compat_integration.py`

- [ ] **Step 1: Add `httpx` to `pyproject.toml`**

In the `dependencies = [ ... ]` list, add the line `"httpx>=0.27",` (after `"watchfiles>=0.24",`). Then run `uv sync` so the lockfile/venv pick it up.

Run: `uv sync`
Expected: resolves and installs httpx.

- [ ] **Step 2: Write the failing tests**

Create `tests/adapters/chat/test_openai_compat.py`:

```python
import json

import httpx
import pytest

from ariostea.adapters.chat.openai_compat import ChatError, OpenAICompatChat


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_builds_request_and_parses_response():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a blurb"}}]})

    chat = OpenAICompatChat(base_url="http://x/v1", model="m", api_key="k", client=_client(handler))
    out = chat.complete(system="sys", user="usr")

    assert out == "a blurb"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_omits_auth_header_without_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    OpenAICompatChat(base_url="http://x/v1", model="m", client=_client(handler)).complete("s", "u")
    assert seen["auth"] is None


def test_raises_on_error_status():
    chat = OpenAICompatChat(
        base_url="http://x/v1", model="m", client=_client(lambda r: httpx.Response(500, text="boom"))
    )
    with pytest.raises(ChatError):
        chat.complete(system="s", user="u")
```

Create `tests/adapters/chat/test_openai_compat_integration.py`:

```python
import os

import pytest

from ariostea.adapters.chat.openai_compat import OpenAICompatChat

BASE_URL = os.environ.get("ARIOSTEA_TEST_CHAT_BASE_URL")
MODEL = os.environ.get("ARIOSTEA_TEST_CHAT_MODEL", "llama3.1")


@pytest.mark.integration
@pytest.mark.skipif(not BASE_URL, reason="set ARIOSTEA_TEST_CHAT_BASE_URL to run")
def test_real_endpoint_returns_a_blurb():
    chat = OpenAICompatChat(
        base_url=BASE_URL, model=MODEL, api_key=os.environ.get("ARIOSTEA_TEST_CHAT_API_KEY", "")
    )
    out = chat.complete(system="Reply with one short sentence.", user="Say hello.")
    assert isinstance(out, str) and out.strip()
```

- [ ] **Step 3: Run the unit tests to verify they fail**

Run: `uv run pytest tests/adapters/chat/test_openai_compat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.adapters.chat.openai_compat'`.

- [ ] **Step 4: Create `src/ariostea/adapters/chat/__init__.py`** (empty file).

- [ ] **Step 5: Create `src/ariostea/adapters/chat/openai_compat.py`**

```python
from __future__ import annotations

import httpx

from ariostea.ports.chat import ChatProvider


class ChatError(RuntimeError):
    """A chat completion request failed (bad status or transport error)."""


class OpenAICompatChat(ChatProvider):
    """Chat via any OpenAI-compatible /chat/completions endpoint (OpenAI,
    Ollama, LM Studio, vLLM, llama.cpp, …). The client is injectable for tests."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_tokens: int = 128,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ChatError(f"chat request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ChatError(f"chat completion failed: {resp.status_code} {resp.text}")
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 6: Run the unit tests to verify they pass**

Run: `uv run pytest tests/adapters/chat/test_openai_compat.py -v`
Expected: PASS (3 passed). The integration test auto-skips without an endpoint.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/ariostea/adapters/chat/ tests/adapters/chat/
git commit -m "feat(contextual): OpenAI-compatible chat adapter (httpx)"
```

---

### Task 4: Config + store column

**Files:**
- Modify: `src/ariostea/config/schema.py`
- Modify: `src/ariostea/adapters/store/sqlite_store.py`
- Test: `tests/config/test_schema.py`, `tests/adapters/store/test_sqlite_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/config/test_schema.py`:

```python
def test_contextual_defaults_off():
    from ariostea.config.schema import Config, VaultCfg

    cfg = Config(vault=VaultCfg(path="/v"))
    assert cfg.contextual.enabled is False
    assert cfg.contextual.base_url == "http://localhost:11434/v1"
    assert cfg.contextual.model == "llama3.1"
    assert cfg.contextual.max_tokens == 128
```

Append to `tests/adapters/store/test_sqlite_store.py` (the `_cchunk` helper there sets `context_blurb=None`; this test uses an explicit blurb):

```python
def test_upsert_persists_context_blurb(tmp_path):
    from ariostea.domain.models import Chunk, ContextualizedChunk

    store = SqliteStore(path=str(tmp_path / "idx.db"), dim=3)
    note = _note()
    chunk = Chunk(note_path=note.path, ordinal=0, heading_path=("A",), text="bare", token_count=1)
    cc = ContextualizedChunk(chunk=chunk, context_blurb="the blurb", embedding_text="the blurb\n\nbare")
    store.upsert_note(note, [cc], [[1.0, 0.0, 0.0]])

    rows = store.db.execute("SELECT context_blurb FROM chunks").fetchall()
    assert rows[0]["context_blurb"] == "the blurb"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/config/test_schema.py::test_contextual_defaults_off tests/adapters/store/test_sqlite_store.py::test_upsert_persists_context_blurb -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'contextual'`; and `sqlite3.OperationalError: table chunks has no column named context_blurb`.

- [ ] **Step 3: Add `ContextualCfg` to `src/ariostea/config/schema.py`**

After the `RerankCfg` class, add:

```python
class ContextualCfg(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.1"
    timeout: float = 30.0
    max_tokens: int = 128
```

And add the field to `Config` (after `rerank: RerankCfg = RerankCfg()`):

```python
    contextual: ContextualCfg = ContextualCfg()
```

- [ ] **Step 4: Add the `context_blurb` column in `src/ariostea/adapters/store/sqlite_store.py`**

In `_init_schema`, change the `chunks` table definition to add the column:

```python
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading_path TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                context_blurb TEXT
            );
```

In `upsert_note`, change the chunk INSERT to write the blurb:

```python
                cur.execute(
                    "INSERT INTO chunks(note_id, ordinal, heading_path, text, token_count, context_blurb) "
                    "VALUES (?,?,?,?,?,?)",
                    (note_id, ch.ordinal, "/".join(ch.heading_path), ch.text, ch.token_count, cc.context_blurb),
                )
```

(`cc` is the loop variable `for cc, vec in zip(chunks, embeddings):`, `ch = cc.chunk`.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/config/test_schema.py tests/adapters/store/test_sqlite_store.py -v`
Expected: PASS (all, including the existing store tests).

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/config/schema.py src/ariostea/adapters/store/sqlite_store.py tests/config/test_schema.py tests/adapters/store/test_sqlite_store.py
git commit -m "feat(contextual): [contextual] config + chunks.context_blurb column"
```

---

### Task 5: Wire the contextualizer into IndexVault

**Files:**
- Modify: `src/ariostea/indexing/index_vault.py`
- Modify: `tests/indexing/test_index_vault.py`

- [ ] **Step 1: Update `tests/indexing/test_index_vault.py` for the new signature and combined fingerprint**

The `IndexVault` constructor gains a 5th positional parameter `contextualizer`. Update the file as follows.

Add an import at the top:

```python
from ariostea.adapters.contextualize.noop import NoopContextualizer
```

Every `IndexVault(...)` call must pass a contextualizer as the 5th positional argument. Update each call site by appending `NoopContextualizer()`:
- `test_index_vault_indexes_each_note`: `IndexVault(parser=..., chunker=..., embeddings=embed, store=store, contextualizer=NoopContextualizer())`
- `test_embedding_text_defaults_to_chunk_text`: `IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, NoopContextualizer())`
- `test_index_removes_notes_deleted_from_disk`: add `contextualizer=NoopContextualizer()`
- `test_index_skips_unchanged_notes_on_reindex`: `IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store, NoopContextualizer())`
- `test_index_reembeds_only_the_changed_note`: same — append `NoopContextualizer()`
- `test_index_reembeds_all_when_fingerprint_changes`: both `IndexVault(...)` calls — append `NoopContextualizer()`

Change the fingerprint assertion in `test_index_vault_indexes_each_note` from:

```python
    assert store.fingerprint() == "fake:v1"
```

to:

```python
    assert store.fingerprint() == "fake:v1|noop"  # combined embeddings|contextualizer
```

Then add a new test at the end of the file:

```python
def test_contextualizer_output_flows_to_store(tmp_path):
    from collections.abc import Sequence

    from ariostea.domain.models import ContextualizedChunk
    from ariostea.ports.pipeline import Contextualizer

    class TitleCtx(Contextualizer):
        def contextualize(self, note, full_doc, chunks):
            return [
                ContextualizedChunk(chunk=c, context_blurb=note.title, embedding_text=f"{note.title}\n\n{c.text}")
                for c in chunks
            ]

        @property
        def fingerprint(self):
            return "titlectx"

    (tmp_path / "a.md").write_text("# Topic\nbody text")
    store = FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, TitleCtx()).index(
        tmp_path, ignore=[]
    )

    _, chunks, _ = store.notes["a.md"]
    assert chunks[0].context_blurb == "Topic"
    assert chunks[0].embedding_text.startswith("Topic\n\n")
    assert store.fingerprint() == "fake:v1|titlectx"  # contextualizer in the fingerprint
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: FAIL — `TypeError: __init__() missing 1 required positional argument: 'contextualizer'` (and the new test errors).

- [ ] **Step 3: Update `src/ariostea/indexing/index_vault.py`**

Replace the whole file with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ariostea.domain.models import IndexStats
from ariostea.indexing.scanner import scan_vault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, Contextualizer, MarkdownParser
from ariostea.ports.store import IndexStore


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: IndexStore,
        contextualizer: Contextualizer,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store
        self._contextualizer = contextualizer

    def _fingerprint(self) -> str:
        # Both the embedding model AND the contextualization change embedding_text,
        # so a change in either must invalidate every stored vector.
        return f"{self._embeddings.fingerprint}|{self._contextualizer.fingerprint}"

    def index(self, root: str | Path, ignore: Sequence[str] = ()) -> IndexStats:
        seen: set[str] = set()
        known = self._store.known_hashes()
        fingerprint_changed = self._store.fingerprint() != self._fingerprint()

        for scanned in scan_vault(root, ignore=ignore):
            if not fingerprint_changed and known.get(scanned.rel_path) == scanned.content_hash:
                seen.add(scanned.rel_path)  # unchanged & already indexed — keep it
                continue
            note, body = self._parser.parse(scanned.rel_path, scanned.raw, scanned.mtime)
            chunks = self._chunker.chunk(note, body)
            if not chunks:
                continue
            cchunks = self._contextualizer.contextualize(note, body, chunks)
            vectors = self._embeddings.embed_documents([cc.embedding_text for cc in cchunks])
            self._store.upsert_note(note, cchunks, vectors)
            seen.add(note.path)
        for path in list(self._store.known_hashes()):
            if path not in seen:
                self._store.delete_note(path)
        self._store.set_fingerprint(self._fingerprint())
        return self._store.stats()
```

(Note: this drops the now-unused `ContextualizedChunk` import that the previous version used inline.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: PASS (all, including the new test).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/indexing/index_vault.py tests/indexing/test_index_vault.py
git commit -m "feat(contextual): inject contextualizer into IndexVault + combined fingerprint"
```

---

### Task 6: Container wiring + example config

**Files:**
- Modify: `src/ariostea/config/container.py`
- Modify: `ariostea.example.toml`
- Test: `tests/config/test_container.py`

- [ ] **Step 1: Write the failing test**

Create (or append to) `tests/config/test_container.py`:

```python
from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.config.container import _build_contextualizer
from ariostea.config.schema import ContextualCfg


def test_build_contextualizer_noop_when_disabled():
    ctx = _build_contextualizer(ContextualCfg(enabled=False))
    assert isinstance(ctx, NoopContextualizer)


def test_build_contextualizer_llm_when_enabled():
    ctx = _build_contextualizer(ContextualCfg(enabled=True, model="m", base_url="http://x/v1"))
    assert isinstance(ctx, LLMContextualizer)
    assert ctx.fingerprint == "llm:m"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/config/test_container.py -v`
Expected: FAIL — `ImportError: cannot import name '_build_contextualizer'`.

- [ ] **Step 3: Update `src/ariostea/config/container.py`**

Add imports near the other adapter imports:

```python
from ariostea.adapters.chat.openai_compat import OpenAICompatChat
from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.config.schema import Config, ContextualCfg, RerankCfg
from ariostea.ports.pipeline import Contextualizer
```

(The existing `from ariostea.config.schema import Config, RerankCfg` line is replaced by the one above.)

Add the builder near `_build_reranker`:

```python
def _build_contextualizer(cfg: ContextualCfg) -> Contextualizer:
    """Build the configured contextualizer, degrading to NoopContextualizer
    (plain chunks) with a warning if disabled or the chat client can't be built."""
    if not cfg.enabled:
        return NoopContextualizer()
    try:
        chat = OpenAICompatChat(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_tokens=cfg.max_tokens,
        )
        return LLMContextualizer(chat, model_name=cfg.model)
    except Exception as exc:  # misconfiguration
        logger.warning("contextualizer unavailable (%s); indexing plain chunks", exc)
        return NoopContextualizer()
```

In `build_container`, change the `IndexVault(...)` construction to inject the contextualizer:

```python
    indexer = IndexVault(
        parser=parser,
        chunker=chunker,
        embeddings=embeddings,
        store=store,
        contextualizer=_build_contextualizer(config.contextual),
    )
```

- [ ] **Step 4: Document `[contextual]` in `ariostea.example.toml`**

Append:

```toml
[contextual]
# Prepend an LLM-written note-level blurb to each chunk before embedding/indexing.
# Off by default; the zero-key install runs without it (plain hybrid search).
enabled = false
base_url = "http://localhost:11434/v1"   # any OpenAI-compatible endpoint (Ollama, OpenAI, vLLM, …)
api_key = ""                              # leave empty for local/keyless servers
model = "llama3.1"
timeout = 30.0
max_tokens = 128
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/config/test_container.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/config/container.py ariostea.example.toml tests/config/test_container.py
git commit -m "feat(contextual): wire contextualizer into the container + example config"
```

---

### Task 7: Demonstrated lift (fast, model-free)

Proves the mechanism end-to-end on the sparse channel with a fake embedding provider (no model download, fully deterministic): a note whose body never names its topic becomes findable once a note-level blurb is prepended.

**Files:**
- Create: `tests/indexing/test_contextual_lift.py`

- [ ] **Step 1: Write the test**

Create `tests/indexing/test_contextual_lift.py`:

```python
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.domain.models import ContextualizedChunk
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.pipeline import Contextualizer


class FakeEmbed:
    """Sparse-channel test: vectors are irrelevant, so return constant dummies."""

    def embed_documents(self, texts):
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [0.0, 0.0]

    @property
    def dimension(self):
        return 2

    @property
    def fingerprint(self):
        return "fake:v1"


class TitleStubContextualizer(Contextualizer):
    """Deterministic stand-in for the LLM: prepends the note title as the blurb."""

    def contextualize(self, note, full_doc, chunks):
        return [
            ContextualizedChunk(
                chunk=c, context_blurb=note.title, embedding_text=f"{note.title}\n\n{c.text}"
            )
            for c in chunks
        ]

    @property
    def fingerprint(self):
        return "stub"


def _index(vault, db_path, contextualizer):
    store = SqliteStore(path=str(db_path), dim=2)
    IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(), FakeEmbed(), store, contextualizer
    ).index(vault, ignore=[])
    return store


def test_contextualization_makes_an_ambiguous_chunk_findable(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # No H1, so the title comes from the filename ("piano"); the body never says "piano".
    (vault / "piano.md").write_text("It has 88 of them, black and white, struck by felt hammers.")

    # Without context: the bare chunk has no "piano" token -> sparse keyword search misses.
    plain = _index(vault, tmp_path / "plain.db", NoopContextualizer())
    assert plain.sparse("piano", k=5) == []

    # With note-level context (title prepended): "piano" is now indexed -> found.
    ctx = _index(vault, tmp_path / "ctx.db", TitleStubContextualizer())
    hits = ctx.sparse("piano", k=5)
    assert [h.chunk.note_path for h in hits] == ["piano.md"]
```

- [ ] **Step 2: Run the test (it should pass immediately — characterization)**

Run: `uv run pytest tests/indexing/test_contextual_lift.py -v`
Expected: PASS. This depends only on already-built pieces (IndexVault wiring + Noop). If `plain.sparse("piano", ...)` is unexpectedly non-empty, confirm the fixture has **no** `#` H1 line (an H1 would put "piano" into the chunk text via the heading) — do not weaken the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/indexing/test_contextual_lift.py
git commit -m "test(contextual): demonstrate retrieval lift from note-level context"
```

---

### Task 8: Mark Phase 5 done in the PRD

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`

- [ ] **Step 1: Update the roadmap row and §15**

In the roadmap table, change the Phase 5 row from `| 5 | Contextual Retrieval (contextualizer + prompt caching) | Blurbs stored; retrieval quality improves on eval set |` to mark it done, e.g. `| 5 ✅ |`, with a short note: "note-level blurb via OpenAI-compatible chat; per-chunk + prompt caching deferred (see contextual-retrieval design §10)".

In §15, add one sentence recording the note-level decision and linking to `2026-06-28-ariostea-contextual-retrieval-design.md` (the per-chunk method and explicit prompt caching are future enhancements).

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-ariostea-rag-design.md
git commit -m "docs: mark Phase 5 (contextual retrieval) done"
```

---

## Notes for the implementer

- **Run order:** Task 1 (ports/Noop) before everything; Task 5 (IndexVault) needs Task 1's `NoopContextualizer`; Task 6 (container) needs Tasks 1–4; Task 7 needs Task 5.
- **Don't weaken the lift assertion.** If `plain.sparse("piano")` is non-empty, the fixture leaked the term (most likely an H1) — fix the fixture, not the test.
- **No native Anthropic / no per-chunk / no HTTP embeddings** — all explicitly out of scope (design §10/§11).
- **Default off:** `[contextual].enabled = false` keeps the zero-key install identical to today. Contextualization is opt-in.
- **Fast suite stays model-free:** only the OpenAI-compat *integration* test touches the network (and auto-skips); the lift test uses a fake embedder. Run `uv run pytest -m "not integration" -q` between tasks.
