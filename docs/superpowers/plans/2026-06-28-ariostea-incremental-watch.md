# Incremental Indexing + File Watcher (Phase 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `reindex` skip unchanged notes (re-embed only what changed) and add a `watch` mode that keeps the index current automatically as the vault is edited.

**Architecture:** `IndexVault.index()` gains a hash gate — it snapshots `known_hashes()` and skips any scanned file whose `content_hash` already matches, *unless* the embedding fingerprint changed (model swap → force full re-embed). A new `WatchVault` use case wraps `watchfiles.watch` (injected for testability) to re-run the now-cheap incremental index on each change batch, exposed via a new `ariostea watch` CLI command.

**Tech Stack:** Python 3.12, `watchfiles` (already a dependency), pytest, typer (CLI), the existing `IndexVault` / scanner / `IndexAdmin` port.

**Source spec:** PRD roadmap Phase 4 — *"Editing a note updates only that note's chunks; unchanged files are skipped; live updates via watcher (deletion sweep already landed in Phase 3)."*

---

## Notes that shaped this plan (verified against the code)

- The scanner (`scan_vault`) already yields `ScannedFile.content_hash = sha256(raw)`, and the parser stores `Note.content_hash` with the **identical** computation — so `scanned.content_hash == stored note.content_hash` for unchanged files. The gate is sound.
- `IndexAdmin` already exposes `known_hashes() -> dict[path, content_hash]` and `fingerprint()` / `set_fingerprint()`. No port changes needed.
- **Deletion already works** (Phase 3 sweep) — Phase 4 is only *skip-unchanged* + *watch*.
- **The fingerprint hazard:** the old `index()` re-embedded everything every run, so a model swap was handled implicitly. Adding a skip-gate breaks that — unchanged content would skip re-embedding even though the new model needs fresh vectors. So the gate must bypass skipping when `store.fingerprint() != embeddings.fingerprint`.
- `watchfiles.watch(*paths, stop_event=None, ...)` yields `set[tuple[Change, str]]` per debounced batch. Injecting it as `watch_fn` keeps `WatchVault` unit-testable without real filesystem events.
- Only integration-marked tests build the real container; `IndexVault` and `WatchVault` are tested with fakes in the fast suite.

## File Structure

- Modify: `src/ariostea/indexing/index_vault.py` — add the hash gate + fingerprint guard to `index()`.
- Create: `src/ariostea/indexing/watch_vault.py` — `WatchVault` use case.
- Modify: `src/ariostea/cli.py` — add the `watch` command.
- Modify: `README.md` — mention `ariostea watch` in the quick start.
- Test: `tests/indexing/test_index_vault.py` (append), `tests/indexing/test_watch_vault.py` (new), `tests/test_cli.py` (new).

---

### Task 1: Incremental skip with fingerprint guard

**Files:**
- Modify: `src/ariostea/indexing/index_vault.py`
- Test: `tests/indexing/test_index_vault.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/indexing/test_index_vault.py`:

```python
def test_index_skips_unchanged_notes_on_reindex(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")
    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store
    )
    indexer.index(tmp_path, ignore=[])

    # Second run, nothing changed on disk: no text should be re-embedded.
    embed.seen.clear()
    stats = indexer.index(tmp_path, ignore=[])
    assert embed.seen == []
    assert set(store.notes) == {"a.md", "b.md"}  # unchanged notes are kept, not swept
    assert stats.notes == 2


def test_index_reembeds_only_the_changed_note(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    (tmp_path / "b.md").write_text("# B\nbeta content here")
    embed, store = FakeEmbed(), FakeStore()
    indexer = IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store
    )
    indexer.index(tmp_path, ignore=[])

    (tmp_path / "a.md").write_text("# A\nalpha content CHANGED now")
    embed.seen.clear()
    indexer.index(tmp_path, ignore=[])
    assert any("CHANGED" in t for t in embed.seen)  # changed note re-embedded
    assert not any("beta" in t for t in embed.seen)  # unchanged note skipped


def test_index_reembeds_all_when_fingerprint_changes(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content here")
    embed, store = FakeEmbed(), FakeStore()
    IndexVault(ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed, store).index(
        tmp_path, ignore=[]
    )

    # Simulate a model swap: same content, different fingerprint -> must re-embed.
    class FakeEmbed2(FakeEmbed):
        @property
        def fingerprint(self):
            return "fake:v2"

    embed2 = FakeEmbed2()
    IndexVault(
        ObsidianMarkdownParser(), HeadingAwareChunker(max_tokens=200), embed2, store
    ).index(tmp_path, ignore=[])
    assert any("alpha" in t for t in embed2.seen)  # re-embedded despite unchanged content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: `test_index_skips_unchanged_notes_on_reindex` and `test_index_reembeds_only_the_changed_note` FAIL (the current code re-embeds everything, so `embed.seen` is not empty / contains "beta"). `test_index_reembeds_all_when_fingerprint_changes` passes incidentally (current code always re-embeds) — that's fine; it guards the next step.

- [ ] **Step 3: Add the hash gate + fingerprint guard**

Replace the entire contents of `src/ariostea/indexing/index_vault.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ariostea.domain.models import ContextualizedChunk, IndexStats
from ariostea.indexing.scanner import scan_vault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Chunker, MarkdownParser
from ariostea.ports.store import IndexStore


class IndexVault:
    def __init__(
        self,
        parser: MarkdownParser,
        chunker: Chunker,
        embeddings: EmbeddingProvider,
        store: IndexStore,
    ) -> None:
        self._parser = parser
        self._chunker = chunker
        self._embeddings = embeddings
        self._store = store

    def index(self, root: str | Path, ignore: Sequence[str] = ()) -> IndexStats:
        seen: set[str] = set()
        known = self._store.known_hashes()
        # A model swap invalidates every stored vector, so the content-hash skip
        # must be bypassed when the embedding fingerprint changed.
        fingerprint_changed = self._store.fingerprint() != self._embeddings.fingerprint

        for scanned in scan_vault(root, ignore=ignore):
            if not fingerprint_changed and known.get(scanned.rel_path) == scanned.content_hash:
                seen.add(scanned.rel_path)  # unchanged & already indexed — keep it
                continue
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
            seen.add(note.path)

        for path in list(self._store.known_hashes()):
            if path not in seen:
                self._store.delete_note(path)
        self._store.set_fingerprint(self._embeddings.fingerprint)
        return self._store.stats()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/indexing/test_index_vault.py -v`
Expected: PASS (all 6 — the 3 originals plus the 3 new).

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/indexing/index_vault.py tests/indexing/test_index_vault.py
git commit -m "feat(index): skip unchanged notes; re-embed all on fingerprint change"
```

---

### Task 2: WatchVault use case

**Files:**
- Create: `src/ariostea/indexing/watch_vault.py`
- Test: `tests/indexing/test_watch_vault.py`

- [ ] **Step 1: Write the failing test**

Create `tests/indexing/test_watch_vault.py`:

```python
from ariostea.indexing.watch_vault import WatchVault


class FakeIndexer:
    def __init__(self):
        self.calls = []

    def index(self, root, ignore=()):
        self.calls.append((str(root), tuple(ignore)))


def test_watch_indexes_once_initially_then_per_change_batch():
    idx = FakeIndexer()

    def fake_watch(root, stop_event=None):
        yield {("modified", "a.md")}
        yield {("modified", "b.md")}

    WatchVault(idx, "/vault", ignore=[".obsidian/"], watch_fn=fake_watch).run()

    # 1 initial full index + 1 per change batch = 3
    assert len(idx.calls) == 3
    # every call targets the configured root + ignore
    assert all(call == ("/vault", (".obsidian/",)) for call in idx.calls)


def test_watch_passes_stop_event_to_watch_fn():
    idx = FakeIndexer()
    received = {}

    def fake_watch(root, stop_event=None):
        received["stop_event"] = stop_event
        return iter(())  # no change batches

    sentinel = object()
    WatchVault(idx, "/vault", ignore=[], watch_fn=fake_watch).run(stop_event=sentinel)

    assert received["stop_event"] is sentinel
    assert len(idx.calls) == 1  # only the initial index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_watch_vault.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.indexing.watch_vault'`

- [ ] **Step 3: Write the implementation**

Create `src/ariostea/indexing/watch_vault.py`:

```python
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from watchfiles import watch as _default_watch

from ariostea.indexing.index_vault import IndexVault

# A watcher: given a path (and stop_event), yield a batch per change.
WatchFn = Callable[..., Iterable[object]]


class WatchVault:
    """Keep the index current: do an initial incremental index, then re-index
    on every filesystem change batch. The watch function is injected so the
    loop is testable without real filesystem events."""

    def __init__(
        self,
        indexer: IndexVault,
        root: str | Path,
        ignore: Sequence[str] = (),
        watch_fn: WatchFn = _default_watch,
    ) -> None:
        self._indexer = indexer
        self._root = root
        self._ignore = tuple(ignore)
        self._watch_fn = watch_fn

    def run(self, stop_event: object | None = None) -> None:
        self._indexer.index(self._root, ignore=self._ignore)  # initial sync
        for _changes in self._watch_fn(self._root, stop_event=stop_event):
            self._indexer.index(self._root, ignore=self._ignore)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_watch_vault.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/indexing/watch_vault.py tests/indexing/test_watch_vault.py
git commit -m "feat(index): WatchVault — re-index on filesystem changes"
```

---

### Task 3: `ariostea watch` CLI command

**Files:**
- Modify: `src/ariostea/cli.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
from types import SimpleNamespace

from typer.testing import CliRunner

from ariostea import cli


def test_watch_command_builds_container_and_runs_watcher(monkeypatch, tmp_path):
    fake_container = SimpleNamespace(
        config=SimpleNamespace(vault=SimpleNamespace(path=str(tmp_path), ignore=[".obsidian/"])),
        indexer=object(),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: None)
    monkeypatch.setattr(cli, "build_container", lambda cfg: fake_container)

    recorded = {}

    class FakeWatchVault:
        def __init__(self, indexer, root, ignore=()):
            recorded["root"] = root
            recorded["ignore"] = list(ignore)

        def run(self, stop_event=None):
            recorded["ran"] = True

    monkeypatch.setattr(cli, "WatchVault", FakeWatchVault)

    result = CliRunner().invoke(cli.app, ["watch", "--config", "x.toml"])

    assert result.exit_code == 0
    assert recorded["ran"] is True
    assert recorded["root"] == str(tmp_path)
    assert recorded["ignore"] == [".obsidian/"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL — `AttributeError: module 'ariostea.cli' has no attribute 'WatchVault'`

- [ ] **Step 3: Add the `watch` command**

In `src/ariostea/cli.py`, add `import os` at the top (after `from __future__ import annotations`), add the import `from ariostea.indexing.watch_vault import WatchVault` with the other `ariostea` imports, and add this command after the existing `reindex` command:

```python
@app.command()
def watch(config: str = typer.Option("ariostea.toml", help="Path to config file")) -> None:
    """Index the vault, then watch for changes and re-index incrementally."""
    container = build_container(load_config(config))
    vault = os.path.expanduser(container.config.vault.path)
    typer.echo(f"Indexing and watching {vault} (Ctrl-C to stop)...")
    WatchVault(container.indexer, vault, ignore=container.config.vault.ignore).run()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Document the command in the README**

In `README.md`, find the quick-start "Other commands" block:

```bash
# Other commands
uv run ariostea status                   # print index health
uv run pytest -m "not integration"       # fast test suite
```

Replace it with:

```bash
# Other commands
uv run ariostea watch                    # index, then auto-reindex on vault edits
uv run ariostea status                   # print index health
uv run pytest -m "not integration"       # fast test suite
```

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/cli.py tests/test_cli.py README.md
git commit -m "feat(cli): ariostea watch command for live incremental indexing"
```

---

### Task 4: Manual smoke test + roadmap tick

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`

- [ ] **Step 1: Full suite green**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS (fast suite, including the new index/watch/CLI tests).

- [ ] **Step 2: Manual watch smoke test (optional but recommended)**

With a real `ariostea.toml` pointing at a small vault, run in one terminal:

Run: `uv run ariostea watch`
Expected: prints "Indexing and watching <vault> ..." then stays running. Edit a note in the vault and save; the process should re-index (observable as a brief pause / no error). Stop with Ctrl-C. This is a human check — no assertion.

- [ ] **Step 3: Tick the roadmap**

In `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`, find the Phase 4 row:

```markdown
| 4 | Incremental **skip of unchanged files** (hash/mtime gate) + watcher | Editing a note updates only that note's chunks; unchanged files are skipped; live updates via watcher (deletion sweep already landed in Phase 3) |
```

Replace it with (mark done):

```markdown
| 4 ✅ | Incremental **skip of unchanged files** (content-hash gate, fingerprint-guarded) + `watch` command | DONE — unchanged notes skipped on reindex; model swap forces full re-embed; `ariostea watch` re-indexes live via watchfiles |
```

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-ariostea-rag-design.md
git commit -m "docs: mark Phase 4 (incremental + watch) done"
```

---

## Notes for the implementer

- **Why the fingerprint guard is essential:** the skip-gate compares only content hashes. A model swap changes the *vectors*, not the *text*, so without the guard every note would be wrongly skipped and the index left full of stale-model vectors. `fingerprint_changed` bypasses the skip so the swap triggers a full re-embed. (Dimension *changes* still require wiping the DB — a separate, pre-existing schema limitation.)
- **Why mtime isn't used as the gate:** content hash is the source of truth (a touched-but-unchanged file shouldn't re-embed; a changed file with a reset mtime still should). `mtime` is available on `ScannedFile` for a future cheap pre-filter, but the hash gate is correct and sufficient now — YAGNI on the mtime optimization.
- **Why `WatchVault` just calls `index()` per batch:** since indexing is now incremental, a full `index()` on each change only re-embeds what changed — cheap. No need to thread individual file paths through; the gate handles it. Simpler and correct.
- **Why `watch_fn` is injected:** `watchfiles.watch` blocks on real filesystem events, untestable in a unit test. Injecting it (like the eval harness's `search_fn`) lets the test drive the loop deterministically.
- **CLI isn't deeply tested:** the `watch` command is a thin driving adapter; the test confirms wiring (container built, watcher constructed with the right root/ignore, `run` called) via monkeypatch, without blocking on a real watch loop.
```
