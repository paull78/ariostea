# Contextual-Lift Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-LLM, one-off eval that indexes a context-dependent English corpus twice (contextualization OFF vs ON against a local Ollama) and reports the per-channel recall@k / MRR lift.

**Architecture:** Reuse the existing eval harness (`evaluate` / `format_report` / `load_gold` / `dedupe`) and per-channel search functions. Add three pure, unit-tested helpers in a new `src/ariostea/eval/contextual.py` (delta formatting, an all-blurbs integrity check, a blurb-readback reader), a new English-only context-dependent fixture corpus + gold set, and a standalone runner script that wires it together. One DRY extraction moves `make_hybrid_search_fn` into the shared `channels.py`.

**Tech Stack:** Python, pytest, fastembed (multilingual embeddings), sqlite-vec + FTS5, an OpenAI-compatible chat endpoint (Ollama).

**Spec:** `docs/superpowers/specs/2026-06-29-ariostea-contextual-lift-eval-design.md`

**Spec refinement note (read before starting):** The spec (§5.1) placed `format_delta` "in the runner script." This plan instead puts `format_delta`, the all-blurbs guard, and the blurb reader in a new package module `src/ariostea/eval/contextual.py` so they are importable and unit-testable (the spec's §6 testing requirements demand this). The runner script imports them and stays thin glue. This is the only deviation from the spec.

**Conventions (this repo):**
- Flat tests: NO `tests/**/__init__.py`, unique test-file basenames.
- Every source file starts with `from __future__ import annotations`.
- `uv run pytest -m "not integration"` is the fast suite; `ruff check .` must stay clean.
- Adapters subclass their port Protocol; eval code is a plain consumer of ports (no new ports here).

---

### Task 1: Extract `make_hybrid_search_fn` into shared `channels.py`

DRY: the hybrid search function is currently inlined in `eval/run_eval.py`. Move it into `src/ariostea/eval/channels.py` next to the dense/sparse factories so both runners share one definition.

**Files:**
- Modify: `src/ariostea/eval/channels.py`
- Modify: `eval/run_eval.py`
- Test: `tests/eval/test_channels.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_channels.py`:

```python
from ariostea.eval.channels import make_hybrid_search_fn


class _Chunk:
    def __init__(self, note_path):
        self.note_path = note_path


class _Ranked:
    def __init__(self, note_path):
        self.chunk = _Chunk(note_path)
        self.score = 1.0


class _Result:
    def __init__(self, chunks):
        self.chunks = chunks


class _Searcher:
    def __init__(self, ranked):
        self._ranked = ranked
        self.last = None

    def search(self, query):
        self.last = query
        return _Result(self._ranked)


class _Container:
    def __init__(self, ranked):
        self.searcher = _Searcher(ranked)


def test_hybrid_search_fn_dedupes_to_notes_and_truncates():
    c = _Container([_Ranked("a.md"), _Ranked("a.md"), _Ranked("b.md")])
    fn = make_hybrid_search_fn(c, pool=50)

    assert fn("q", 5) == ["a.md", "b.md"]  # two chunks of a.md collapse to one note
    assert fn("q", 1) == ["a.md"]  # truncates to k
    assert c.searcher.last.text == "q"  # query text forwarded
    assert c.searcher.last.k == 50  # pool forwarded as the search k
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_channels.py::test_hybrid_search_fn_dedupes_to_notes_and_truncates -v`
Expected: FAIL with `ImportError: cannot import name 'make_hybrid_search_fn'`.

- [ ] **Step 3: Add `make_hybrid_search_fn` to `channels.py`**

In `src/ariostea/eval/channels.py`, update the imports and append the factory. The new top of the file:

```python
"""Per-channel search functions for the eval harness.

Each factory returns a SearchFn (query, k -> note paths) that exercises a
single retrieval channel in isolation, so the harness can attribute results
to the dense or sparse side rather than only the blended pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariostea.eval.harness import SearchFn, dedupe
from ariostea.mcp.handlers import search_payload
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever

if TYPE_CHECKING:
    from ariostea.config.container import Container
```

Then append at the end of the file:

```python
def make_hybrid_search_fn(container: "Container", pool: int) -> SearchFn:
    """Full blended pipeline (dense+sparse+fuse+rerank) via the production
    search use case, deduped to notes. Pulls a generous chunk pool, then
    collapses to note paths before taking the top k."""

    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=pool)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    return search_fn
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_channels.py -v`
Expected: PASS (all channel tests).

- [ ] **Step 5: Update `run_eval.py` to import the shared helper**

In `eval/run_eval.py`, delete the local `make_hybrid_search_fn` definition (lines defining `def make_hybrid_search_fn(container): ...`) and import it instead. Change the channels import line:

```python
from ariostea.eval.channels import (
    make_dense_search_fn,
    make_hybrid_search_fn,
    make_sparse_search_fn,
)
```

And update the `HYBRID` wiring to pass the pool explicitly (it previously closed over the module-level `CHUNK_POOL`):

```python
        channels = {
            "DENSE": make_dense_search_fn(embeddings, store, CHUNK_POOL),
            "SPARSE": make_sparse_search_fn(store, CHUNK_POOL),
            "HYBRID": make_hybrid_search_fn(container, CHUNK_POOL),
        }
```

- [ ] **Step 6: Verify nothing regressed**

Run: `uv run pytest -m "not integration" -q && ruff check .`
Expected: all pass, ruff clean. (The `run_eval.py` script is not imported by tests, so confirm it still parses:)
Run: `uv run python -c "import ast; ast.parse(open('eval/run_eval.py').read())"`
Expected: no output (parses cleanly).

- [ ] **Step 7: Commit**

```bash
git add src/ariostea/eval/channels.py eval/run_eval.py tests/eval/test_channels.py
git commit -m "refactor(eval): share make_hybrid_search_fn across runners"
```

---

### Task 2: `format_delta` — OFF→ON per-scenario delta table

**Files:**
- Create: `src/ariostea/eval/contextual.py`
- Test: `tests/eval/test_contextual_helpers.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_contextual_helpers.py`:

```python
from ariostea.eval.contextual import format_delta
from ariostea.eval.harness import EvalReport, ScenarioScore


def _report(buried, overall):
    return EvalReport(k=5, overall=overall, by_scenario=(buried,))


def test_format_delta_shows_before_after_and_signed_delta():
    off = _report(
        ScenarioScore("buried", 5, 0.200, 0.150),
        ScenarioScore("overall", 5, 0.200, 0.150),
    )
    on = _report(
        ScenarioScore("buried", 5, 0.800, 0.700),
        ScenarioScore("overall", 5, 0.800, 0.700),
    )
    out = format_delta(off, on)

    assert "buried" in out
    assert "0.200 → 0.800 (+0.600)" in out  # recall pair
    assert "0.150 → 0.700 (+0.550)" in out  # mrr pair
    assert "overall" in out  # overall row present


def test_format_delta_renders_negative_and_zero_deltas():
    off = _report(
        ScenarioScore("direct", 2, 1.000, 1.000),
        ScenarioScore("overall", 2, 1.000, 1.000),
    )
    on = _report(
        ScenarioScore("direct", 2, 1.000, 0.500),
        ScenarioScore("overall", 2, 1.000, 0.500),
    )
    out = format_delta(off, on)

    assert "1.000 → 1.000 (+0.000)" in out  # zero delta is explicitly signed
    assert "1.000 → 0.500 (-0.500)" in out  # regressions show a minus
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.contextual'`.

- [ ] **Step 3: Create `contextual.py` with `format_delta`**

Create `src/ariostea/eval/contextual.py`:

```python
"""Pure helpers for the contextual-lift eval (run_contextual_eval.py).

Kept here, in the package, rather than in the runner script so they can be
unit-tested without a live model or database.
"""

from __future__ import annotations

import sqlite3

from ariostea.eval.harness import EvalReport


def _pair(before: float, after: float) -> str:
    return f"{before:.3f} → {after:.3f} ({after - before:+.3f})"


def format_delta(off: EvalReport, on: EvalReport) -> str:
    """Render one OFF→ON row per scenario (plus overall) for a single channel.

    Both reports come from the same gold file, so their scenario sets match;
    rows are paired by scenario name.
    """
    before = {s.scenario: s for s in (*off.by_scenario, off.overall)}
    header = f"{'scenario':<12} {'n':>3}  {'recall@' + str(on.k):<24} mrr"
    lines = [header]
    for s in (*on.by_scenario, on.overall):
        b = before[s.scenario]
        recall = _pair(b.recall_at_k, s.recall_at_k)
        mrr = _pair(b.mrr, s.mrr)
        lines.append(f"{s.scenario:<12} {s.n:>3}  {recall:<24} {mrr}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/contextual.py tests/eval/test_contextual_helpers.py
git commit -m "feat(eval): format_delta for OFF->ON contextual lift tables"
```

---

### Task 3: `find_uncontextualized_notes` — the all-blurbs integrity guard

**Files:**
- Modify: `src/ariostea/eval/contextual.py`
- Test: `tests/eval/test_contextual_helpers.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_contextual_helpers.py`:

```python
from ariostea.eval.contextual import find_uncontextualized_notes


def test_all_blurbs_present_returns_empty():
    rows = [("a.md", "blurb"), ("a.md", "blurb"), ("b.md", "x")]
    assert find_uncontextualized_notes(rows) == []


def test_any_null_blurb_flags_its_note():
    rows = [("a.md", "blurb"), ("b.md", None), ("b.md", None)]
    assert find_uncontextualized_notes(rows) == ["b.md"]


def test_partial_and_empty_blurbs_flagged_and_sorted():
    # z.md has an empty-string blurb; a.md has one null among its chunks.
    rows = [("z.md", ""), ("a.md", None), ("a.md", "ok")]
    assert find_uncontextualized_notes(rows) == ["a.md", "z.md"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -k uncontextualized -v`
Expected: FAIL with `ImportError: cannot import name 'find_uncontextualized_notes'`.

- [ ] **Step 3: Add `find_uncontextualized_notes` to `contextual.py`**

Append to `src/ariostea/eval/contextual.py`:

```python
def find_uncontextualized_notes(rows: list[tuple[str, str | None]]) -> list[str]:
    """Given (note_path, context_blurb) rows (one per chunk), return the sorted,
    distinct note paths that have any null/empty blurb.

    A non-empty result means the ON index is only partially contextualized, so
    an OFF-vs-ON comparison would be confounded — the runner aborts on it.
    """
    missed = {path for path, blurb in rows if not blurb}
    return sorted(missed)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -v`
Expected: PASS (all helper tests).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/contextual.py tests/eval/test_contextual_helpers.py
git commit -m "feat(eval): all-blurbs guard for contextual-lift integrity"
```

---

### Task 4: `read_blurb_rows` — read (note_path, blurb) back from a DB

**Files:**
- Modify: `src/ariostea/eval/contextual.py`
- Test: `tests/eval/test_contextual_helpers.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_contextual_helpers.py`:

```python
import sqlite3

from ariostea.eval.contextual import read_blurb_rows


def test_read_blurb_rows_joins_notes_and_chunks(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE notes (id INTEGER PRIMARY KEY, path TEXT, title TEXT,
                            content_hash TEXT, mtime REAL);
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, note_id INTEGER, ordinal INTEGER,
                             heading_path TEXT, text TEXT, token_count INTEGER,
                             context_blurb TEXT);
        """
    )
    con.execute("INSERT INTO notes(id, path, title, content_hash, mtime) VALUES (1,'a.md','A','h',0.0)")
    con.execute(
        "INSERT INTO chunks(note_id, ordinal, heading_path, text, token_count, context_blurb) "
        "VALUES (1,0,'A','t',1,'blurb'), (1,1,'B','t2',1,NULL)"
    )
    con.commit()
    con.close()

    rows = read_blurb_rows(str(db))

    assert ("a.md", "blurb") in rows
    assert ("a.md", None) in rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -k read_blurb_rows -v`
Expected: FAIL with `ImportError: cannot import name 'read_blurb_rows'`.

- [ ] **Step 3: Add `read_blurb_rows` to `contextual.py`**

Append to `src/ariostea/eval/contextual.py`:

```python
def read_blurb_rows(db_path: str) -> list[tuple[str, str | None]]:
    """Read (note_path, context_blurb) for every chunk in the index at db_path.

    Plain read over the notes/chunks tables — does not touch the sqlite-vec
    virtual table, so the vec extension need not be loaded.
    """
    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "SELECT n.path, c.context_blurb FROM chunks c JOIN notes n ON c.note_id = n.id"
        )
        return list(cur.fetchall())
    finally:
        con.close()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_contextual_helpers.py -v && ruff check .`
Expected: PASS, ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/contextual.py tests/eval/test_contextual_helpers.py
git commit -m "feat(eval): read_blurb_rows to verify contextualization in a DB"
```

---

### Task 5: Context-dependent corpus + gold set + integrity tests

Build the English-only fixture vault. **Target (`buried`) notes** never name their own topic and describe a distinctive fact anaphorically (so OFF the sparse channel misses entirely and distractors compete). **Distractor notes** name their topic and share surface vocabulary with a target. The `direct` control queries hit distractor notes that contain the topic word (expected ≈ no lift).

**Files:**
- Create: `eval/contextual_corpus/piano.md`, `chess.md`, `bicycle.md`, `guitar.md`, `camera.md` (buried targets)
- Create: `eval/contextual_corpus/violin.md`, `drums.md`, `checkers.md`, `motorcycle.md`, `smartphone.md` (distractors)
- Create: `eval/contextual_gold.json`
- Test: `tests/eval/test_contextual_corpus.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_contextual_corpus.py`:

```python
from pathlib import Path

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.eval.harness import load_gold

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "contextual_corpus"
GOLD = REPO / "eval" / "contextual_gold.json"

# buried target note -> topic words that must NOT appear anywhere in the note
BURIED = {
    "piano.md": ["piano"],
    "chess.md": ["chess"],
    "bicycle.md": ["bicycle", "bike"],
    "guitar.md": ["guitar"],
    "camera.md": ["camera"],
}


def test_every_expected_note_exists_and_is_single():
    for case in load_gold(GOLD):
        assert len(case.expected) == 1, f"{case.query!r} has {len(case.expected)} expected"
        assert (CORPUS / case.expected[0]).exists(), f"missing: {case.expected[0]}"


def test_gold_covers_buried_and_direct_only():
    assert {c.scenario for c in load_gold(GOLD)} == {"buried", "direct"}


def test_buried_notes_never_name_their_topic():
    for name, words in BURIED.items():
        text = (CORPUS / name).read_text(encoding="utf-8").lower()
        for w in words:
            assert w not in text, f"{name} leaks its topic word {w!r}"


def test_buried_targets_chunk_into_at_least_two_chunks():
    parser, chunker = ObsidianMarkdownParser(), HeadingAwareChunker()
    for name in BURIED:
        raw = (CORPUS / name).read_text(encoding="utf-8")
        note, body = parser.parse(name, raw, 0.0)
        assert len(chunker.chunk(note, body)) >= 2, f"{name} did not chunk into >=2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_contextual_corpus.py -v`
Expected: FAIL (corpus dir / gold file do not exist yet).

- [ ] **Step 3: Create the five buried target notes**

`eval/contextual_corpus/piano.md`:

```markdown
# Morning Warm-up
Spend twenty minutes on scales and arpeggios before anything else.

# The Instrument
It has eighty-eight of them in black and white, each struck by a felt hammer the moment you press down.

# Upkeep
Call a technician twice a year so the strings stay in tune.
```

`eval/contextual_corpus/chess.md`:

```markdown
# The Opening
Control the centre early and bring out your knights and bishops before castling.

# The Board
Sixty-four light and dark squares, with sixteen carved figures arranged on each side at the start.

# The Finish
Once only a few figures remain, the lone monarch turns into an active piece and hunts the rival.
```

`eval/contextual_corpus/bicycle.md`:

```markdown
# Before Setting Off
Check that both tyres are firm and squeeze the brakes to be sure they bite.

# The Machine
Two wheels with a chain that drives the rear one, all hung on a light frame you straddle while pushing the pedals.

# Afterwards
Wipe the chain and add a drop of oil so it does not rust.
```

`eval/contextual_corpus/guitar.md`:

```markdown
# Warm-up
Loosen your fingers with a few simple chord changes.

# The Instrument
Six strings stretched over a fretted neck, plucked or strummed above a hollow wooden body.

# Care
Slacken the strings a little before storing it in a dry case.
```

`eval/contextual_corpus/camera.md`:

```markdown
# Framing
Decide what belongs in the frame and where the light falls before anything else.

# The Device
Light passes through a lens onto a sensor the instant you press the shutter, freezing a single moment.

# Afterwards
Copy the saved images onto a computer and back them up.
```

- [ ] **Step 4: Create the five distractor notes**

`eval/contextual_corpus/violin.md`:

```markdown
# The Violin
A violin has four strings over a smooth fingerboard and a curved wooden body.

# Playing
Draw the bow across the strings, or pluck them, while pressing notes with the left hand.
```

`eval/contextual_corpus/drums.md`:

```markdown
# Drums
A drum kit is a set of skins you strike with sticks, plus a few metal cymbals.

# Rhythm
Keep steady time with the bass pedal while the hands move around the kit.
```

`eval/contextual_corpus/checkers.md`:

```markdown
# Checkers
Checkers is played on a board of sixty-four squares using round pieces.

# Moves
Each piece slides diagonally, and you capture by jumping over a rival piece.
```

`eval/contextual_corpus/motorcycle.md`:

```markdown
# The Motorcycle
A motorcycle has two wheels and a frame, but an engine drives the rear wheel instead of a chain you pedal.

# Riding
Twist the throttle to speed up and squeeze the front brake gently to slow down.
```

`eval/contextual_corpus/smartphone.md`:

```markdown
# The Smartphone
A smartphone packs a tiny lens and sensor behind the glass for quick snapshots.

# Apps
Tap the screen to open apps, send messages, or browse the web.
```

- [ ] **Step 5: Create the gold set**

`eval/contextual_gold.json`:

```json
[
  {"query": "how to play the piano", "query_lang": "en", "expected": ["piano.md"], "scenario": "buried"},
  {"query": "the rules of the game of chess", "query_lang": "en", "expected": ["chess.md"], "scenario": "buried"},
  {"query": "how to ride a bicycle", "query_lang": "en", "expected": ["bicycle.md"], "scenario": "buried"},
  {"query": "learning to play the guitar", "query_lang": "en", "expected": ["guitar.md"], "scenario": "buried"},
  {"query": "taking photos with a camera", "query_lang": "en", "expected": ["camera.md"], "scenario": "buried"},

  {"query": "playing the violin", "query_lang": "en", "expected": ["violin.md"], "scenario": "direct"},
  {"query": "playing checkers on a board", "query_lang": "en", "expected": ["checkers.md"], "scenario": "direct"}
]
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_contextual_corpus.py -v`
Expected: PASS (all four tests). If `test_buried_notes_never_name_their_topic` fails, a note leaked its topic word — reword that section, do not change the test.

- [ ] **Step 7: Commit**

```bash
git add eval/contextual_corpus eval/contextual_gold.json tests/eval/test_contextual_corpus.py
git commit -m "test(eval): context-dependent corpus + gold for contextual lift"
```

---

### Task 6: The runner script `run_contextual_eval.py`

Thin glue: build OFF and ON containers, index both into temp DBs, run the loud all-blurbs guard on the ON index, evaluate all three channels over both, and print OFF / ON / delta tables. No unit test (real-LLM, non-deterministic `main` — run manually); the pure helpers it calls are already covered by Tasks 2–5.

**Files:**
- Create: `eval/run_contextual_eval.py`

- [ ] **Step 1: Write the script**

Create `eval/run_contextual_eval.py`:

```python
"""Measure the Contextual Retrieval (Phase 5) lift against a local Ollama.

Usage:  uv run python eval/run_contextual_eval.py [k]

Indexes the context-dependent corpus twice — contextualization OFF then ON
(against a real OpenAI-compatible chat endpoint) — and prints recall@k / MRR
per channel for each, plus an OFF->ON delta table.

Point it at a running endpoint with:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:11434/v1)
    ARIOSTEA_CTX_MODEL     (default llama3.1)
    ARIOSTEA_CTX_API_KEY   (default empty)

Requires the chat endpoint to be reachable: if any note fails to get a blurb
the run aborts, so a partial ON index can never masquerade as "no lift".
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, ContextualCfg, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import (
    make_dense_search_fn,
    make_hybrid_search_fn,
    make_sparse_search_fn,
)
from ariostea.eval.contextual import find_uncontextualized_notes, format_delta, read_blurb_rows
from ariostea.eval.harness import evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "contextual_corpus"
GOLD = EVAL_DIR / "contextual_gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_POOL = 50

BASE_URL = os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("ARIOSTEA_CTX_MODEL", "llama3.1")
API_KEY = os.environ.get("ARIOSTEA_CTX_API_KEY", "")


def _config(db: str, contextual: ContextualCfg) -> Config:
    return Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=db),
        contextual=contextual,
    )


def _build_index(db: str, contextual: ContextualCfg) -> None:
    container = build_container(_config(db, contextual))
    reindex_payload(container)


def _channels(db: str, embeddings: FastEmbedEmbeddings) -> dict:
    # A second store handle over the indexed DB keeps the production Container
    # ports-only while the eval reads the dense/sparse channels directly.
    store = SqliteStore(path=db, dim=embeddings.dimension)
    container = build_container(_config(db, ContextualCfg(enabled=False)))
    return {
        "DENSE": make_dense_search_fn(embeddings, store, CHUNK_POOL),
        "SPARSE": make_sparse_search_fn(store, CHUNK_POOL),
        "HYBRID": make_hybrid_search_fn(container, CHUNK_POOL),
    }


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)

    with tempfile.TemporaryDirectory() as tmp:
        off_db = str(Path(tmp) / "off.db")
        on_db = str(Path(tmp) / "on.db")

        print(f"Indexing OFF (contextualization disabled) ...")
        _build_index(off_db, ContextualCfg(enabled=False))

        print(f"Indexing ON  (contextualization via {MODEL} at {BASE_URL}) ...")
        _build_index(
            on_db,
            ContextualCfg(enabled=True, base_url=BASE_URL, model=MODEL, api_key=API_KEY),
        )

        missed = find_uncontextualized_notes(read_blurb_rows(on_db))
        if missed:
            total = len({path for path, _ in read_blurb_rows(on_db)})
            raise SystemExit(
                f"contextualization incomplete — {len(missed)}/{total} notes produced "
                f"no blurb (is the chat endpoint running at {BASE_URL}?): {', '.join(missed)}"
            )

        off_channels = _channels(off_db, embeddings)
        on_channels = _channels(on_db, embeddings)

        for label in ("DENSE", "SPARSE", "HYBRID"):
            off_report = evaluate(cases, off_channels[label], k=k)
            on_report = evaluate(cases, on_channels[label], k=k)
            print(f"\n=== {label} — OFF ===")
            print(format_report(off_report))
            print(f"\n=== {label} — ON ===")
            print(format_report(on_report))
            print(f"\n=== {label} — Δ (OFF → ON) ===")
            print(format_delta(off_report, on_report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it parses and lints**

Run: `uv run python -c "import ast; ast.parse(open('eval/run_contextual_eval.py').read())"`
Expected: no output (parses cleanly).

Run: `ruff check eval/run_contextual_eval.py`
Expected: clean. (If ruff flags the f-strings without placeholders on the "Indexing OFF" line, drop the `f` prefix there.)

- [ ] **Step 3: Verify the full fast suite is still green**

Run: `uv run pytest -m "not integration" -q && ruff check .`
Expected: all pass, ruff clean.

- [ ] **Step 4: Commit**

```bash
git add eval/run_contextual_eval.py
git commit -m "feat(eval): run_contextual_eval.py — measure Phase 5 lift OFF vs ON"
```

---

### Task 7: Run the measurement and record the result

This is the payoff: run against a live Ollama and write the numbers back into the design as evidence. **Requires the user's local Ollama** — if it is not running, surface that and stop (do not fake numbers).

**Files:**
- Modify: `docs/superpowers/specs/2026-06-28-ariostea-contextual-retrieval-design.md` (add a "Measured lift" subsection under §9 or §10)

- [ ] **Step 1: Confirm an endpoint is available**

Ask the user to confirm Ollama is running and which model to use (default `llama3.1`). If unreachable, stop here and report — the rest of this task cannot proceed.

- [ ] **Step 2: Run the eval**

Run: `uv run python eval/run_contextual_eval.py 5`
Expected: OFF / ON / Δ tables for DENSE, SPARSE, HYBRID. First run downloads the embedding model (~minute). The guard aborts with a clear message if any note lacks a blurb.

- [ ] **Step 3: Record the measured lift in the design doc**

Add a short subsection to `docs/superpowers/specs/2026-06-28-ariostea-contextual-retrieval-design.md` capturing the date, the model used, and the per-channel Δ table (paste the `Δ (OFF → ON)` output for each channel), plus a one-line interpretation (e.g. "lift concentrated on sparse `buried`, flat on `direct`, as predicted").

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-28-ariostea-contextual-retrieval-design.md
git commit -m "docs: record measured contextual-retrieval lift (Phase 5)"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §3 (files: corpus, gold, runner, channels extraction) → Tasks 1, 5, 6. New `contextual.py` module flagged as a spec refinement (top of plan). ✓
- Spec §4 (corpus: buried/direct scenarios, distractors, anaphora, never-name-topic) → Task 5 + its integrity tests. ✓
- Spec §5 (runner: env config, index twice, loud all-blurbs guard, evaluate three channels, delta output) → Task 6 (+ guard/reader from Tasks 3–4). ✓
- Spec §5.3 / user refinement (require **all** blurbs, error names missed count) → Task 3 (`find_uncontextualized_notes` flags any null) + Task 6 (`SystemExit` with `len(missed)/total`). ✓
- Spec §6 (TDD `format_delta`, the all-blurbs guard, corpus/gold integrity incl. ≥2 chunks) → Tasks 2, 3, 5. ✓
- Spec §8 (acceptance: runs end-to-end; guard aborts when no endpoint; helpers pass fast suite; record the number) → Tasks 6, 7. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**3. Type consistency:** `make_hybrid_search_fn(container, pool)` defined in Task 1, used identically in Task 6. `format_delta(off, on)`, `find_uncontextualized_notes(rows)`, `read_blurb_rows(db_path)` signatures defined in Tasks 2–4 match their Task 6 call sites. `ContextualCfg`/`Config`/`VaultCfg`/`EmbeddingCfg`/`StoreCfg` fields match `src/ariostea/config/schema.py`. Store SQL (`notes.id`, `notes.path`, `chunks.note_id`, `chunks.context_blurb`) matches the verified schema. ✓
