# Cross-Lingual Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measurement harness that scores cross-lingual retrieval quality (recall@k / MRR, broken down by language direction) against a committed bilingual fixture vault, so every later retrieval change is proven rather than assumed.

**Architecture:** The harness is a *driving consumer* of the existing use cases — the same architectural role as `cli.py` and the MCP server. Pure metric functions and the scoring engine live in a new `ariostea.eval` package (unit-testable in the fast suite); the scoring engine takes a `search_fn` callback so it can be tested with a fake ranker (no model downloads). A committed `eval/` directory holds the bilingual fixture corpus, the JSON gold set, and a thin runner that wires the real container.

**Tech Stack:** Python 3.12, pytest (`integration` marker for model-loading tests), stdlib `json` for the gold set (no new dependency), the existing `build_container` / `search_payload` API.

**Source spec:** [`docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md`](../specs/2026-06-27-ariostea-multilingual-retrieval-design.md) — this plan implements **Component 1** only.

---

## File Structure

- Create: `src/ariostea/eval/__init__.py` — package marker.
- Create: `src/ariostea/eval/metrics.py` — pure metric functions (`recall_at_k`, `reciprocal_rank`). No I/O, no models.
- Create: `src/ariostea/eval/harness.py` — `GoldCase` model, `load_gold`, `dedupe`, `evaluate`, `format_report`, and the report dataclasses. Depends only on `metrics` + stdlib.
- Create: `eval/corpus/*.md` — committed bilingual fixture vault (4 notes: 2 topics × EN/IT).
- Create: `eval/gold.json` — gold set of query→expected-note cases tagged by direction.
- Create: `eval/run_eval.py` — thin runner: wires the real container against the fixture, prints the report.
- Create: `tests/eval/test_metrics.py` — fast unit tests for metric math.
- Create: `tests/eval/test_harness.py` — fast unit tests for loader + `evaluate` (fake `search_fn`) + `format_report` + `dedupe`.
- Create: `tests/eval/test_harness_integration.py` — `@pytest.mark.integration` end-to-end run against the fixture with the multilingual model.
- Modify: `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md` — roadmap (§17) + §19 deltas from the multilingual design.

---

### Task 1: Metric functions

**Files:**
- Create: `src/ariostea/eval/__init__.py`
- Create: `src/ariostea/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_metrics.py`:

```python
from ariostea.eval.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_hit_within_k():
    assert recall_at_k({"a.md"}, ["x.md", "a.md", "y.md"], k=3) == 1.0


def test_recall_at_k_miss_outside_k():
    # a.md is at index 2 (rank 3); with k=2 it is outside the window.
    assert recall_at_k({"a.md"}, ["x.md", "y.md", "a.md"], k=2) == 0.0


def test_recall_at_k_any_expected_counts():
    assert recall_at_k({"a.md", "b.md"}, ["b.md", "z.md"], k=1) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank({"a.md"}, ["a.md", "b.md"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank({"a.md"}, ["b.md", "a.md"]) == 0.5


def test_reciprocal_rank_absent_is_zero():
    assert reciprocal_rank({"a.md"}, ["b.md", "c.md"]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ariostea/eval/__init__.py` (empty file):

```python
```

Create `src/ariostea/eval/metrics.py`:

```python
from __future__ import annotations


def recall_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """1.0 if any expected note appears in the top-k ranked notes, else 0.0.

    `ranked` is a list of note paths in rank order (best first), already
    deduplicated to one entry per note.
    """
    top = ranked[:k]
    return 1.0 if any(path in top for path in expected) else 0.0


def reciprocal_rank(expected: set[str], ranked: list[str]) -> float:
    """Reciprocal of the 1-based rank of the first expected note; 0.0 if none."""
    for index, path in enumerate(ranked):
        if path in expected:
            return 1.0 / (index + 1)
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_metrics.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/eval/__init__.py src/ariostea/eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat(eval): recall@k and reciprocal-rank metric functions"
```

---

### Task 2: Gold-set model and loader

**Files:**
- Create: `src/ariostea/eval/harness.py`
- Test: `tests/eval/test_harness.py`

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_harness.py`:

```python
from ariostea.eval.harness import GoldCase, load_gold


def test_load_gold_parses_cases(tmp_path):
    gold = tmp_path / "gold.json"
    # → is the "→" arrow; written escaped to keep the test ASCII-safe.
    gold.write_text(
        '[{"query": "dice game", "query_lang": "en", '
        '"expected": ["dadi_it.md"], "direction": "en\\u2192it"}]',
        encoding="utf-8",
    )

    cases = load_gold(gold)

    assert cases == [
        GoldCase(
            query="dice game",
            query_lang="en",
            expected=("dadi_it.md",),
            direction="en→it",
        )
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: FAIL — `ImportError: cannot import name 'GoldCase' from 'ariostea.eval.harness'`

- [ ] **Step 3: Write minimal implementation**

Create `src/ariostea/eval/harness.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldCase:
    query: str
    query_lang: str
    expected: tuple[str, ...]
    direction: str  # "en→it" | "it→en" | "same"


def load_gold(path: str | Path) -> list[GoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            expected=tuple(row["expected"]),
            direction=row["direction"],
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/eval/harness.py tests/eval/test_harness.py
git commit -m "feat(eval): GoldCase model and JSON gold-set loader"
```

---

### Task 3: Scoring engine, dedupe, and report formatter

**Files:**
- Modify: `src/ariostea/eval/harness.py` (append)
- Test: `tests/eval/test_harness.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_harness.py`:

```python
import pytest

from ariostea.eval.harness import dedupe, evaluate, format_report


def test_dedupe_keeps_first_occurrence_in_order():
    assert dedupe(["a.md", "b.md", "a.md", "c.md", "b.md"]) == ["a.md", "b.md", "c.md"]


def test_evaluate_aggregates_overall_and_by_direction():
    cases = [
        GoldCase("q1", "en", ("it1.md",), "en→it"),
        GoldCase("q2", "it", ("en1.md",), "it→en"),
        GoldCase("q3", "en", ("en2.md",), "same"),
    ]
    # Fake ranker: q1 hits at rank 1, q2 misses entirely, q3 hits at rank 2.
    table = {
        "q1": ["it1.md", "x.md"],
        "q2": ["y.md", "z.md"],
        "q3": ["w.md", "en2.md"],
    }

    report = evaluate(cases, lambda query, k: table[query][:k], k=5)

    assert report.k == 5
    assert report.overall.n == 3
    assert report.overall.recall_at_k == pytest.approx(2 / 3)
    assert report.overall.mrr == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    by = {d.direction: d for d in report.by_direction}
    assert by["en→it"].recall_at_k == 1.0 and by["en→it"].mrr == 1.0
    assert by["it→en"].recall_at_k == 0.0
    assert by["same"].mrr == 0.5


def test_format_report_contains_directions_and_overall():
    cases = [GoldCase("q1", "en", ("a.md",), "same")]
    report = evaluate(cases, lambda query, k: ["a.md"], k=3)

    text = format_report(report)

    assert "recall@3" in text
    assert "same" in text
    assert "overall" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: FAIL — `ImportError: cannot import name 'dedupe' from 'ariostea.eval.harness'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/ariostea/eval/harness.py`:

```python
from collections.abc import Callable

from ariostea.eval.metrics import recall_at_k, reciprocal_rank

# A ranker: given (query, k), return up to k note paths in rank order (best first).
SearchFn = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class DirectionScore:
    direction: str
    n: int
    recall_at_k: float
    mrr: float


@dataclass(frozen=True)
class EvalReport:
    k: int
    overall: DirectionScore
    by_direction: tuple[DirectionScore, ...]


def dedupe(paths: list[str]) -> list[str]:
    """Collapse a chunk-level path list to one entry per note, preserving order."""
    seen: list[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _aggregate(direction: str, rows: list[tuple[float, float]]) -> DirectionScore:
    n = len(rows)
    if n == 0:
        return DirectionScore(direction=direction, n=0, recall_at_k=0.0, mrr=0.0)
    return DirectionScore(
        direction=direction,
        n=n,
        recall_at_k=sum(recall for recall, _ in rows) / n,
        mrr=sum(rr for _, rr in rows) / n,
    )


def evaluate(cases: list[GoldCase], search_fn: SearchFn, k: int) -> EvalReport:
    """Run every gold case through search_fn once and aggregate recall@k / MRR
    overall and per direction. search_fn must return deduped note paths."""
    scored: list[tuple[str, float, float]] = []
    for case in cases:
        ranked = search_fn(case.query, k)
        expected = set(case.expected)
        scored.append(
            (case.direction, recall_at_k(expected, ranked, k), reciprocal_rank(expected, ranked))
        )

    overall = _aggregate("overall", [(r, rr) for _, r, rr in scored])
    directions = sorted({direction for direction, _, _ in scored})
    by_direction = tuple(
        _aggregate(d, [(r, rr) for direction, r, rr in scored if direction == d])
        for d in directions
    )
    return EvalReport(k=k, overall=overall, by_direction=by_direction)


def format_report(report: EvalReport) -> str:
    header = f"{'direction':<10} {'n':>3}  recall@{report.k:<3}  mrr"
    lines = [header]
    for d in (*report.by_direction, report.overall):
        lines.append(f"{d.direction:<10} {d.n:>3}  {d.recall_at_k:>8.3f}  {d.mrr:.3f}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: PASS (4 passed — loader test from Task 2 plus the 3 new ones)

- [ ] **Step 5: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add src/ariostea/eval/harness.py tests/eval/test_harness.py
git commit -m "feat(eval): scoring engine, dedupe, and report formatter"
```

---

### Task 4: Fixture corpus, gold set, runner, and integration test

**Files:**
- Create: `eval/corpus/dice_en.md`, `eval/corpus/dadi_it.md`, `eval/corpus/pasta_en.md`, `eval/corpus/pasta_it.md`
- Create: `eval/gold.json`
- Create: `eval/run_eval.py`
- Test: `tests/eval/test_harness_integration.py`

- [ ] **Step 1: Create the fixture corpus**

Create `eval/corpus/dice_en.md`:

```markdown
# Dice and board games

We rolled the dice and moved our tokens around the board to win the game.
```

Create `eval/corpus/dadi_it.md`:

```markdown
# Dadi e giochi da tavolo

Abbiamo tirato i dadi e mosso le pedine sul tabellone per vincere la partita.
```

Create `eval/corpus/pasta_en.md`:

```markdown
# Cooking pasta

Bring the water to a boil, add salt, then cook the spaghetti until al dente.
```

Create `eval/corpus/pasta_it.md`:

```markdown
# Cucinare la pasta

Porta l'acqua a ebollizione, aggiungi il sale, poi cuoci gli spaghetti al dente.
```

- [ ] **Step 2: Create the gold set**

Create `eval/gold.json`:

```json
[
  {"query": "how to cook spaghetti at home", "query_lang": "en", "expected": ["pasta_en.md"], "direction": "same"},
  {"query": "come cuocere gli spaghetti in casa", "query_lang": "it", "expected": ["pasta_it.md"], "direction": "same"},
  {"query": "rolling dice in a board game", "query_lang": "en", "expected": ["dice_en.md"], "direction": "same"},
  {"query": "tirare i dadi in un gioco da tavolo", "query_lang": "it", "expected": ["dadi_it.md"], "direction": "same"},
  {"query": "rolling dice board game pieces", "query_lang": "en", "expected": ["dadi_it.md"], "direction": "en→it"},
  {"query": "cooking spaghetti al dente", "query_lang": "en", "expected": ["pasta_it.md"], "direction": "en→it"},
  {"query": "tirare i dadi e muovere le pedine", "query_lang": "it", "expected": ["dice_en.md"], "direction": "it→en"},
  {"query": "bollire la pasta con il sale", "query_lang": "it", "expected": ["pasta_en.md"], "direction": "it→en"}
]
```

- [ ] **Step 3: Write the runner**

Create `eval/run_eval.py`:

```python
"""Run the cross-lingual eval harness against the committed fixture vault.

Usage:  uv run python eval/run_eval.py [k]

Loads the multilingual embedding model (downloads on first run), indexes the
fixture corpus into a throwaway database, and prints a recall@k / MRR table
broken down by language direction.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.harness import dedupe, evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload, search_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
GOLD = EVAL_DIR / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Pull a generous chunk pool, then dedupe to notes before taking the top k.
CHUNK_POOL = 50


def make_search_fn(container):
    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=CHUNK_POOL)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    return search_fn


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            vault=VaultCfg(path=str(CORPUS), ignore=[]),
            embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
            store=StoreCfg(backend="sqlite", path=str(Path(tmp) / "eval.db")),
        )
        container = build_container(cfg)
        reindex_payload(container)
        report = evaluate(cases, make_search_fn(container), k=k)
    print(format_report(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write the failing integration test**

Create `tests/eval/test_harness_integration.py`:

```python
from pathlib import Path

import pytest

from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.harness import dedupe, evaluate, load_gold
from ariostea.mcp.handlers import reindex_payload, search_payload

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
GOLD = REPO / "eval" / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@pytest.mark.integration
def test_eval_harness_runs_and_same_language_is_perfect(tmp_path):
    cfg = Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "eval.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    cases = load_gold(GOLD)

    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=50)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    report = evaluate(cases, search_fn, k=3)

    # The harness produces a complete report over every gold case...
    assert report.overall.n == len(cases)
    by = {d.direction: d for d in report.by_direction}
    assert set(by) == {"same", "en→it", "it→en"}

    # ...and same-language retrieval on the fixture is a stable 1.0 floor.
    # Cross-lingual directions are reported but NOT asserted — lifting those
    # numbers is exactly what the later multilingual-reranking work is for.
    assert by["same"].recall_at_k == 1.0
```

- [ ] **Step 5: Run the integration test to verify it passes**

Run: `uv run pytest tests/eval/test_harness_integration.py -v -m integration`
Expected: PASS (1 passed). First run downloads the multilingual model (~1GB); subsequent runs are fast.

- [ ] **Step 6: Run the runner manually to confirm the report prints**

Run: `uv run python eval/run_eval.py`
Expected: a table like (exact cross-lingual numbers will vary):

```
direction    n  recall@5    mrr
en→it        2     X.XXX  X.XXX
it→en        2     X.XXX  X.XXX
same         4     1.000  X.XXX
overall      8     X.XXX  X.XXX
```

- [ ] **Step 7: Confirm the fast suite still excludes the integration test**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS, and the integration test is deselected (the eval unit tests from Tasks 1–3 run; the model is NOT downloaded).

- [ ] **Step 8: Commit**

```bash
uv run ruff check --fix . && uv run ruff format . && uv run ruff check .
git add eval/ tests/eval/test_harness_integration.py
git commit -m "feat(eval): bilingual fixture vault, gold set, runner, integration test"
```

---

### Task 5: Wire the eval harness into the roadmap

This is the "add solutions to the roadmap" deliverable: fold the multilingual design's deltas into the PRD.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`

- [ ] **Step 1: Insert the eval-harness roadmap row and amend Phase 6**

In `docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`, find the roadmap table row for Phase 6:

```markdown
| 6 | Reranking stage | Rerank reorders top-N measurably |
```

Replace it with these two rows (a new Eval row precedes it, and Phase 6 is amended):

```markdown
| **Eval** | **Cross-lingual eval harness** — committed bilingual fixture vault + JSON gold set + recall@k / MRR scorer (per-direction). Driving consumer of the use cases; no new ports. | Baseline reported; scorer math unit-tested; satisfies the "eval set" acceptance referenced by Phases 5–6. See [`2026-06-27-ariostea-multilingual-retrieval-design.md`](2026-06-27-ariostea-multilingual-retrieval-design.md). |
| 6 | Reranking stage — **default reranker is multilingual** (`bge-reranker-v2-m3`, ONNX); RRF demoted to a recall gatherer feeding the reranker | Rerank reorders top-N measurably; **recall@k / MRR gain on `en→it` and `it→en`, no regression on `same`** |
```

- [ ] **Step 2: Add the BGE-M3 learned-sparse backlog item to Phase 8**

Find the Phase 8 roadmap row:

```markdown
| 8 | Packaging polish (`init` wizard, docs) + extra store/rerank adapters + configurable FTS tokenizer | One-command onboarding documented; alt adapters pass contract tests |
```

Replace it with:

```markdown
| 8 | Packaging polish (`init` wizard, docs) + extra store/rerank adapters + configurable FTS tokenizer + **(conditional) BGE-M3 learned multilingual sparse** | One-command onboarding documented; alt adapters pass contract tests; BGE-M3 pursued only if post-rerank eval still shows a cross-lingual gap |
```

- [ ] **Step 3: Point the §19 cross-lingual findings at the design doc**

Find the end of the §19 paragraph that begins "**RRF suppresses strong single-channel matches**" — it currently ends with:

```markdown
... and per-note diversity/MMR (a single long note can otherwise flood both channels and dominate the fused list).
```

Append (in the same paragraph, after that sentence):

```markdown
 The committed resolution — eval harness → multilingual reranking → optional `WeightedFuser` interim → conditional BGE-M3 learned sparse — is specified in [`2026-06-27-ariostea-multilingual-retrieval-design.md`](2026-06-27-ariostea-multilingual-retrieval-design.md).
```

- [ ] **Step 4: Verify the spec still reads cleanly**

Run: `grep -n "Cross-lingual eval harness\|bge-reranker-v2-m3\|2026-06-27-ariostea-multilingual" docs/superpowers/specs/2026-06-12-ariostea-rag-design.md`
Expected: matches for the new Eval row, the amended Phase 6, the Phase 8 backlog, and the §19 pointer.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-06-12-ariostea-rag-design.md
git commit -m "docs: add eval harness + multilingual reranking to roadmap"
```

---

## Notes for the implementer

- **Why `evaluate` takes a `search_fn`:** it decouples scoring from the container so Tasks 1–3 are fast unit tests with a fake ranker (no model download). Only Task 4's integration test wires the real pipeline.
- **Why dedupe + small k:** retrieval returns *chunks*; one note has many. We score at the *note* level, so collapse chunks to notes first. The fixture has only 4 notes, so a large k would make recall trivially 1.0 — keep k small (the runner defaults to 5, the integration test asserts at 3).
- **Why pin the multilingual model:** the schema default `local_model` is English-only `BAAI/bge-small-en-v1.5`; cross-lingual retrieval requires the multilingual model, so the runner and integration test set it explicitly.
- **Gold set format is JSON, not YAML** (the design said YAML): JSON is stdlib (`json`/`tomllib` only — no new dependency), which matches the project's zero-extra-dep posture. Same declarative content.
```
