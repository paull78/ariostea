# Gold-Set Expansion + Per-Channel Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the eval corpus/gold set to en/it/es with conclusive cross-lingual, accent, and inflection scenarios, and let the harness measure the dense, sparse, and hybrid channels separately.

**Architecture:** Single-language topics make cross-lingual queries conclusive (one relevant note, no twin). The harness's `evaluate(cases, search_fn, k)` is unchanged — per-channel measurement is achieved by passing three different `search_fn`s built from raw `dense()`/`sparse()` plus the existing hybrid pipeline. The gold-case grouping field is renamed `direction` → `scenario` since it now also labels accent/inflection.

**Tech Stack:** Python, pytest (with an `integration` marker for model-backed tests), fastembed (multilingual model), sqlite-vec + FTS5.

**Design:** `docs/superpowers/specs/2026-06-28-ariostea-gold-set-expansion-design.md`

---

## File Structure

- `src/ariostea/eval/harness.py` — **modify**: rename `direction`→`scenario` (`GoldCase`, `ScenarioScore`, `EvalReport.by_scenario`, `load_gold`, `evaluate`, `format_report`).
- `src/ariostea/eval/channels.py` — **create**: `make_dense_search_fn` / `make_sparse_search_fn` factories (the two new, unit-testable channel rankers).
- `eval/corpus/{astronomia_it,cucito_it,beekeeping_en,ciclismo_es,alfareria_es}.md` — **create**: 5 single-language fixture notes.
- `eval/gold.json` — **modify**: rename key (Task 1), then replace with the ~17-case set (Task 4).
- `eval/run_eval.py` — **modify**: run all three channels, print three labeled reports.
- `tests/eval/test_harness.py` — **modify**: rename references.
- `tests/eval/test_harness_integration.py` — **modify**: rename references; update scenario set.
- `tests/eval/test_channels.py` — **create**: unit tests for the factories.
- `tests/eval/test_corpus_fixtures.py` — **create**: guard tests on note content invariants.
- `tests/eval/test_gold_set.py` — **create**: guard tests on the gold set.
- `tests/eval/test_channels_integration.py` — **create**: per-channel hypotheses (accent→sparse hit; inflection→sparse miss, dense hit).

---

### Task 1: Rename `direction` → `scenario` in the harness

Pure rename. Update the tests to the new names first (RED), then rename the source (GREEN).

**Files:**
- Modify: `tests/eval/test_harness.py`
- Modify: `src/ariostea/eval/harness.py`
- Modify: `eval/gold.json`
- Modify: `tests/eval/test_harness_integration.py`

- [ ] **Step 1: Update `tests/eval/test_harness.py` to the new names**

Replace the whole file with:

```python
import pytest

from ariostea.eval.harness import GoldCase, dedupe, evaluate, format_report, load_gold


def test_load_gold_parses_cases(tmp_path):
    gold = tmp_path / "gold.json"
    # → is the "→" arrow; written escaped to keep the test ASCII-safe.
    gold.write_text(
        '[{"query": "dice game", "query_lang": "en", '
        '"expected": ["dadi_it.md"], "scenario": "en\\u2192it"}]',
        encoding="utf-8",
    )

    cases = load_gold(gold)

    assert cases == [
        GoldCase(
            query="dice game",
            query_lang="en",
            expected=("dadi_it.md",),
            scenario="en→it",
        )
    ]


def test_dedupe_keeps_first_occurrence_in_order():
    assert dedupe(["a.md", "b.md", "a.md", "c.md", "b.md"]) == ["a.md", "b.md", "c.md"]


def test_evaluate_aggregates_overall_and_by_scenario():
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

    by = {s.scenario: s for s in report.by_scenario}
    assert by["en→it"].recall_at_k == 1.0 and by["en→it"].mrr == 1.0
    assert by["it→en"].recall_at_k == 0.0
    assert by["same"].mrr == 0.5


def test_format_report_contains_scenarios_and_overall():
    cases = [GoldCase("q1", "en", ("a.md",), "same")]
    report = evaluate(cases, lambda query, k: ["a.md"], k=3)

    text = format_report(report)

    assert "recall@3" in text
    assert "same" in text
    assert "overall" in text
```

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `uv run pytest tests/eval/test_harness.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'scenario'` (and `AttributeError` on `by_scenario`).

- [ ] **Step 3: Rename in `src/ariostea/eval/harness.py`**

Replace the whole file with:

```python
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ariostea.eval.metrics import recall_at_k, reciprocal_rank

# A ranker: given (query, k), return up to k note paths in rank order (best first).
SearchFn = Callable[[str, int], list[str]]


@dataclass(frozen=True)
class GoldCase:
    query: str
    query_lang: str
    expected: tuple[str, ...]
    scenario: str  # "same" | "en→it" | … | "accent" | "inflection"


@dataclass(frozen=True)
class ScenarioScore:
    scenario: str
    n: int
    recall_at_k: float
    mrr: float


@dataclass(frozen=True)
class EvalReport:
    k: int
    overall: ScenarioScore
    by_scenario: tuple[ScenarioScore, ...]


def load_gold(path: str | Path) -> list[GoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            expected=tuple(row["expected"]),
            scenario=row["scenario"],
        )
        for row in rows
    ]


def dedupe(paths: list[str]) -> list[str]:
    """Collapse a chunk-level path list to one entry per note, preserving order."""
    seen: list[str] = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen


def _aggregate(scenario: str, rows: list[tuple[float, float]]) -> ScenarioScore:
    n = len(rows)
    if n == 0:
        return ScenarioScore(scenario=scenario, n=0, recall_at_k=0.0, mrr=0.0)
    return ScenarioScore(
        scenario=scenario,
        n=n,
        recall_at_k=sum(recall for recall, _ in rows) / n,
        mrr=sum(rr for _, rr in rows) / n,
    )


def evaluate(cases: list[GoldCase], search_fn: SearchFn, k: int) -> EvalReport:
    """Run every gold case through search_fn once and aggregate recall@k / MRR
    overall and per scenario. search_fn must return deduped note paths."""
    scored: list[tuple[str, float, float]] = []
    for case in cases:
        ranked = search_fn(case.query, k)
        expected = set(case.expected)
        scored.append(
            (case.scenario, recall_at_k(expected, ranked, k), reciprocal_rank(expected, ranked))
        )

    overall = _aggregate("overall", [(r, rr) for _, r, rr in scored])
    scenarios = sorted({scenario for scenario, _, _ in scored})
    by_scenario = tuple(
        _aggregate(s, [(r, rr) for scenario, r, rr in scored if scenario == s])
        for s in scenarios
    )
    return EvalReport(k=k, overall=overall, by_scenario=by_scenario)


def format_report(report: EvalReport) -> str:
    header = f"{'scenario':<12} {'n':>3}  recall@{report.k:<3}  mrr"
    lines = [header]
    for s in (*report.by_scenario, report.overall):
        lines.append(f"{s.scenario:<12} {s.n:>3}  {s.recall_at_k:>8.3f}  {s.mrr:.3f}")
    return "\n".join(lines)
```

- [ ] **Step 4: Rename the key in `eval/gold.json`**

In the existing `eval/gold.json`, rename every `"direction"` key to `"scenario"` (leave the 8 cases and their values unchanged for now). The file becomes:

```json
[
  {"query": "how to cook spaghetti at home", "query_lang": "en", "expected": ["pasta_en.md"], "scenario": "same"},
  {"query": "come cuocere gli spaghetti in casa", "query_lang": "it", "expected": ["pasta_it.md"], "scenario": "same"},
  {"query": "rolling dice in a board game", "query_lang": "en", "expected": ["dice_en.md"], "scenario": "same"},
  {"query": "tirare i dadi in un gioco da tavolo", "query_lang": "it", "expected": ["dadi_it.md"], "scenario": "same"},
  {"query": "rolling dice board game pieces", "query_lang": "en", "expected": ["dadi_it.md"], "scenario": "en→it"},
  {"query": "cooking spaghetti al dente", "query_lang": "en", "expected": ["pasta_it.md"], "scenario": "en→it"},
  {"query": "tirare i dadi e muovere le pedine", "query_lang": "it", "expected": ["dice_en.md"], "scenario": "it→en"},
  {"query": "bollire la pasta con il sale", "query_lang": "it", "expected": ["pasta_en.md"], "scenario": "it→en"}
]
```

- [ ] **Step 5: Rename references in `tests/eval/test_harness_integration.py`**

Change the two `by_direction` references to `by_scenario`. The assertion block becomes:

```python
    report = evaluate(cases, search_fn, k=3)

    # The harness produces a complete report over every gold case...
    assert report.overall.n == len(cases)
    by = {s.scenario: s for s in report.by_scenario}
    assert set(by) == {"same", "en→it", "it→en"}

    # ...and same-language retrieval on the fixture is a stable 1.0 floor.
    assert by["same"].recall_at_k == 1.0
```

- [ ] **Step 6: Run the fast suite to verify it passes**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS (all green, no warnings). The integration test is deselected; it will be exercised in later tasks.

- [ ] **Step 7: Commit**

```bash
git add src/ariostea/eval/harness.py tests/eval/test_harness.py tests/eval/test_harness_integration.py eval/gold.json
git commit -m "refactor(eval): rename gold-case grouping field direction -> scenario"
```

---

### Task 2: Per-channel search-fn factories

The two new channel rankers. `evaluate` already takes a `search_fn`, so these are the only new code needed for isolation.

**Files:**
- Create: `tests/eval/test_channels.py`
- Create: `src/ariostea/eval/channels.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/eval/test_channels.py`:

```python
from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.eval.channels import make_dense_search_fn, make_sparse_search_fn


def _rc(note_path, ordinal):
    chunk = Chunk(
        note_path=note_path, ordinal=ordinal, heading_path=("A",), text="t", token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=1.0, dense_rank=ordinal, sparse_rank=None)


class FakeEmbeddings:
    def embed_query(self, text):
        self.last_query = text
        return [0.1, 0.2, 0.3]


class FakeRetriever:
    def __init__(self):
        self.dense_call = None
        self.sparse_call = None

    def dense(self, vec, k, filters=None):
        self.dense_call = (vec, k, filters)
        return [_rc("a.md", 0), _rc("a.md", 1), _rc("b.md", 2)]  # two chunks of a.md

    def sparse(self, query, k, filters=None):
        self.sparse_call = (query, k, filters)
        return [_rc("c.md", 0), _rc("c.md", 1)]


def test_dense_search_fn_embeds_query_and_dedupes_to_notes():
    emb, ret = FakeEmbeddings(), FakeRetriever()
    fn = make_dense_search_fn(emb, ret, pool=50)

    assert fn("hello", 5) == ["a.md", "b.md"]
    assert emb.last_query == "hello"  # query was embedded
    assert ret.dense_call == ([0.1, 0.2, 0.3], 50, None)  # vec + pool passed through


def test_dense_search_fn_truncates_to_k():
    fn = make_dense_search_fn(FakeEmbeddings(), FakeRetriever(), pool=50)
    assert fn("hello", 1) == ["a.md"]


def test_sparse_search_fn_passes_raw_query_and_dedupes():
    ret = FakeRetriever()
    fn = make_sparse_search_fn(ret, pool=30)

    assert fn("dice", 5) == ["c.md"]
    assert ret.sparse_call == ("dice", 30, None)  # raw text + pool, no embedding
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/eval/test_channels.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.channels'`.

- [ ] **Step 3: Implement `src/ariostea/eval/channels.py`**

```python
"""Per-channel search functions for the eval harness.

Each factory returns a SearchFn (query, k -> note paths) that exercises a
single retrieval channel in isolation, so the harness can attribute results
to the dense or sparse side rather than only the blended pipeline.
"""

from __future__ import annotations

from ariostea.eval.harness import SearchFn, dedupe
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.store import ChunkRetriever


def make_dense_search_fn(
    embeddings: EmbeddingProvider, retriever: ChunkRetriever, pool: int
) -> SearchFn:
    def search_fn(query: str, k: int) -> list[str]:
        vec = embeddings.embed_query(query)
        hits = retriever.dense(vec=vec, k=pool, filters=None)
        return dedupe([h.chunk.note_path for h in hits])[:k]

    return search_fn


def make_sparse_search_fn(retriever: ChunkRetriever, pool: int) -> SearchFn:
    def search_fn(query: str, k: int) -> list[str]:
        hits = retriever.sparse(query=query, k=pool, filters=None)
        return dedupe([h.chunk.note_path for h in hits])[:k]

    return search_fn
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/eval/test_channels.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/channels.py tests/eval/test_channels.py
git commit -m "feat(eval): per-channel dense/sparse search-fn factories"
```

---

### Task 3: Single-language corpus notes

Five fixture notes. A guard test enforces the two content invariants from the design (accent term present; inflection notes hold only the base form).

**Files:**
- Create: `tests/eval/test_corpus_fixtures.py`
- Create: `eval/corpus/astronomia_it.md`
- Create: `eval/corpus/cucito_it.md`
- Create: `eval/corpus/beekeeping_en.md`
- Create: `eval/corpus/ciclismo_es.md`
- Create: `eval/corpus/alfareria_es.md`

- [ ] **Step 1: Write the failing guard test**

Create `tests/eval/test_corpus_fixtures.py`:

```python
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "eval" / "corpus"


def _read(name):
    return (CORPUS / name).read_text(encoding="utf-8")


def test_accent_targets_contain_their_keyword():
    assert "città" in _read("astronomia_it.md")
    assert "montaña" in _read("ciclismo_es.md")


def test_inflection_notes_contain_only_the_base_form():
    cucito = _read("cucito_it.md")
    assert "bottone" in cucito and "bottoni" not in cucito
    alfareria = _read("alfareria_es.md")
    assert "vasija" in alfareria and "vasijas" not in alfareria
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_corpus_fixtures.py -v`
Expected: FAIL — `FileNotFoundError` (the notes do not exist yet).

- [ ] **Step 3: Create the five notes**

`eval/corpus/astronomia_it.md`:

```markdown
# Osservare le stelle

Con il telescopio guardiamo i pianeti di notte, lontano dalle luci della città.
```

`eval/corpus/cucito_it.md`:

```markdown
# Cucito a mano

Con ago e filo attacco un bottone alla camicia di lana.
```

`eval/corpus/beekeeping_en.md`:

```markdown
# Keeping bees

The beekeeper checks the hive and collects honey from the colony.
```

`eval/corpus/ciclismo_es.md`:

```markdown
# Ciclismo de montaña

El ciclista sube la montaña en bicicleta por un sendero empinado.
```

`eval/corpus/alfareria_es.md`:

```markdown
# Alfarería tradicional

El alfarero moldea una vasija de barro en el torno.
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/eval/test_corpus_fixtures.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add eval/corpus/astronomia_it.md eval/corpus/cucito_it.md eval/corpus/beekeeping_en.md eval/corpus/ciclismo_es.md eval/corpus/alfareria_es.md tests/eval/test_corpus_fixtures.py
git commit -m "test(eval): add single-language corpus notes (it/en/es)"
```

---

### Task 4: Expanded gold set

Replace the 8-case gold set with the ~17-case set. A guard test (fast suite) enforces the single-correct-note assumption, that every expected note exists, and the full scenario coverage.

**Files:**
- Create: `tests/eval/test_gold_set.py`
- Modify: `eval/gold.json`
- Modify: `tests/eval/test_harness_integration.py`

- [ ] **Step 1: Write the failing guard test**

Create `tests/eval/test_gold_set.py`:

```python
from pathlib import Path

from ariostea.eval.harness import load_gold

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
GOLD = REPO / "eval" / "gold.json"


def test_every_expected_note_exists_and_is_single():
    for case in load_gold(GOLD):
        assert len(case.expected) == 1  # single-correct-note assumption holds
        assert (CORPUS / case.expected[0]).exists()


def test_gold_covers_all_scenarios():
    scenarios = {c.scenario for c in load_gold(GOLD)}
    assert scenarios == {
        "same",
        "en→it",
        "es→it",
        "it→en",
        "es→en",
        "en→es",
        "it→es",
        "accent",
        "inflection",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/eval/test_gold_set.py -v`
Expected: FAIL — `test_gold_covers_all_scenarios` fails (current set is `{same, en→it, it→en}`).

- [ ] **Step 3: Replace `eval/gold.json`**

```json
[
  {"query": "how to cook spaghetti at home", "query_lang": "en", "expected": ["pasta_en.md"], "scenario": "same"},
  {"query": "come cuocere gli spaghetti in casa", "query_lang": "it", "expected": ["pasta_it.md"], "scenario": "same"},
  {"query": "rolling dice in a board game", "query_lang": "en", "expected": ["dice_en.md"], "scenario": "same"},
  {"query": "tirare i dadi in un gioco da tavolo", "query_lang": "it", "expected": ["dadi_it.md"], "scenario": "same"},
  {"query": "osservare le stelle con il telescopio", "query_lang": "it", "expected": ["astronomia_it.md"], "scenario": "same"},
  {"query": "keeping bees and collecting honey", "query_lang": "en", "expected": ["beekeeping_en.md"], "scenario": "same"},
  {"query": "subir la montaña en bicicleta", "query_lang": "es", "expected": ["ciclismo_es.md"], "scenario": "same"},

  {"query": "looking at planets and stars through a telescope at night", "query_lang": "en", "expected": ["astronomia_it.md"], "scenario": "en→it"},
  {"query": "mirar las estrellas y los planetas con un telescopio", "query_lang": "es", "expected": ["astronomia_it.md"], "scenario": "es→it"},
  {"query": "apicoltura e raccolta del miele dall'alveare", "query_lang": "it", "expected": ["beekeeping_en.md"], "scenario": "it→en"},
  {"query": "apicultura y recoleccion de miel de la colmena", "query_lang": "es", "expected": ["beekeeping_en.md"], "scenario": "es→en"},
  {"query": "mountain biking up a steep trail on a bicycle", "query_lang": "en", "expected": ["ciclismo_es.md"], "scenario": "en→es"},
  {"query": "andare in bicicletta in montagna su un sentiero", "query_lang": "it", "expected": ["ciclismo_es.md"], "scenario": "it→es"},

  {"query": "città", "query_lang": "it", "expected": ["astronomia_it.md"], "scenario": "accent"},
  {"query": "montaña", "query_lang": "es", "expected": ["ciclismo_es.md"], "scenario": "accent"},

  {"query": "bottoni", "query_lang": "it", "expected": ["cucito_it.md"], "scenario": "inflection"},
  {"query": "vasijas", "query_lang": "es", "expected": ["alfareria_es.md"], "scenario": "inflection"}
]
```

- [ ] **Step 4: Update the scenario set in `tests/eval/test_harness_integration.py`**

Change the `set(by)` assertion to the full set:

```python
    assert set(by) == {
        "same",
        "en→it",
        "es→it",
        "it→en",
        "es→en",
        "en→es",
        "it→es",
        "accent",
        "inflection",
    }
```

- [ ] **Step 5: Run the fast suite to verify it passes**

Run: `uv run pytest -m "not integration" -q`
Expected: PASS (all green). `test_gold_set.py` passes; the model-backed integration test is deselected.

- [ ] **Step 6: Commit**

```bash
git add eval/gold.json tests/eval/test_gold_set.py tests/eval/test_harness_integration.py
git commit -m "test(eval): expand gold set to 17 cases across en/it/es scenarios"
```

---

### Task 5: Per-channel runner + integration hypotheses

Wire the three channels into `run_eval.py`, and add a model-backed integration test that encodes the design's hypotheses: accent terms hit on sparse; inflected queries miss on sparse but are recovered by dense.

**Files:**
- Create: `tests/eval/test_channels_integration.py`
- Modify: `eval/run_eval.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/eval/test_channels_integration.py`:

```python
from pathlib import Path

import pytest

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import make_dense_search_fn, make_sparse_search_fn
from ariostea.mcp.handlers import reindex_payload

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@pytest.mark.integration
def test_accent_hits_sparse_and_inflection_needs_dense(tmp_path):
    db = tmp_path / "eval.db"
    cfg = Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=str(db)),
    )
    container = build_container(cfg)
    reindex_payload(container)

    # A second read-only handle on the same indexed DB for raw channel access.
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
    store = SqliteStore(path=str(db), dim=embeddings.dimension)
    dense = make_dense_search_fn(embeddings, store, pool=50)
    sparse = make_sparse_search_fn(store, pool=50)

    # Accent: the diacritic fix means an accented keyword matches on sparse.
    assert "astronomia_it.md" in sparse("città", 5)
    assert "ciclismo_es.md" in sparse("montaña", 5)

    # Inflection: FTS has no stemming, so a plural query misses the singular
    # note on sparse — but multilingual dense embeddings recover it. This is
    # the evidence that keeps multilingual FTS stemming a YAGNI backlog item.
    assert "cucito_it.md" not in sparse("bottoni", 5)
    assert "cucito_it.md" in dense("bottoni", 5)
    assert "alfareria_es.md" not in sparse("vasijas", 5)
    assert "alfareria_es.md" in dense("vasijas", 5)
```

- [ ] **Step 2: Run the integration test and confirm it is green**

Run: `uv run pytest tests/eval/test_channels_integration.py -v -m integration`
Expected: PASS.

This is a **characterization test**, not a RED-first unit test: it depends only on the factories (Task 2) and the corpus (Task 3), so it passes as soon as those exist. Its job is to lock in the design's hypotheses against the real model. If any assertion **fails**, that is a genuine finding about the embedding model on this fixture (e.g. dense did not recover an inflected form) — investigate the cause (note too short? a competing craft note? wrong model?) and record it. Do **not** weaken an assertion to make it pass.

- [ ] **Step 3: Rewrite `eval/run_eval.py` to run all three channels**

```python
"""Run the cross-lingual eval harness against the committed fixture vault.

Usage:  uv run python eval/run_eval.py [k]

Loads the multilingual embedding model (downloads on first run), indexes the
fixture corpus into a throwaway database, and prints a recall@k / MRR table
per scenario for each retrieval channel (dense, sparse, hybrid).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import make_dense_search_fn, make_sparse_search_fn
from ariostea.eval.harness import dedupe, evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload, search_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
GOLD = EVAL_DIR / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Pull a generous chunk pool, then dedupe to notes before taking the top k.
CHUNK_POOL = 50


def make_hybrid_search_fn(container):
    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=CHUNK_POOL)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    return search_fn


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        cfg = Config(
            vault=VaultCfg(path=str(CORPUS), ignore=[]),
            embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
            store=StoreCfg(backend="sqlite", path=db),
        )
        container = build_container(cfg)
        reindex_payload(container)

        # Raw channels read the same indexed DB through a second handle, so the
        # production Container stays ports-only (no adapter leakage).
        embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
        store = SqliteStore(path=db, dim=embeddings.dimension)
        channels = {
            "DENSE": make_dense_search_fn(embeddings, store, CHUNK_POOL),
            "SPARSE": make_sparse_search_fn(store, CHUNK_POOL),
            "HYBRID": make_hybrid_search_fn(container),
        }

        for label, search_fn in channels.items():
            report = evaluate(cases, search_fn, k=k)
            print(f"\n=== {label} ===")
            print(format_report(report))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the eval script end-to-end to confirm it produces three reports**

Run: `uv run python eval/run_eval.py 5`
Expected: three blocks (`=== DENSE ===`, `=== SPARSE ===`, `=== HYBRID ===`), each a per-scenario table. Sanity check: SPARSE shows `accent` recall ≈ 1.0 and `inflection` recall ≈ 0.0; DENSE shows `inflection` recall ≈ 1.0.

- [ ] **Step 5: Run the full suite (including integration) to confirm everything is green**

Run: `uv run pytest -q`
Expected: PASS (all, including the integration tests). Output pristine.

- [ ] **Step 6: Commit**

```bash
git add eval/run_eval.py tests/eval/test_channels_integration.py
git commit -m "feat(eval): per-channel runner + accent/inflection hypotheses"
```

---

### Task 6: Record measured results

Write the observed numbers back into the design docs so the stemming decision has a durable evidence trail.

**Files:**
- Modify: `docs/superpowers/specs/2026-06-28-ariostea-gold-set-expansion-design.md`
- Modify: `docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md`

- [ ] **Step 1: Capture the numbers**

Run: `uv run python eval/run_eval.py 5`
Copy the three per-scenario tables.

- [ ] **Step 2: Append a "Measured results" section to the gold-set design**

In `docs/superpowers/specs/2026-06-28-ariostea-gold-set-expansion-design.md`, replace the §7 sentence "Measured numbers will be written back…" with an actual results block: paste the DENSE/SPARSE/HYBRID per-scenario tables, and state the verdict in one or two sentences (e.g. "sparse confirms the accent fix; inflection misses on sparse and is recovered by dense → multilingual FTS stemming stays YAGNI").

- [ ] **Step 3: Cross-reference from the multilingual-retrieval design**

In `docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md` §4.2, add a one-line pointer to the gold-set results (accent fix verified; stemming evidence recorded) with a link to the gold-set design file.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-28-ariostea-gold-set-expansion-design.md docs/superpowers/specs/2026-06-27-ariostea-multilingual-retrieval-design.md
git commit -m "docs(eval): record per-channel results and stemming verdict"
```

---

## Notes for the implementer

- **Run order matters.** Task 3 (notes) must precede Task 4 (gold references them) and Task 5 (integration indexes them).
- **Do not weaken the inflection assertions.** If `dense("bottoni")` fails to recover `cucito_it.md`, that is a real finding about the embedding model on this fixture — investigate (is the note too short? is another craft note competing?) rather than relaxing the test. Record what you learn.
- **Accents are real UTF-8** in both the corpus notes and `gold.json`; keep files UTF-8 encoded.
- **Integration tests** download the multilingual model on first run and are excluded from the fast suite (`-m "not integration"`).
