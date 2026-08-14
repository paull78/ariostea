# Span-Anchored Gold + Harness Upgrade — Implementation Plan

**Goal:** Add a span-anchored, dual-granularity evaluation capability to the harness so future chunking/blurb/BM25 experiments produce attributable signal — provable end-to-end on fixtures, with no network or LLM dependency.

**Architecture:** New pure-Python modules under `src/ariostea/eval/` for the richer gold schema, span-containment metrics, chunk-returning channels, and a span-level evaluator that reports metrics at both note and span granularity, broken down by query type. Everything is unit-tested with in-memory fakes; the real Wikipedia corpus and generated gold arrive in Plans 2 and 3.

**Tech stack:** Python 3.12, dataclasses, `pytest`. Follows the existing `eval/` module patterns (`harness.py`, `metrics.py`, `channels.py`).

This is Plan 1 of 3 for the eval-corpus expansion (spec: `docs/design/2026-07-09-eval-corpus-expansion.md`). Steps use `- [ ]` checkboxes for progress tracking.

---

## File Structure

- Create `src/ariostea/eval/wiki_gold.py` — gold schema (`AnswerSpan`, `WikiGoldCase`), loader, validator.
- Create `src/ariostea/eval/span_metrics.py` — whitespace normalization, span-containment, span-level recall@k / MRR.
- Create `src/ariostea/eval/spaneval.py` — `evaluate_spans`, report types, formatter.
- Modify `src/ariostea/eval/harness.py` — add the `SpanSearchFn` type alias.
- Modify `src/ariostea/eval/channels.py` — add chunk-returning channel factories.
- Create `eval/wiki/gold.sample.json` — a committed schema reference (2 rows).
- Tests: `tests/eval/test_wiki_gold.py`, `tests/eval/test_span_metrics.py`, `tests/eval/test_span_channels.py`, `tests/eval/test_spaneval.py`.

---

## Task 1: Gold schema and loader

**Files:**
- Create: `src/ariostea/eval/wiki_gold.py`
- Test: `tests/eval/test_wiki_gold.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_wiki_gold.py
import json

from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase, load_wiki_gold


def test_load_wiki_gold_parses_spans(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "how is a violin tuned",
                    "query_lang": "en",
                    "type": "buried",
                    "scenario": "buried",
                    "expected_notes": ["string-instruments/violin.md"],
                    "answer_spans": [
                        {"note": "string-instruments/violin.md", "text": "tuned in perfect fifths"}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_wiki_gold(path)

    assert cases == [
        WikiGoldCase(
            query="how is a violin tuned",
            query_lang="en",
            type="buried",
            scenario="buried",
            expected_notes=("string-instruments/violin.md",),
            answer_spans=(
                AnswerSpan(note="string-instruments/violin.md", text="tuned in perfect fifths"),
            ),
        )
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_gold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.wiki_gold'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/wiki_gold.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Query types that stress a specific retrieval track (see the eval-corpus design doc).
SPAN_TYPES = ("paraphrase", "exact_term", "buried", "cross_lingual")


@dataclass(frozen=True)
class AnswerSpan:
    note: str
    text: str


@dataclass(frozen=True)
class WikiGoldCase:
    query: str
    query_lang: str
    type: str
    scenario: str
    expected_notes: tuple[str, ...]
    answer_spans: tuple[AnswerSpan, ...]


def load_wiki_gold(path: str | Path) -> list[WikiGoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        WikiGoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            type=row["type"],
            scenario=row["scenario"],
            expected_notes=tuple(row["expected_notes"]),
            answer_spans=tuple(
                AnswerSpan(note=span["note"], text=span["text"]) for span in row["answer_spans"]
            ),
        )
        for row in rows
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wiki_gold.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wiki_gold.py tests/eval/test_wiki_gold.py
git commit -m "feat(eval): span-anchored wiki gold schema and loader"
```

---

## Task 2: Span-containment metrics

**Files:**
- Create: `src/ariostea/eval/span_metrics.py`
- Test: `tests/eval/test_span_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_span_metrics.py
from ariostea.eval.span_metrics import (
    chunk_contains_span,
    normalize_ws,
    span_recall_at_k,
    span_reciprocal_rank,
)
from ariostea.eval.wiki_gold import AnswerSpan


def test_normalize_collapses_whitespace_and_case():
    assert normalize_ws("The  Violin\nis\tTuned") == "the violin is tuned"


def test_chunk_contains_span_ignores_whitespace_and_case():
    assert chunk_contains_span("The violin is  tuned in\nperfect fifths.", "Tuned In Perfect Fifths")
    assert not chunk_contains_span("A cello has four strings.", "tuned in perfect fifths")


def test_span_recall_requires_matching_note_and_text():
    spans = (AnswerSpan(note="violin.md", text="perfect fifths"),)
    retrieved = [("cello.md", "perfect fifths"), ("violin.md", "tuned in perfect fifths")]
    # right text but wrong note at rank 1; correct note at rank 2 → within k=2, not k=1
    assert span_recall_at_k(spans, retrieved, k=1) == 0.0
    assert span_recall_at_k(spans, retrieved, k=2) == 1.0


def test_span_reciprocal_rank_uses_first_true_hit():
    spans = (AnswerSpan(note="violin.md", text="perfect fifths"),)
    retrieved = [("cello.md", "perfect fifths"), ("violin.md", "tuned in perfect fifths")]
    assert span_reciprocal_rank(spans, retrieved) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_span_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.span_metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/span_metrics.py
from __future__ import annotations

import re

from ariostea.eval.wiki_gold import AnswerSpan

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Lowercase and collapse all runs of whitespace to single spaces."""
    return _WS.sub(" ", text).strip().lower()


def chunk_contains_span(chunk_text: str, span_text: str) -> bool:
    return normalize_ws(span_text) in normalize_ws(chunk_text)


def _is_hit(spans: tuple[AnswerSpan, ...], note_path: str, chunk_text: str) -> bool:
    return any(
        span.note == note_path and chunk_contains_span(chunk_text, span.text) for span in spans
    )


def span_recall_at_k(
    spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]], k: int
) -> float:
    """1.0 if any of the top-k retrieved (note_path, chunk_text) pairs contains
    an answer span in its own note, else 0.0."""
    return 1.0 if any(_is_hit(spans, note, text) for note, text in retrieved[:k]) else 0.0


def span_reciprocal_rank(
    spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]]
) -> float:
    for index, (note, text) in enumerate(retrieved):
        if _is_hit(spans, note, text):
            return 1.0 / (index + 1)
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_span_metrics.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/span_metrics.py tests/eval/test_span_metrics.py
git commit -m "feat(eval): span-containment recall@k and reciprocal rank"
```

---

## Task 3: Gold validator

**Files:**
- Modify: `src/ariostea/eval/wiki_gold.py`
- Test: `tests/eval/test_wiki_gold.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/eval/test_wiki_gold.py
from ariostea.eval.wiki_gold import validate_wiki_gold


def _case(**overrides):
    base = dict(
        query="q",
        query_lang="en",
        type="buried",
        scenario="buried",
        expected_notes=("violin.md",),
        answer_spans=(AnswerSpan(note="violin.md", text="perfect fifths"),),
    )
    base.update(overrides)
    return WikiGoldCase(**base)


def test_validate_accepts_well_formed_case():
    notes = {"violin.md": "The violin is tuned in perfect fifths."}
    assert validate_wiki_gold([_case()], notes) == []


def test_validate_flags_missing_span_text_unknown_type_and_empty_notes():
    notes = {"violin.md": "The violin is tuned in perfect fifths."}
    cases = [
        _case(answer_spans=(AnswerSpan(note="violin.md", text="not in the article"),)),
        _case(type="mystery"),
        _case(expected_notes=()),
        _case(answer_spans=(AnswerSpan(note="missing.md", text="perfect fifths"),)),
    ]
    errors = validate_wiki_gold(cases, notes)
    assert any("span text not found" in e for e in errors)
    assert any("unknown type" in e for e in errors)
    assert any("expected_notes is empty" in e for e in errors)
    assert any("not in corpus" in e for e in errors)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_gold.py -v`
Expected: FAIL with `ImportError: cannot import name 'validate_wiki_gold'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/ariostea/eval/wiki_gold.py
from ariostea.eval.span_metrics import normalize_ws


def validate_wiki_gold(cases: list[WikiGoldCase], notes: dict[str, str]) -> list[str]:
    """Return a list of human-readable errors; an empty list means valid.

    `notes` maps a note path to its full text. A case is valid when it names at
    least one expected note, uses a known query type, and every answer span both
    references a corpus note and appears verbatim (whitespace/case-insensitive)
    in that note.
    """
    errors: list[str] = []
    for i, case in enumerate(cases):
        if not case.expected_notes:
            errors.append(f"case {i}: expected_notes is empty")
        if case.type not in SPAN_TYPES:
            errors.append(f"case {i}: unknown type {case.type!r}")
        if not case.answer_spans:
            errors.append(f"case {i}: no answer_spans")
        for span in case.answer_spans:
            if span.note not in notes:
                errors.append(f"case {i}: span note {span.note!r} not in corpus")
            elif normalize_ws(span.text) not in normalize_ws(notes[span.note]):
                errors.append(f"case {i}: span text not found in {span.note!r}")
    return errors
```

Note: the `from ariostea.eval.span_metrics import normalize_ws` line goes at the top of `wiki_gold.py` with the other imports; `span_metrics` imports only from `wiki_gold`'s dataclasses, and this import is used only inside the function, so there is no import cycle at module load. To be safe, place the import inside `validate_wiki_gold` if a cycle appears.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_wiki_gold.py -v`
Expected: PASS (both the loader and validator tests)

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/wiki_gold.py tests/eval/test_wiki_gold.py
git commit -m "feat(eval): validate wiki gold spans against corpus notes"
```

---

## Task 4: `SpanSearchFn` type and chunk-returning channels

**Files:**
- Modify: `src/ariostea/eval/harness.py` (add type alias near the existing `SearchFn`)
- Modify: `src/ariostea/eval/channels.py`
- Test: `tests/eval/test_span_channels.py`

- [ ] **Step 1: Add the `SpanSearchFn` type alias**

In `src/ariostea/eval/harness.py`, directly below the existing `SearchFn` definition, add:

```python
# A span ranker: given (query, k), return up to k (note_path, chunk_text) pairs
# in rank order (best first) — chunk-level, NOT deduped to notes.
SpanSearchFn = Callable[[str, int], list[tuple[str, str]]]
```

- [ ] **Step 2: Write the failing test**

```python
# tests/eval/test_span_channels.py
from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.eval.channels import make_dense_chunk_fn, make_sparse_chunk_fn


def _rc(note_path, text, ordinal=0):
    chunk = Chunk(
        note_path=note_path, ordinal=ordinal, heading_path=("H",), text=text, token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=1.0)


class _FakeEmbeddings:
    def embed_query(self, text):
        return [0.0, 0.0, 0.0]


class _FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def dense(self, vec, k, filters=None):
        return self._hits[:k]

    def sparse(self, query, k, filters=None):
        return self._hits[:k]


def test_dense_chunk_fn_returns_note_text_pairs_truncated_to_k():
    hits = [_rc("violin.md", "tuned in fifths"), _rc("cello.md", "four strings")]
    fn = make_dense_chunk_fn(_FakeEmbeddings(), _FakeRetriever(hits), pool=10)
    assert fn("q", k=1) == [("violin.md", "tuned in fifths")]
    assert fn("q", k=2) == [("violin.md", "tuned in fifths"), ("cello.md", "four strings")]


def test_sparse_chunk_fn_returns_note_text_pairs():
    hits = [_rc("guitar.md", "six strings")]
    fn = make_sparse_chunk_fn(_FakeRetriever(hits), pool=10)
    assert fn("q", k=5) == [("guitar.md", "six strings")]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_span_channels.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_dense_chunk_fn'`

- [ ] **Step 4: Write minimal implementation**

Add to `src/ariostea/eval/channels.py`. Update the import line at the top from `from ariostea.eval.harness import SearchFn, dedupe` to:

```python
from ariostea.eval.harness import SearchFn, SpanSearchFn, dedupe
```

Then append the three factories (mirrors the existing note-level ones, but keeps chunk granularity — no dedupe):

```python
def make_dense_chunk_fn(
    embeddings: EmbeddingProvider, retriever: ChunkRetriever, pool: int
) -> SpanSearchFn:
    def search_fn(query: str, k: int) -> list[tuple[str, str]]:
        vec = embeddings.embed_query(query)
        hits = retriever.dense(vec=vec, k=pool, filters=None)
        return [(h.chunk.note_path, h.chunk.text) for h in hits][:k]

    return search_fn


def make_sparse_chunk_fn(retriever: ChunkRetriever, pool: int) -> SpanSearchFn:
    def search_fn(query: str, k: int) -> list[tuple[str, str]]:
        hits = retriever.sparse(query=query, k=pool, filters=None)
        return [(h.chunk.note_path, h.chunk.text) for h in hits][:k]

    return search_fn


def make_hybrid_chunk_fn(container: "Container", pool: int) -> SpanSearchFn:
    """Full blended pipeline at chunk granularity (no dedupe to notes)."""

    def search_fn(query: str, k: int) -> list[tuple[str, str]]:
        payload = search_payload(container, query=query, k=pool)
        return [(r["note_path"], r["text"]) for r in payload["results"]][:k]

    return search_fn
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_span_channels.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/harness.py src/ariostea/eval/channels.py tests/eval/test_span_channels.py
git commit -m "feat(eval): chunk-returning channels and SpanSearchFn type"
```

---

## Task 5: Span-level evaluator with per-type breakdown

**Files:**
- Create: `src/ariostea/eval/spaneval.py`
- Test: `tests/eval/test_spaneval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/eval/test_spaneval.py
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase


def _case(query, type_, note, span_text):
    return WikiGoldCase(
        query=query,
        query_lang="en",
        type=type_,
        scenario=type_,
        expected_notes=(note,),
        answer_spans=(AnswerSpan(note=note, text=span_text),),
    )


def test_evaluate_spans_reports_note_and_span_metrics_per_type():
    cases = [
        _case("q_hit", "paraphrase", "violin.md", "perfect fifths"),
        _case("q_miss", "exact_term", "cello.md", "spruce top"),
    ]

    # q_hit: correct note + span text at rank 1. q_miss: right note but span text absent.
    responses = {
        "q_hit": [("violin.md", "the violin is tuned in perfect fifths")],
        "q_miss": [("cello.md", "the cello is large")],
    }

    def span_fn(query, k):
        return responses[query][:k]

    report = evaluate_spans(cases, span_fn, k=5)

    assert report.overall.n == 2
    assert report.overall.note_recall_at_k == 1.0  # both retrieved the right note
    assert report.overall.span_recall_at_k == 0.5  # only q_hit matched the span
    by_type = {s.group: s for s in report.by_type}
    assert by_type["paraphrase"].span_recall_at_k == 1.0
    assert by_type["exact_term"].span_recall_at_k == 0.0


def test_format_span_report_includes_overall_row():
    cases = [_case("q_hit", "paraphrase", "violin.md", "perfect fifths")]

    def span_fn(query, k):
        return [("violin.md", "tuned in perfect fifths")]

    text = format_span_report(evaluate_spans(cases, span_fn, k=5))
    assert "overall" in text
    assert "paraphrase" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_spaneval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ariostea.eval.spaneval'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/ariostea/eval/spaneval.py
from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.metrics import recall_at_k, reciprocal_rank
from ariostea.eval.span_metrics import span_recall_at_k, span_reciprocal_rank
from ariostea.eval.wiki_gold import WikiGoldCase


@dataclass(frozen=True)
class SpanScore:
    group: str
    n: int
    note_recall_at_k: float
    note_mrr: float
    span_recall_at_k: float
    span_mrr: float


@dataclass(frozen=True)
class SpanEvalReport:
    k: int
    overall: SpanScore
    by_type: tuple[SpanScore, ...]


def _dedupe_notes(retrieved: list[tuple[str, str]]) -> list[str]:
    seen: list[str] = []
    for note, _ in retrieved:
        if note not in seen:
            seen.append(note)
    return seen


# Each scored row is (note_recall, note_mrr, span_recall, span_mrr).
def _aggregate(group: str, rows: list[tuple[float, float, float, float]]) -> SpanScore:
    n = len(rows)
    if n == 0:
        return SpanScore(group, 0, 0.0, 0.0, 0.0, 0.0)
    return SpanScore(
        group=group,
        n=n,
        note_recall_at_k=sum(r[0] for r in rows) / n,
        note_mrr=sum(r[1] for r in rows) / n,
        span_recall_at_k=sum(r[2] for r in rows) / n,
        span_mrr=sum(r[3] for r in rows) / n,
    )


def evaluate_spans(cases: list[WikiGoldCase], span_fn: SpanSearchFn, k: int) -> SpanEvalReport:
    """Run every case through span_fn once and aggregate note-level and
    span-level recall@k / MRR, overall and grouped by query type."""
    scored: list[tuple[str, tuple[float, float, float, float]]] = []
    for case in cases:
        retrieved = span_fn(case.query, k)
        notes = _dedupe_notes(retrieved)
        expected = set(case.expected_notes)
        row = (
            recall_at_k(expected, notes, k),
            reciprocal_rank(expected, notes),
            span_recall_at_k(case.answer_spans, retrieved, k),
            span_reciprocal_rank(case.answer_spans, retrieved),
        )
        scored.append((case.type, row))

    overall = _aggregate("overall", [row for _, row in scored])
    types = sorted({t for t, _ in scored})
    by_type = tuple(
        _aggregate(t, [row for typ, row in scored if typ == t]) for t in types
    )
    return SpanEvalReport(k=k, overall=overall, by_type=by_type)


def format_span_report(report: SpanEvalReport) -> str:
    header = (
        f"{'type':<14} {'n':>3}  note_r@{report.k:<2} note_mrr  span_r@{report.k:<2} span_mrr"
    )
    lines = [header]
    for s in (*report.by_type, report.overall):
        lines.append(
            f"{s.group:<14} {s.n:>3}  "
            f"{s.note_recall_at_k:>7.3f} {s.note_mrr:>7.3f}  "
            f"{s.span_recall_at_k:>7.3f} {s.span_mrr:>7.3f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_spaneval.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/spaneval.py tests/eval/test_spaneval.py
git commit -m "feat(eval): span-level evaluator with per-type breakdown"
```

---

## Task 6: Committed schema sample and smoke test

**Files:**
- Create: `eval/wiki/gold.sample.json`
- Test: `tests/eval/test_wiki_gold.py`

- [ ] **Step 1: Write the schema sample**

```json
[
  {
    "query": "how is a violin tuned",
    "query_lang": "en",
    "type": "buried",
    "scenario": "buried",
    "expected_notes": ["string-instruments/violin.md"],
    "answer_spans": [
      {"note": "string-instruments/violin.md", "text": "tuned in perfect fifths: G, D, A, E"}
    ]
  },
  {
    "query": "quante corde ha un violoncello",
    "query_lang": "it",
    "type": "cross_lingual",
    "scenario": "en→it",
    "expected_notes": ["string-instruments/cello.md"],
    "answer_spans": [
      {"note": "string-instruments/cello.md", "text": "four strings tuned in perfect fifths"}
    ]
  }
]
```

- [ ] **Step 2: Write the failing test**

```python
# append to tests/eval/test_wiki_gold.py
from pathlib import Path

SAMPLE = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "gold.sample.json"


def test_committed_schema_sample_loads_and_has_expected_shape():
    cases = load_wiki_gold(SAMPLE)
    assert len(cases) == 2
    assert {c.type for c in cases} == {"buried", "cross_lingual"}
    assert all(c.expected_notes and c.answer_spans for c in cases)
    for case in cases:
        for span in case.answer_spans:
            assert span.note in case.expected_notes
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_wiki_gold.py::test_committed_schema_sample_loads_and_has_expected_shape -v`
Expected: FAIL (file missing) — if you wrote the JSON in Step 1 first, instead confirm it PASSES and the sample is what drives it.

- [ ] **Step 4: Run the full eval test suite**

Run: `uv run pytest tests/eval -v`
Expected: PASS (all eval tests, old and new)

- [ ] **Step 5: Commit**

```bash
git add eval/wiki/gold.sample.json tests/eval/test_wiki_gold.py
git commit -m "test(eval): committed span-gold schema sample and smoke test"
```

---

## Task 7: Full verification

- [ ] **Step 1: Run lint, format check, and the fast suite**

Run:
```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -m "not integration"
```
Expected: all pass. If `ruff format --check` reports changes, run `uv run ruff format .`, re-run the checks, and amend the last commit.

- [ ] **Step 2: Confirm no production code changed**

Run: `git diff --stat master -- src/ariostea | grep -v '/eval/'`
Expected: empty output — this plan touches only `src/ariostea/eval/` and tests, never the production retrieval path.

---

## Self-Review

- **Spec coverage:** span-anchored dual-granularity gold (Tasks 1, 3, 6) ✓; span-containment metric surviving re-chunking (Task 2) ✓; per-`type` breakdown (Task 5) ✓; note-level + span-level reporting (Task 5) ✓; chunk-granularity channels feeding the metric (Task 4) ✓. Corpus acquisition, LLM generation, and the difficulty guard are **out of scope for Plan 1** (Plans 2 and 3) — by design.
- **Type consistency:** `SpanSearchFn = list[tuple[str, str]]` is defined in `harness.py` (Task 4) and consumed identically in `channels.py` (Task 4) and `spaneval.py` (Task 5). `AnswerSpan`/`WikiGoldCase` fields (`note`, `text`, `expected_notes`, `answer_spans`, `type`) are used consistently across Tasks 1–6.
- **Placeholder scan:** every code step contains complete code; no TBD/TODO.
