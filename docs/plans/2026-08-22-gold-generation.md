# Gold Generation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate, validate and commit ~150 span-anchored gold queries over the pinned Wikipedia corpus, so retrieval experiments (chunking, contextual blurbs, BM25 tuning) produce measurable per-mechanism signal.

**Architecture:** Pure, unit-testable modules in `src/ariostea/eval/` do passage selection, prompt construction, response parsing, validation and filtering; two thin runner scripts in `eval/` wire them to a live OpenAI-compatible endpoint and to a throwaway index of the corpus. Generation goes through the existing `ChatProvider` port, so every module is tested against a fake provider with no network. The LLM is the only nondeterminism in the pipeline — passage selection, validation and filtering are all deterministic.

**Tech Stack:** Python 3.12, pytest, the existing `OpenAICompatChat` adapter, LM Studio serving two local models (`qwen/qwen3.6-35b-a3b` as generator, `qwen2.5-14b-instruct-mlx` as adversarial judge), fastembed + sqlite-vec for the discrimination filter.

**Spec:** `docs/design/2026-07-09-eval-corpus-expansion.md`, sections "2. Gold generation pipeline", "3. Gold schema", "4. Harness upgrade".

**Prior plans:** Plan 1 (`docs/plans/2026-07-13-eval-harness-span-gold.md`, merged) built the gold schema and span metrics. Plan 2 (`docs/plans/2026-08-14-wiki-corpus-acquisition.md`, merged) built the 79-article pinned corpus under `eval/wiki/`.

---

## Context an implementer needs

**What already exists and must be reused, not rebuilt:**

- `src/ariostea/eval/wiki_gold.py` — `AnswerSpan`, `WikiGoldCase`, `SPAN_TYPES = ("paraphrase", "exact_term", "buried", "cross_lingual")`, `load_wiki_gold`, `validate_wiki_gold(cases, notes) -> list[str]`.
- `src/ariostea/eval/normalize.py` — `normalize_ws(text)` lowercases and collapses whitespace. **Every** textual comparison in this plan goes through it.
- `src/ariostea/eval/span_metrics.py` — `span_recall_at_k(spans, retrieved, k)`, `span_reciprocal_rank(spans, retrieved)`. `retrieved` is a list of `(note_path, chunk_text)` pairs.
- `src/ariostea/eval/spaneval.py` — `evaluate_spans(cases, span_fn, k, pool)`, `SpanScore`, `SpanEvalReport`, `format_span_report`.
- `src/ariostea/eval/harness.py` — `SpanSearchFn = Callable[[str, int], list[tuple[str, str]]]`, `dedupe`.
- `src/ariostea/eval/channels.py` — `make_dense_chunk_fn`, `make_sparse_chunk_fn`, `make_hybrid_chunk_fn`.
- `src/ariostea/ports/chat.py` — `ChatProvider` Protocol, one method: `complete(system: str, user: str) -> str`.
- `src/ariostea/adapters/chat/openai_compat.py` — `OpenAICompatChat(base_url, model, api_key, timeout, max_tokens, client)`, raises `ChatError`. Temperature is hardcoded to 0.
- `eval/wiki/` — 79 notes as `<cluster>/<slug>.md`, each with a frontmatter block (`title`, `lang`, `cluster`, `revid`, `source`, `license`) followed by `# <Title>` and the body.

**Architecture rule:** `src/ariostea/eval/` is the outermost ring and is eval-only. It may import from anywhere in the package and may import third-party libraries directly. **No task in this plan may modify anything under `src/ariostea/` outside `eval/`.** A guard command in Task 14 enforces this.

**Test conventions:** tests live in `tests/eval/test_<module>.py`. Tests that load a real model or hit a network service are marked `@pytest.mark.integration`. The fast suite is `uv run pytest -m "not integration"`.

---

## File structure

| File | Responsibility |
| --- | --- |
| `src/ariostea/eval/wiki_notes.py` | Read the committed corpus back off disk: strip frontmatter, map note path → body, note path → title, note path → cluster. |
| `src/ariostea/eval/gold_passages.py` | Turn notes into candidate `Passage` objects and pick which ones feed which query type. This is where "a query that stresses BM25" is actually decided. |
| `src/ariostea/eval/gold_prompts.py` | The generation and judge prompts, plus `parse_json_object` — the tolerant reader for local-model responses. |
| `src/ariostea/eval/gold_generate.py` | One generation call: passage + type → `Candidate`, over the `ChatProvider` port. |
| `src/ariostea/eval/gold_validate.py` | Validation stages 1 (automatic) and 2 (adversarial judge). Returns a rejection reason string or `None`. |
| `src/ariostea/eval/gold_discriminate.py` | Validation stage 3: drop cases every channel already answers at rank 1. |
| `src/ariostea/eval/wiki_index.py` | Index `eval/wiki/` into a throwaway DB and expose the three chunk-level channels. Shared by both runners. |
| `src/ariostea/eval/difficulty.py` | Per-cluster dense-only baseline and the ≥0.95 "too easy" flag. |
| `src/ariostea/eval/metrics.py` (modify) | Add `ndcg_at_k`. |
| `src/ariostea/eval/span_metrics.py` (modify) | Add `span_ndcg_at_k`. |
| `src/ariostea/eval/spaneval.py` (modify) | Report nDCG alongside recall/MRR. |
| `eval/generate_gold.py` | Runner: select → generate → gate 1 → gate 2 → gate 3 → write `eval/wiki/gold.json`, `gold_rejected.json`, `gold.meta.json`, `gold_review.md`. |
| `eval/run_wiki_eval.py` | Runner: index the corpus, load the gold, print the per-type span report per channel plus the difficulty guard. |

**Why this split:** everything except the two `eval/*.py` runners is pure or port-dependent, so the whole pipeline is unit-testable with a fake `ChatProvider` and no network. The runners hold the I/O, the CLI and the wiring, exactly as `eval/build_wiki_corpus.py` does for Plan 2.

---

### Task 1: Corpus readback

**Files:**
- Create: `src/ariostea/eval/wiki_notes.py`
- Test: `tests/eval/test_wiki_notes.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from ariostea.eval.wiki_notes import cluster_of, load_corpus_notes, note_titles, strip_frontmatter

NOTE = """---
title: Violin
lang: en
cluster: string-instruments
revid: 12345
source: https://en.wikipedia.org/w/index.php?title=Violin&oldid=12345
license: CC BY-SA 4.0
---

# Violin

The violin is a wooden chordophone.
"""


def test_strip_frontmatter_removes_the_block_and_leading_blank_lines():
    assert strip_frontmatter(NOTE).startswith("# Violin")


def test_strip_frontmatter_leaves_a_note_without_frontmatter_alone():
    assert strip_frontmatter("# Violin\n\nBody.\n") == "# Violin\n\nBody.\n"


def test_strip_frontmatter_leaves_an_unterminated_block_alone():
    # A truncated file must not have its whole body silently eaten.
    raw = "---\ntitle: Violin\n\n# Violin\n"
    assert strip_frontmatter(raw) == raw


def test_load_corpus_notes_keys_by_cluster_relative_path(tmp_path: Path):
    (tmp_path / "string-instruments").mkdir()
    (tmp_path / "string-instruments" / "violin.md").write_text(NOTE, encoding="utf-8")
    notes = load_corpus_notes(tmp_path)
    assert list(notes) == ["string-instruments/violin.md"]
    assert notes["string-instruments/violin.md"].startswith("# Violin")


def test_note_titles_reads_the_h1():
    assert note_titles({"a/violin.md": "# Violin\n\nBody."}) == {"a/violin.md": "Violin"}


def test_note_titles_rejects_a_note_with_no_h1():
    # Every corpus note is rendered with an H1; a missing one means the file is
    # damaged, and a silent path-stem fallback would hide that from the gate
    # that uses titles to reject title-answerable queries.
    with pytest.raises(ValueError, match="no H1"):
        note_titles({"a/violin.md": "Body with no heading."})


def test_cluster_of_takes_the_first_path_segment():
    assert cluster_of("string-instruments/violin.md") == "string-instruments"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_wiki_notes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.wiki_notes'`

- [ ] **Step 3: Write the implementation**

```python
"""Read the committed Wikipedia corpus back off disk.

Gold generation and validation both need the note *body* — the text a span
must appear in — not the frontmatter that `wiki_corpus.render_note` wrapped it
in. This module is the one place that knows how to undo that wrapping, so the
two consumers cannot drift apart on what "the note text" means.
"""

from __future__ import annotations

import re
from pathlib import Path

_FENCE = "---\n"
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def strip_frontmatter(raw: str) -> str:
    """Drop a leading `---\\n...\\n---\\n` block and the blank lines after it.

    Returns `raw` unchanged when there is no frontmatter, and — deliberately —
    also when the opening fence is never closed. A truncated note is a damaged
    note; returning its whole text keeps the damage visible to the caller
    (the H1 check in `note_titles`, the span check in the validation gate)
    instead of silently returning an empty body that would just look like an
    article with nothing in it.
    """
    if not raw.startswith(_FENCE):
        return raw
    end = raw.find("\n---\n", len(_FENCE) - 1)
    if end == -1:
        return raw
    return raw[end + len("\n---\n") :].lstrip("\n")


def load_corpus_notes(wiki_dir: Path) -> dict[str, str]:
    """Every `<cluster>/<slug>.md` under `wiki_dir`, keyed by that relative
    path, frontmatter stripped.

    Globbed one level deep on purpose: the corpus layout is exactly one
    directory per cluster, and a recursive glob would also pick up anything a
    future subdirectory holds. Paths are sorted so every consumer — passage
    selection above all — iterates in the same order on every machine.
    """
    return {
        f"{path.parent.name}/{path.name}": strip_frontmatter(path.read_text(encoding="utf-8"))
        for path in sorted(wiki_dir.glob("*/*.md"))
    }


def note_titles(notes: dict[str, str]) -> dict[str, str]:
    """Note path -> article title, read from each note's H1.

    Raises on a note with no H1 rather than falling back to the path stem.
    The title is used to reject queries answerable from the title alone, so a
    wrong title silently weakens a validation gate — the failure has to be loud.
    """
    titles: dict[str, str] = {}
    for path, text in notes.items():
        match = _H1.search(text)
        if match is None:
            raise ValueError(f"note {path!r} has no H1 heading to read a title from")
        titles[path] = match.group(1).strip()
    return titles


def cluster_of(note_path: str) -> str:
    """`string-instruments/violin.md` -> `string-instruments`."""
    return note_path.split("/", 1)[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_wiki_notes.py -v`
Expected: 7 passed

- [ ] **Step 5: Verify it reads the real corpus**

Run:
```bash
uv run python -c "
from pathlib import Path
from ariostea.eval.wiki_notes import load_corpus_notes, note_titles
notes = load_corpus_notes(Path('eval/wiki'))
titles = note_titles(notes)
print(len(notes), 'notes')
print(sorted(notes)[0], '->', titles[sorted(notes)[0]])
assert len(notes) == 79, len(notes)
assert not any(t.startswith('---') for t in notes.values())
print('OK')
"
```
Expected: `79 notes`, a title line, then `OK`

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/wiki_notes.py tests/eval/test_wiki_notes.py
git commit -m "feat(eval): read the committed wiki corpus back off disk"
```

---

### Task 2: Passage splitting and rare-term detection

**Files:**
- Create: `src/ariostea/eval/gold_passages.py`
- Test: `tests/eval/test_gold_passages.py`

- [ ] **Step 1: Write the failing tests**

```python
from collections import Counter

from ariostea.eval.gold_passages import (
    MAX_CHARS,
    Passage,
    document_frequency,
    rare_terms,
    split_passages,
)

BODY = (
    "# Violin\n\n"
    + "The violin is a wooden chordophone in the violin family. " * 6
    + "\n\n## Construction\n\n"
    + "Most violins have a hollow wooden body with a spruce top. " * 6
    + "\n\n- a list item that is not prose\n"
)


def test_split_passages_attributes_a_heading_to_each_passage():
    passages = split_passages("a/violin.md", BODY)
    assert {p.heading for p in passages} == {"", "Construction"}


def test_split_passages_records_the_offset_within_the_note():
    passages = split_passages("a/violin.md", BODY)
    construction = next(p for p in passages if p.heading == "Construction")
    assert BODY[construction.offset :].startswith(construction.text[:40])
    assert construction.offset > 0


def test_split_passages_emits_text_that_is_verbatim_in_the_note():
    # The whole pipeline rests on this: a span copied from a passage must be
    # findable in the note, so a passage must be a substring of the note.
    for passage in split_passages("a/violin.md", BODY):
        assert passage.text in BODY


def test_split_passages_skips_list_items():
    joined = " ".join(p.text for p in split_passages("a/violin.md", BODY))
    assert "a list item that is not prose" not in joined


def test_split_passages_never_exceeds_the_maximum_length():
    for passage in split_passages("a/violin.md", BODY):
        assert len(passage.text) <= MAX_CHARS


def test_split_passages_drops_a_paragraph_shorter_than_the_minimum():
    assert split_passages("a/x.md", "# X\n\nToo short.\n") == []


def test_document_frequency_counts_notes_not_occurrences():
    df = document_frequency({"a.md": "violin violin violin", "b.md": "cello"})
    assert df["violin"] == 1
    assert df["cello"] == 1


def test_document_frequency_ignores_short_tokens_and_digits():
    df = document_frequency({"a.md": "the 1737 sul ponticello"})
    assert "the" not in df and "1737" not in df
    assert df["ponticello"] == 1


def test_rare_terms_keeps_only_tokens_in_few_notes():
    df = Counter({"violin": 40, "ponticello": 1})
    assert rare_terms("Bow sul ponticello on the violin", df) == ("ponticello",)


def test_rare_terms_deduplicates_and_preserves_first_appearance_order():
    df = Counter({"ponticello": 1, "tasto": 2})
    assert rare_terms("tasto ponticello tasto", df) == ("tasto", "ponticello")


def test_passage_cluster_comes_from_the_note_path():
    passage = Passage(note="cheese/brie.md", heading="", text="x", offset=0, note_chars=1)
    assert passage.cluster == "cheese"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_passages.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.gold_passages'`

- [ ] **Step 3: Write the implementation**

```python
"""Choose the passages gold queries are generated from.

This module, not the prompt, is where "a query that stresses BM25" is decided.
A prompt can ask for a rare-term query, but only passage selection can
guarantee a rare term is present to build one from; a prompt can ask for a
buried fact, but only selection knows whether the passage is actually buried.
The prompt phrases the request, this module makes it satisfiable.

Selection is fully deterministic — notes in sorted path order, passages in
document order — so the same corpus always offers the same candidates. The
LLM is the only nondeterminism in the pipeline, which is as much as one
generation run can reasonably contain.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace

from ariostea.eval.wiki_notes import cluster_of

_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
# Letters only, four or more: skips digits, punctuation and short function
# words without needing a stopword list per language.
_WORD = re.compile(r"[^\W\d_]{4,}", re.UNICODE)

# A passage has to state enough to hold an answer, and stay short enough that
# a model copying a span out of it does not have to search a page of text.
MIN_CHARS = 300
MAX_CHARS = 1500
# A note has to be long before a fact inside it counts as "buried".
BURIED_MIN_NOTE_CHARS = 8000
# ...and the passage has to sit past this fraction of the way into it.
BURIED_MIN_OFFSET = 0.4
# A token appearing in at most this many notes is rare enough that a lexical
# channel should find it and a dense channel may not.
RARE_MAX_NOTES = 2


@dataclass(frozen=True)
class Passage:
    note: str
    heading: str
    text: str
    offset: int  # character offset of `text` within the note body
    note_chars: int
    rare_terms: tuple[str, ...] = ()

    @property
    def cluster(self) -> str:
        return cluster_of(self.note)


def _sections(text: str) -> list[tuple[str, int, str]]:
    """`(heading, offset of the body, body)` per `##` section.

    Text before the first `##` is returned with an empty heading — that is the
    article lead, which is prose worth generating from.
    """
    matches = list(_H2.finditer(text))
    if not matches:
        return [("", 0, text)]
    sections = [("", 0, text[: matches[0].start()])]
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), match.end(), text[match.end() : end]))
    return sections


def _is_prose(paragraph: str) -> bool:
    """A paragraph worth generating a query from: not a heading, list, quote or
    table row, and ending like a sentence."""
    stripped = paragraph.strip()
    if not stripped or stripped[0] in "#-*>|":
        return False
    return stripped.endswith((".", ".)", '."', "!", "?"))
```

Continue the same file with the splitter:

```python
def split_passages(note: str, text: str) -> list[Passage]:
    """Candidate passages from one note body, in document order.

    Consecutive prose paragraphs are accumulated until they reach `MIN_CHARS`
    and emitted if they still fit in `MAX_CHARS`. A run that overshoots
    `MAX_CHARS` is discarded rather than truncated: a truncated passage could
    end mid-sentence, and a model copying a span out of it would produce one
    that is verbatim in the passage but reads as a fragment in the note.
    Discarding candidates is free — this is selection, not conversion, and the
    corpus offers far more passages than the ~150 queries need.

    `offset` is the character offset of the passage's first paragraph within
    `text`, which is what `_eligible` uses to decide whether a fact is buried.
    """
    passages: list[Passage] = []
    for heading, base, body in _sections(text):
        cursor = base
        buffer: list[str] = []
        start = base
        for paragraph in body.split("\n\n"):
            if _is_prose(paragraph):
                if not buffer:
                    start = cursor
                buffer.append(paragraph.strip())
                joined = "\n\n".join(buffer)
                if len(joined) >= MIN_CHARS:
                    if len(joined) <= MAX_CHARS:
                        passages.append(
                            Passage(
                                note=note,
                                heading=heading,
                                text=joined,
                                offset=start,
                                note_chars=len(text),
                            )
                        )
                    buffer = []
            else:
                buffer = []
            cursor += len(paragraph) + 2  # the "\n\n" that split() consumed
    return passages


def document_frequency(notes: dict[str, str]) -> Counter[str]:
    """How many notes each lowercased word appears in — note count, not
    occurrence count, which is what makes a term *rare in the corpus* rather
    than merely infrequent in one article."""
    df: Counter[str] = Counter()
    for text in notes.values():
        df.update({word.lower() for word in _WORD.findall(text)})
    return df


def rare_terms(
    text: str, df: Counter[str], max_notes: int = RARE_MAX_NOTES
) -> tuple[str, ...]:
    """Tokens of `text` present in at most `max_notes` corpus notes, deduped,
    in first-appearance order.

    A term absent from `df` entirely (count 0) is excluded: that means it came
    from outside the corpus the frequencies were built over, so nothing is
    known about how rare it is there.
    """
    found: list[str] = []
    for word in _WORD.findall(text):
        lowered = word.lower()
        if 0 < df[lowered] <= max_notes and lowered not in found:
            found.append(lowered)
    return tuple(found)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_passages.py -v`
Expected: 11 passed

- [ ] **Step 5: Verify against the real corpus**

Run:
```bash
uv run python -c "
from pathlib import Path
from ariostea.eval.gold_passages import document_frequency, rare_terms, split_passages
from ariostea.eval.wiki_notes import load_corpus_notes
notes = load_corpus_notes(Path('eval/wiki'))
df = document_frequency(notes)
total = 0
for note, text in notes.items():
    passages = split_passages(note, text)
    total += len(passages)
    for p in passages:
        assert p.text in text, note
print(total, 'candidate passages')
print('notes with no candidates:', [n for n, t in notes.items() if not split_passages(n, t)])
rare = sum(1 for note, text in notes.items() for p in split_passages(note, text) if rare_terms(p.text, df))
print(rare, 'passages carry a rare term')
"
```
Expected: several hundred candidate passages, an empty or very short no-candidates list, and a rare-term count comfortably above 40. If fewer than 40 passages carry a rare term, raise `RARE_MAX_NOTES` to 3 and re-run before continuing — Task 3 cannot fill an `exact_term` budget it has no candidates for.

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/gold_passages.py tests/eval/test_gold_passages.py
git commit -m "feat(eval): split corpus notes into candidate passages"
```

---

### Task 3: Type-aware passage selection

**Files:**
- Modify: `src/ariostea/eval/gold_passages.py` (append)
- Test: `tests/eval/test_gold_passages.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ariostea.eval.gold_passages import BURIED_MIN_NOTE_CHARS, select_passages

LONG_PROSE = "The instrument has a hollow wooden body with a carved spruce top. " * 8


def _corpus() -> dict[str, str]:
    filler = "Filler prose that keeps the note long enough to bury a fact. " * 200
    return {
        "strings/violin.md": f"# Violin\n\n{LONG_PROSE}\n\n## History\n\n{filler}\n\n{LONG_PROSE}\n",
        "strings/cello.md": (
            f"# Cello\n\n{LONG_PROSE} The player may bow sul ponticello near the bridge.\n"
        ),
        "strings/violino-it.md": f"# Violino\n\n{LONG_PROSE}\n",
    }


def test_select_passages_respects_the_per_type_budget():
    chosen = select_passages(_corpus(), {"paraphrase": 2})
    assert len(chosen) == 2
    assert {t for t, _ in chosen} == {"paraphrase"}


def test_select_passages_spreads_across_notes_before_repeating_one():
    chosen = select_passages(_corpus(), {"paraphrase": 2})
    assert len({p.note for _, p in chosen}) == 2


def test_select_passages_never_reuses_a_passage_across_types():
    # Two query types over the same fact are not independent samples.
    chosen = select_passages(_corpus(), {"paraphrase": 3, "buried": 1})
    keys = [(p.note, p.offset) for _, p in chosen]
    assert len(keys) == len(set(keys))


def test_buried_selection_requires_a_long_note_and_a_late_passage():
    chosen = select_passages(_corpus(), {"buried": 5})
    for _, passage in chosen:
        assert passage.note_chars >= BURIED_MIN_NOTE_CHARS
        assert passage.offset >= 0.4 * passage.note_chars


def test_exact_term_selection_only_picks_passages_with_a_rare_term():
    chosen = select_passages(_corpus(), {"exact_term": 5})
    assert chosen, "fixture must offer at least one rare-term passage or this proves nothing"
    for _, passage in chosen:
        assert passage.rare_terms


def test_cross_lingual_selection_only_picks_english_notes():
    chosen = select_passages(_corpus(), {"cross_lingual": 5})
    assert chosen
    assert all(not p.note.endswith(("-it.md", "-es.md")) for _, p in chosen)


def test_select_passages_returns_fewer_than_asked_when_the_corpus_runs_out():
    # Silent truncation would read as "the corpus supports 150 queries" when it
    # does not; the runner compares the returned count against the budget.
    chosen = select_passages({"a/x.md": f"# X\n\n{LONG_PROSE}\n"}, {"paraphrase": 99})
    assert 0 < len(chosen) < 99


def test_select_passages_is_deterministic():
    assert select_passages(_corpus(), {"paraphrase": 3}) == select_passages(
        _corpus(), {"paraphrase": 3}
    )


def test_select_passages_rejects_an_unknown_query_type():
    with pytest.raises(ValueError, match="unknown query type"):
        select_passages(_corpus(), {"nonsense": 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_passages.py -v -k select or buried or exact_term or cross_lingual`
Expected: FAIL — `ImportError: cannot import name 'select_passages'`

- [ ] **Step 3: Append the implementation to `src/ariostea/eval/gold_passages.py`**

```python
def _eligible(passage: Passage, query_type: str) -> bool:
    """Whether `passage` can support a query of `query_type`.

    Each branch encodes what the corresponding retrieval track needs:

    - `exact_term` needs a term rare enough that a lexical channel has an
      advantage; without one the model would invent a "rare" word that is
      actually common and the query would test nothing.
    - `buried` needs a long note *and* a passage late in it. Both matter: a
      late passage in a short note is not buried, and an early passage in a
      long one is what the lead paragraph already says.
    - `cross_lingual` needs an English note, because the design is a query in
      Italian or Spanish whose answer span is English. A passage from the
      it/es notes would make the query same-language and test nothing; those
      notes stay in the corpus as the distractors that make the track hard.
    - `paraphrase` needs nothing beyond being prose.
    """
    if query_type == "paraphrase":
        return True
    if query_type == "exact_term":
        return bool(passage.rare_terms)
    if query_type == "buried":
        return (
            passage.note_chars >= BURIED_MIN_NOTE_CHARS
            and passage.offset >= BURIED_MIN_OFFSET * passage.note_chars
        )
    if query_type == "cross_lingual":
        return not passage.note.endswith(("-it.md", "-es.md"))
    raise ValueError(f"unknown query type {query_type!r}")


def select_passages(
    notes: dict[str, str], per_type: dict[str, int]
) -> list[tuple[str, Passage]]:
    """`(query_type, passage)` pairs, at most `per_type[type]` of each.

    Passages are taken round-robin across notes — one from every eligible note,
    then a second from every eligible note, and so on — so a single long
    article cannot supply a whole query type. A passage is used at most once
    across *all* types: two queries about the same sentence are not
    independent samples, and having both a `paraphrase` and an `exact_term`
    case resolve to the same chunk would make the two tracks correlate for
    reasons that have nothing to do with retrieval.

    Returns fewer pairs than asked when the corpus runs out of eligible
    passages. The caller must compare the counts and report the shortfall:
    silently returning 30 `buried` cases for a budget of 40 would read as a
    corpus that supports the requested design when it does not.
    """
    for query_type in per_type:
        _eligible(
            Passage(note="probe/probe.md", heading="", text="", offset=0, note_chars=0),
            query_type,
        )  # raises on an unknown type before any work happens

    frequencies = document_frequency(notes)
    by_note = {
        note: [
            replace(passage, rare_terms=rare_terms(passage.text, frequencies))
            for passage in split_passages(note, text)
        ]
        for note, text in sorted(notes.items())
    }

    chosen: list[tuple[str, Passage]] = []
    used: set[tuple[str, int]] = set()
    for query_type in sorted(per_type):
        budget = per_type[query_type]
        picked = 0
        depth = 0
        while picked < budget:
            any_note_had_a_passage_at_this_depth = False
            for note, passages in by_note.items():
                pool = [p for p in passages if _eligible(p, query_type)]
                if depth >= len(pool):
                    continue
                any_note_had_a_passage_at_this_depth = True
                passage = pool[depth]
                if (note, passage.offset) in used:
                    continue
                used.add((note, passage.offset))
                chosen.append((query_type, passage))
                picked += 1
                if picked == budget:
                    break
            if not any_note_had_a_passage_at_this_depth:
                break  # every note is exhausted for this type
            depth += 1
    return chosen
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_passages.py -v`
Expected: 20 passed

- [ ] **Step 5: Verify the real corpus can fill the planned budget**

Run:
```bash
uv run python -c "
from pathlib import Path
from collections import Counter
from ariostea.eval.gold_passages import select_passages
from ariostea.eval.wiki_notes import load_corpus_notes
notes = load_corpus_notes(Path('eval/wiki'))
budget = {'paraphrase': 40, 'exact_term': 40, 'buried': 40, 'cross_lingual': 30}
chosen = select_passages(notes, budget)
got = Counter(t for t, _ in chosen)
for t, want in sorted(budget.items()):
    print(f'{t:<14} {got[t]:>3} / {want}')
print('distinct notes used:', len({p.note for _, p in chosen}))
"
```
Expected: every type at or near its budget and at least 30 distinct notes used. If `buried` falls well short, lower `BURIED_MIN_NOTE_CHARS` and re-run; record whatever value you settle on in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/gold_passages.py tests/eval/test_gold_passages.py
git commit -m "feat(eval): type-aware deterministic passage selection"
```

---

### Task 4: Prompts and tolerant response parsing

**Files:**
- Create: `src/ariostea/eval/gold_prompts.py`
- Test: `tests/eval/test_gold_prompts.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ariostea.eval.gold_passages import Passage
from ariostea.eval.gold_prompts import (
    GENERATION_SYSTEM,
    JUDGE_SYSTEM,
    generation_user,
    judge_user,
    parse_json_object,
)

PASSAGE = Passage(
    note="strings/violin.md",
    heading="Construction",
    text="The bridge transmits vibration to the body.",
    offset=100,
    note_chars=9000,
    rare_terms=("ponticello", "tasto"),
)


def test_parse_json_object_reads_a_bare_object():
    assert parse_json_object('{"query": "a", "answer_span": "b"}') == {
        "query": "a",
        "answer_span": "b",
    }


def test_parse_json_object_unwraps_a_fenced_block():
    raw = 'Here you go:\n```json\n{"query": "a"}\n```\n'
    assert parse_json_object(raw) == {"query": "a"}


def test_parse_json_object_discards_a_reasoning_block():
    # qwen3-family models emit <think>...</think> before the answer. A brace
    # inside that block would otherwise be read as the start of the object.
    raw = '<think>I should answer {maybe} like this</think>\n{"query": "a"}'
    assert parse_json_object(raw) == {"query": "a"}


def test_parse_json_object_reads_an_object_surrounded_by_prose():
    assert parse_json_object('Sure. {"query": "a"} Hope that helps!') == {"query": "a"}


def test_parse_json_object_raises_when_there_is_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_object("I cannot help with that.")


def test_parse_json_object_raises_on_malformed_json():
    with pytest.raises(ValueError):
        parse_json_object('{"query": }')


def test_parse_json_object_raises_when_the_object_is_not_a_mapping():
    # `[{"query": "a"}]` finds a brace and parses, but indexing it by key
    # would raise TypeError deep inside the caller instead of here.
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_json_object('["a", "b"]')


def test_generation_user_includes_the_passage_and_the_section():
    prompt = generation_user(PASSAGE, "paraphrase", title="Violin")
    assert PASSAGE.text in prompt
    assert "Construction" in prompt
    assert "Violin" in prompt


def test_exact_term_prompt_names_the_rare_terms():
    prompt = generation_user(PASSAGE, "exact_term", title="Violin")
    assert "ponticello" in prompt


def test_cross_lingual_prompt_names_the_target_language():
    prompt = generation_user(PASSAGE, "cross_lingual", title="Violin", lang_name="Italian")
    assert "Italian" in prompt


def test_generation_user_rejects_an_unknown_type():
    with pytest.raises(KeyError):
        generation_user(PASSAGE, "nonsense", title="Violin")


def test_judge_user_shows_the_query_and_span_but_not_the_passage():
    # The judge must decide whether the span ALONE answers the query. Showing
    # it the passage lets it answer from context the retrieval system will
    # never have, which is exactly the failure this gate exists to catch.
    prompt = judge_user(query="how is it tuned", span="tuned in fifths", title="Violin")
    assert "how is it tuned" in prompt
    assert "tuned in fifths" in prompt
    assert PASSAGE.text not in prompt


def test_system_prompts_demand_json_only():
    assert "JSON" in GENERATION_SYSTEM
    assert "JSON" in JUDGE_SYSTEM
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.gold_prompts'`

- [ ] **Step 3: Write the implementation**

```python
"""The generation and judge prompts, and the reader that survives what local
instruct models actually return.

`parse_json_object` is deliberately forgiving. `OpenAICompatChat` has no
`response_format` support, the endpoints this runs against are local models
rather than a hosted API with strict JSON mode, and every unparsed response is
a candidate silently lost from a ~150-query budget. Being strict here would
buy nothing: a malformed response is not more correct for being rejected on a
technicality, and the *content* gates come later.
"""

from __future__ import annotations

import json
import re

from ariostea.eval.gold_passages import Passage

# qwen3-family models emit their reasoning in a <think> block before the
# answer. Stripped before anything else, because that block routinely contains
# braces and would otherwise be mistaken for the JSON object.
_THINK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(raw: str) -> dict:
    """Extract the single JSON object from a chat response.

    Handles, in order: a reasoning block, a ``` fence, and surrounding prose
    (by taking the span between the first `{` and the last `}`). Raises
    `ValueError` — which `json.JSONDecodeError` already subclasses — so a
    caller has one exception type to catch for every unusable response.
    """
    text = _THINK.sub("", raw).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"response is not a JSON object: {raw[:200]!r}")
    return parsed


GENERATION_SYSTEM = (
    "You write evaluation queries for a document retrieval system. You are given "
    "one passage from a Wikipedia article. You reply with a single JSON object and "
    "nothing else, with exactly these keys:\n"
    '  "query": the question a user would type into a search box\n'
    '  "answer_span": text copied VERBATIM and CONTIGUOUSLY from the passage, '
    "between 10 and 200 characters, that answers the query\n"
    "\n"
    "Rules that override every other instruction:\n"
    "1. answer_span must be an exact substring of the passage. Copy it character by "
    "character. Do not paraphrase it, do not correct its punctuation or spelling, "
    "and do not stitch together two non-adjacent pieces of text.\n"
    "2. The query must be answerable only from the passage. Someone who knows just "
    "the article title must not be able to answer it.\n"
    "3. The query must not quote the article title verbatim.\n"
    "4. Ask about one fact, with one correct answer.\n"
)

_TYPE_INSTRUCTIONS = {
    "paraphrase": (
        "Write the query as a restatement that shares as few words with the passage "
        "as you can manage. Use synonyms throughout. A keyword search for your query "
        "should struggle to find this passage; only a meaning-based match should."
    ),
    "exact_term": (
        "Build the query around one of these rare technical terms, spelled exactly as "
        "given: {rare}. The query must hinge on that term — someone who does not know "
        "the term cannot answer the query."
    ),
    "buried": (
        "This passage sits deep inside a long article, far from what its title "
        "suggests. Ask about the specific fact this passage states, not about the "
        "article's main subject, and do not mention the article title."
    ),
    "cross_lingual": (
        "Write the query in {lang_name}, not in English, even though the passage is in "
        "English. Use natural, fluent {lang_name} of the kind a native speaker would "
        "type. The answer_span stays in English, copied verbatim from the passage."
    ),
}


def generation_user(
    passage: Passage, query_type: str, title: str, lang_name: str = "Italian"
) -> str:
    """The per-passage user turn. Raises `KeyError` on an unknown query type —
    a typo must not silently degrade to a generic prompt that produces
    a case labelled `exact_term` which stresses nothing in particular."""
    instruction = _TYPE_INSTRUCTIONS[query_type].format(
        rare=", ".join(passage.rare_terms[:5]), lang_name=lang_name
    )
    return (
        f"Article title: {title}\n"
        f"Section: {passage.heading or '(introduction)'}\n"
        f"\n"
        f"Passage:\n{passage.text}\n"
        f"\n"
        f"Task: {instruction}\n"
        f"Reply with the JSON object only."
    )


JUDGE_SYSTEM = (
    "You audit evaluation data for a retrieval benchmark. You are strict: when in "
    "doubt, reject. You reply with a single JSON object and nothing else, with "
    "exactly these keys:\n"
    '  "answers": true only if the answer span, read on its own with no other '
    "context, answers the query\n"
    '  "unambiguous": true only if the query has one clear reading and one correct '
    "answer\n"
    '  "title_only": true if someone who knows only the article title, and has not '
    "read the span, could already answer the query\n"
    '  "reason": one short sentence explaining your judgement\n'
)


def judge_user(query: str, span: str, title: str) -> str:
    """The judge's user turn.

    It shows the query, the span and the article title — and deliberately not
    the passage the span came from. The question this gate answers is whether
    the *span alone* answers the query, because a retrieved chunk is all the
    span-level metric will ever credit. A judge that could read the
    surrounding passage would approve spans that only make sense in context,
    which is the exact failure this stage exists to catch.
    """
    return (
        f"Article title: {title}\n"
        f"Query: {query}\n"
        f"Answer span: {span}\n"
        f"\n"
        f"Judge the query and span against your four keys. Reply with the JSON "
        f"object only."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_prompts.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/gold_prompts.py tests/eval/test_gold_prompts.py
git commit -m "feat(eval): gold generation prompts and tolerant JSON parsing"
```

---

### Task 5: Generation over the chat port

**Files:**
- Create: `src/ariostea/eval/gold_generate.py`
- Test: `tests/eval/test_gold_generate.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ariostea.eval.gold_generate import Candidate, generate_case
from ariostea.eval.gold_passages import Passage

PASSAGE = Passage(
    note="strings/violin.md",
    heading="Tuning",
    text="The violin is tuned in perfect fifths: G, D, A, E.",
    offset=100,
    note_chars=9000,
    rare_terms=("ponticello",),
)


class FakeChat:
    """A ChatProvider that returns canned responses and records its prompts."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def test_generate_case_builds_a_candidate_from_the_response():
    chat = FakeChat('{"query": "how is a violin tuned", "answer_span": "perfect fifths"}')
    candidate = generate_case(chat, PASSAGE, "paraphrase", title="Violin")
    assert candidate == Candidate(
        query="how is a violin tuned",
        query_lang="en",
        type="paraphrase",
        note="strings/violin.md",
        passage=PASSAGE.text,
        span="perfect fifths",
    )


def test_generate_case_marks_cross_lingual_cases_with_the_query_language():
    chat = FakeChat('{"query": "come si accorda", "answer_span": "perfect fifths"}')
    candidate = generate_case(
        chat, PASSAGE, "cross_lingual", title="Violin", lang_name="Italian", query_lang="it"
    )
    assert candidate.query_lang == "it"


def test_generate_case_strips_surrounding_whitespace():
    chat = FakeChat('{"query": "  q  ", "answer_span": "  perfect fifths  "}')
    assert generate_case(chat, PASSAGE, "paraphrase", title="Violin").span == "perfect fifths"


def test_generate_case_raises_on_an_empty_query():
    chat = FakeChat('{"query": "", "answer_span": "perfect fifths"}')
    with pytest.raises(ValueError, match="empty query"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_on_a_missing_span():
    chat = FakeChat('{"query": "how is a violin tuned"}')
    with pytest.raises(ValueError, match="empty answer_span"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_on_an_unparseable_response():
    chat = FakeChat("I am sorry, I cannot do that.")
    with pytest.raises(ValueError, match="no JSON object"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_when_the_span_is_a_list():
    # A model that returns ["a", "b"] would otherwise stringify to "['a', 'b']"
    # and fail the verbatim check later with a confusing reason.
    chat = FakeChat('{"query": "q", "answer_span": ["a", "b"]}')
    with pytest.raises(ValueError, match="answer_span must be a string"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.gold_generate'`

- [ ] **Step 3: Write the implementation**

```python
"""One generation call: a passage and a query type in, a `Candidate` out.

Deliberately thin. Everything it could get wrong — is the span really in the
note, is the query answerable from the title, does the span actually answer
the query — belongs to the validation gates, not here. This module's only job
is to turn a chat response into a typed object or a `ValueError` the caller
can record as a rejection, so no unusable response ever reaches a gate as if
it were data.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.gold_passages import Passage
from ariostea.eval.gold_prompts import GENERATION_SYSTEM, generation_user, parse_json_object
from ariostea.ports.chat import ChatProvider


@dataclass(frozen=True)
class Candidate:
    """A generated query before any gate has looked at it.

    `passage` is carried alongside `note` because the first automatic check is
    that the span is verbatim in the passage *the model was shown*, which is a
    sharper signal than "verbatim somewhere in the note": a span that is in
    the note but not in the passage means the model reproduced text from
    memory rather than copying, and the next one may be a plausible
    fabrication instead.
    """

    query: str
    query_lang: str
    type: str
    note: str
    passage: str
    span: str


def _text_field(data: dict, key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {type(value).__name__}")
    return value.strip()


def generate_case(
    chat: ChatProvider,
    passage: Passage,
    query_type: str,
    title: str,
    lang_name: str = "Italian",
    query_lang: str = "en",
) -> Candidate:
    """Ask `chat` for one query anchored to `passage`.

    Raises `ValueError` for any response that cannot become a candidate. The
    caller catches it and records a stage-0 rejection; a generation run of
    ~150 passages against a local model will always produce a handful.
    """
    raw = chat.complete(
        system=GENERATION_SYSTEM,
        user=generation_user(passage, query_type, title=title, lang_name=lang_name),
    )
    data = parse_json_object(raw)
    query = _text_field(data, "query")
    span = _text_field(data, "answer_span")
    if not query:
        raise ValueError("empty query")
    if not span:
        raise ValueError("empty answer_span")
    return Candidate(
        query=query,
        query_lang=query_lang,
        type=query_type,
        note=passage.note,
        passage=passage.text,
        span=span,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_generate.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/gold_generate.py tests/eval/test_gold_generate.py
git commit -m "feat(eval): generate one gold candidate over the chat port"
```

---

### Task 6: Automatic validation gate (stage 1)

**Files:**
- Create: `src/ariostea/eval/gold_validate.py`
- Test: `tests/eval/test_gold_validate.py`

- [ ] **Step 1: Write the failing tests**

```python
from dataclasses import replace

from ariostea.eval.gold_generate import Candidate
from ariostea.eval.gold_validate import automatic_gate

NOTES = {
    "strings/violin.md": (
        "# Violin\n\nThe violin is tuned in perfect fifths: G, D, A, E. "
        "Players sometimes bow sul ponticello, near the bridge.\n"
    )
}
TITLES = {"strings/violin.md": "Violin"}

GOOD = Candidate(
    query="what note does the lowest string sound",
    query_lang="en",
    type="paraphrase",
    note="strings/violin.md",
    passage="The violin is tuned in perfect fifths: G, D, A, E.",
    span="tuned in perfect fifths: G, D, A, E",
)


def test_a_well_formed_candidate_passes():
    assert automatic_gate(GOOD, NOTES, TITLES) is None


def test_a_span_absent_from_the_passage_is_rejected():
    bad = replace(GOOD, span="tuned in perfect fourths")
    assert "verbatim" in automatic_gate(bad, NOTES, TITLES)


def test_a_span_present_in_the_passage_but_not_the_note_is_rejected():
    # Catches a passage that was hand-built or drifted from the corpus.
    bad = replace(GOOD, passage="Invented text about the bridge.", span="Invented text")
    assert "cited note" in automatic_gate(bad, NOTES, TITLES)


def test_matching_is_whitespace_and_case_insensitive():
    ok = replace(GOOD, span="Tuned  in\nperfect fifths: g, d, a, e")
    assert automatic_gate(ok, NOTES, TITLES) is None


def test_a_span_shorter_than_the_minimum_is_rejected():
    bad = replace(GOOD, span="G, D")
    assert "shorter" in automatic_gate(bad, NOTES, TITLES)


def test_a_span_longer_than_the_maximum_is_rejected():
    long_note = {"a/x.md": "# X\n\n" + "word " * 200}
    bad = Candidate(
        query="q about many words",
        query_lang="en",
        type="paraphrase",
        note="a/x.md",
        passage="word " * 200,
        span="word " * 100,
    )
    assert "longer" in automatic_gate(bad, long_note, {"a/x.md": "X"})


def test_a_query_that_restates_the_title_is_rejected():
    bad = replace(GOOD, query="violin")
    assert "restates the article title" in automatic_gate(bad, NOTES, TITLES)


def test_a_span_adding_nothing_beyond_the_title_is_rejected():
    notes = {"a/x.md": "# Violin family\n\nThe violin family violin family violin.\n"}
    bad = Candidate(
        query="which family of instruments is discussed here",
        query_lang="en",
        type="paraphrase",
        note="a/x.md",
        passage="The violin family violin family violin.",
        span="violin family violin",
    )
    assert "beyond the article title" in automatic_gate(bad, notes, {"a/x.md": "Violin family"})


def test_a_note_outside_the_corpus_is_rejected():
    bad = replace(GOOD, note="strings/ghost.md")
    assert "not in corpus" in automatic_gate(bad, NOTES, TITLES)


def test_a_cross_lingual_query_written_in_english_is_rejected():
    bad = replace(GOOD, type="cross_lingual", query_lang="it", query="tuned in perfect fifths")
    assert "not in another language" in automatic_gate(bad, NOTES, TITLES)


def test_a_genuine_italian_cross_lingual_query_passes():
    ok = replace(
        GOOD, type="cross_lingual", query_lang="it", query="come si accorda uno strumento"
    )
    assert automatic_gate(ok, NOTES, TITLES) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.gold_validate'`

- [ ] **Step 3: Write the implementation**

```python
"""The validation gates a generated candidate must survive.

Stage 1 (`automatic_gate`) is mechanical and free: it checks what can be
decided by comparing strings. Stage 2 (`adversarial_gate`, Task 7) costs a
model call and only ever sees candidates stage 1 already accepted, because
there is no point paying a judge to read a span that is not even in the note.

Every check returns a *reason string* rather than raising or returning a
bool. The reasons are written to `eval/wiki/gold_rejected.json`, and a gate
whose rejections cannot be read is a gate nobody can tune: the first real run
is expected to reject a large fraction, and the only way to tell "the model is
bad at this" from "my threshold is wrong" is to read why.
"""

from __future__ import annotations

import re

from ariostea.eval.gold_generate import Candidate
from ariostea.eval.normalize import normalize_ws

MIN_SPAN_CHARS = 10
MAX_SPAN_CHARS = 300
# A query sharing this fraction of its content words with the article title is
# a restatement of the title, answerable by any chunk of the note.
TITLE_OVERLAP = 0.8

_CONTENT_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _content_tokens(text: str) -> set[str]:
    return {word.lower() for word in _CONTENT_WORD.findall(text)}


def automatic_gate(
    candidate: Candidate, notes: dict[str, str], titles: dict[str, str]
) -> str | None:
    """Return why `candidate` is unusable, or `None` if it survives stage 1.

    Checks are ordered cheapest-and-most-fundamental first, so the reason a
    candidate is rejected is the most informative one available: a span that
    is neither in the note nor long enough should be reported as fabricated,
    not as short.
    """
    if candidate.note not in notes:
        return f"note {candidate.note!r} not in corpus"

    span = normalize_ws(candidate.span)
    if span not in normalize_ws(candidate.passage):
        return "span is not verbatim in the passage the model was shown"
    if span not in normalize_ws(notes[candidate.note]):
        return "span is not verbatim in the cited note"
    if len(candidate.span) < MIN_SPAN_CHARS:
        return f"span is shorter than {MIN_SPAN_CHARS} characters"
    if len(candidate.span) > MAX_SPAN_CHARS:
        return f"span is longer than {MAX_SPAN_CHARS} characters"

    title_tokens = _content_tokens(titles[candidate.note])
    query_tokens = _content_tokens(candidate.query)
    if not query_tokens:
        return "query has no content words"
    if len(query_tokens & title_tokens) / len(query_tokens) >= TITLE_OVERLAP:
        return "query restates the article title"

    span_tokens = _content_tokens(candidate.span)
    if span_tokens and span_tokens <= title_tokens:
        return "span adds nothing beyond the article title"

    if candidate.type == "cross_lingual" and query_tokens <= _content_tokens(candidate.passage):
        # Every content word of the query already appears in the English
        # passage, so the query was written in English despite the
        # instruction. A real Italian or Spanish query always contributes
        # words the English text does not have.
        return "cross_lingual query is not in another language"

    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_validate.py -v`
Expected: 11 passed

- [ ] **Step 5: Verify the gate is not deletable — mutation check**

Comment out the `query restates the article title` branch, run the tests, confirm at least one fails, then restore it. Repeat for the `span adds nothing beyond the article title` branch and the `cross_lingual` branch. A branch no test notices the loss of is a branch that is not really guarding anything.

Run: `uv run pytest tests/eval/test_gold_validate.py -q`
Expected after each restore: 11 passed

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/gold_validate.py tests/eval/test_gold_validate.py
git commit -m "feat(eval): automatic validation gate for generated gold"
```

---

### Task 7: Adversarial judge gate (stage 2)

**Files:**
- Modify: `src/ariostea/eval/gold_validate.py` (append)
- Test: `tests/eval/test_gold_validate.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from ariostea.eval.gold_validate import adversarial_gate


class FakeJudge:
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._response


def test_a_candidate_the_judge_approves_passes():
    judge = FakeJudge(
        '{"answers": true, "unambiguous": true, "title_only": false, "reason": "fine"}'
    )
    assert adversarial_gate(judge, GOOD, title="Violin") is None


def test_a_span_the_judge_says_does_not_answer_is_rejected():
    judge = FakeJudge(
        '{"answers": false, "unambiguous": true, "title_only": false, "reason": "off topic"}'
    )
    reason = adversarial_gate(judge, GOOD, title="Violin")
    assert "does not answer" in reason and "off topic" in reason


def test_an_ambiguous_query_is_rejected():
    judge = FakeJudge(
        '{"answers": true, "unambiguous": false, "title_only": false, "reason": "two readings"}'
    )
    assert "ambiguous" in adversarial_gate(judge, GOOD, title="Violin")


def test_a_title_answerable_query_is_rejected():
    judge = FakeJudge(
        '{"answers": true, "unambiguous": true, "title_only": true, "reason": "title says it"}'
    )
    assert "title alone" in adversarial_gate(judge, GOOD, title="Violin")


def test_a_missing_verdict_key_is_rejected_rather_than_treated_as_approval():
    # `data.get("answers")` on a truncated response returns None, which is
    # falsy — approval must never be the default for a response we could not
    # fully read.
    judge = FakeJudge('{"reason": "truncated"}')
    assert adversarial_gate(judge, GOOD, title="Violin") is not None


def test_an_unparseable_judge_response_is_a_rejection_not_a_crash():
    judge = FakeJudge("I think it is fine, honestly.")
    assert "unreadable" in adversarial_gate(judge, GOOD, title="Violin")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_validate.py -v -k judge or adversarial or verdict`
Expected: FAIL — `ImportError: cannot import name 'adversarial_gate'`

- [ ] **Step 3: Append the implementation to `src/ariostea/eval/gold_validate.py`**

```python
def adversarial_gate(judge: ChatProvider, candidate: Candidate, title: str) -> str | None:
    """Return why a second model rejects `candidate`, or `None` if it approves.

    `judge` must be a *different model* from the one that generated the
    candidate. A model asked to check its own output agrees with itself: the
    point of this stage is an independent reading, and pointing both at the
    same model turns a gate into a rubber stamp.

    Every failure to read a verdict is a rejection. An unparseable response,
    a missing key, a truncated object — none of them are evidence the
    candidate is good, and defaulting to approval would let exactly the
    responses the judge struggled with through the gate.
    """
    raw = judge.complete(
        system=JUDGE_SYSTEM, user=judge_user(candidate.query, candidate.span, title)
    )
    try:
        verdict = parse_json_object(raw)
    except ValueError as exc:
        return f"judge verdict unreadable: {exc}"

    reason = str(verdict.get("reason", "")).strip()
    if not verdict.get("answers"):
        return f"judge: span does not answer the query ({reason})"
    if not verdict.get("unambiguous"):
        return f"judge: query is ambiguous ({reason})"
    if verdict.get("title_only"):
        return f"judge: answerable from the title alone ({reason})"
    return None
```

Add to the existing imports at the top of the file:

```python
from ariostea.eval.gold_prompts import JUDGE_SYSTEM, judge_user, parse_json_object
from ariostea.ports.chat import ChatProvider
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_validate.py -v`
Expected: 17 passed

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/gold_validate.py tests/eval/test_gold_validate.py
git commit -m "feat(eval): adversarial second-model gate for generated gold"
```

---

### Task 8: Shared wiki index and channels

**Files:**
- Create: `src/ariostea/eval/wiki_index.py`
- Test: `tests/eval/test_wiki_index.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

import pytest

from ariostea.eval.wiki_index import CHUNK_POOL, MULTILINGUAL_MODEL, wiki_config

CORPUS = Path("eval/wiki")


def test_wiki_config_points_at_the_corpus_and_the_given_database(tmp_path):
    db = str(tmp_path / "eval.db")
    config = wiki_config(CORPUS, db)
    assert config.vault.path == str(CORPUS)
    assert config.store.path == db


def test_wiki_config_ignores_nothing_so_every_note_is_indexed():
    # The default vault ignore list skips `.obsidian/`; the eval corpus has no
    # such directory and every file in it is a note under test.
    assert wiki_config(CORPUS, "x.db").ignore == []


def test_wiki_config_uses_the_multilingual_embedding_model():
    # An English-only model would score the it/es notes near zero and make the
    # cross_lingual track measure the model choice rather than the pipeline.
    assert wiki_config(CORPUS, "x.db").embedding.local_model == MULTILINGUAL_MODEL


def test_wiki_config_disables_contextualization_by_default():
    # The baseline index must not silently depend on a running chat endpoint.
    assert wiki_config(CORPUS, "x.db").contextual.enabled is False


@pytest.mark.integration
def test_index_and_channels_retrieve_from_the_real_corpus(tmp_path):
    from ariostea.eval.wiki_index import index_wiki_corpus, wiki_channels

    db = str(tmp_path / "eval.db")
    container = index_wiki_corpus(CORPUS, db)
    assert container.admin.stats().notes == 79

    channels = wiki_channels(db, container)
    assert set(channels) == {"DENSE", "SPARSE", "HYBRID"}
    for name, fn in channels.items():
        hits = fn("how is a violin tuned", CHUNK_POOL)
        assert hits, name
        note, text = hits[0]
        assert note.endswith(".md") and text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_wiki_index.py -v -m "not integration"`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.wiki_index'`

- [ ] **Step 3: Write the implementation**

```python
"""Index the committed Wikipedia corpus into a throwaway database and expose
the three retrieval channels over it.

Shared by both runners rather than duplicated: `generate_gold.py` needs
channels for the discrimination filter, `run_wiki_eval.py` needs them to
report. If the two built their indexes differently — a different embedding
model, contextualization on in one and off in the other — the filter would
drop cases as "too easy" for a pipeline the evaluation never actually runs.
"""

from __future__ import annotations

from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import Container, build_container
from ariostea.config.schema import Config, ContextualCfg, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import (
    make_dense_chunk_fn,
    make_hybrid_chunk_fn,
    make_sparse_chunk_fn,
)
from ariostea.eval.harness import SpanSearchFn
from ariostea.mcp.handlers import reindex_payload

# The corpus is deliberately multilingual; an English-only embedding model
# would make the cross_lingual track measure the model rather than the
# pipeline. Same model the existing eval runners use, so numbers compare.
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_POOL = 50


def wiki_config(corpus: Path, db: str, contextual: ContextualCfg | None = None) -> Config:
    """Config for indexing `corpus` into `db`.

    `ignore=[]` overrides the default `.obsidian/` skip: the eval corpus has
    no such directory, and every file in it is a note under test.
    """
    return Config(
        vault=VaultCfg(path=str(corpus), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=db),
        contextual=contextual or ContextualCfg(enabled=False),
    )


def index_wiki_corpus(
    corpus: Path, db: str, contextual: ContextualCfg | None = None
) -> Container:
    """Build the index and return the container that owns it."""
    container = build_container(wiki_config(corpus, db, contextual))
    reindex_payload(container)
    return container


def wiki_channels(db: str, container: Container) -> dict[str, SpanSearchFn]:
    """The three chunk-level channels over an already-indexed `db`.

    Opens a second store handle rather than reaching into the container: the
    `Container` deliberately exposes ports and use cases, never the concrete
    `SqliteStore`, and the raw dense/sparse channels need the adapter. A
    second read-only handle over the same file is the cheapest way to keep
    that boundary intact — the same trick `run_eval.py` already uses.
    """
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
    store = SqliteStore(path=db, dim=embeddings.dimension)
    return {
        "DENSE": make_dense_chunk_fn(embeddings, store, CHUNK_POOL),
        "SPARSE": make_sparse_chunk_fn(store, CHUNK_POOL),
        "HYBRID": make_hybrid_chunk_fn(container, CHUNK_POOL),
    }
```

- [ ] **Step 4: Run the fast tests**

Run: `uv run pytest tests/eval/test_wiki_index.py -v -m "not integration"`
Expected: 4 passed, 1 deselected

- [ ] **Step 5: Run the integration test (downloads the model on first run, then indexes 79 notes — expect several minutes)**

Run: `uv run pytest tests/eval/test_wiki_index.py -v -m integration`
Expected: 1 passed

- [ ] **Step 6: Commit**

```bash
git add src/ariostea/eval/wiki_index.py tests/eval/test_wiki_index.py
git commit -m "feat(eval): shared index and channels over the wiki corpus"
```

---

### Task 9: Discrimination filter (stage 3)

**Files:**
- Create: `src/ariostea/eval/gold_discriminate.py`
- Test: `tests/eval/test_gold_discriminate.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ariostea.eval.gold_discriminate import discrimination_filter
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase

CASE = WikiGoldCase(
    query="how is a violin tuned",
    query_lang="en",
    type="paraphrase",
    scenario="paraphrase",
    expected_notes=("strings/violin.md",),
    answer_spans=(AnswerSpan(note="strings/violin.md", text="perfect fifths"),),
)


def _hit(query, k):
    return [("strings/violin.md", "It is tuned in perfect fifths.")]


def _miss(query, k):
    return [("strings/cello.md", "Something else entirely.")]


def test_a_case_every_channel_answers_at_rank_1_is_dropped():
    kept, dropped = discrimination_filter([CASE], {"DENSE": _hit, "SPARSE": _hit})
    assert kept == [] and dropped == [CASE]


def test_a_case_one_channel_misses_is_kept():
    kept, dropped = discrimination_filter([CASE], {"DENSE": _hit, "SPARSE": _miss})
    assert kept == [CASE] and dropped == []


def test_a_case_every_channel_misses_is_kept():
    # Hard is not the same as useless: a case nothing answers is exactly the
    # kind an improvement should later be able to move.
    kept, _ = discrimination_filter([CASE], {"DENSE": _miss, "SPARSE": _miss})
    assert kept == [CASE]


def test_rank_2_does_not_count_as_answered():
    def rank_two(query, k):
        return [("strings/cello.md", "Wrong."), ("strings/violin.md", "perfect fifths")]

    kept, _ = discrimination_filter([CASE], {"DENSE": rank_two, "SPARSE": _hit})
    assert kept == [CASE]


def test_an_empty_channel_map_raises():
    # `all()` over no channels is vacuously true, which would silently drop
    # every case as "too easy" and produce an empty gold file.
    with pytest.raises(ValueError, match="at least one channel"):
        discrimination_filter([CASE], {})


def test_each_channel_is_asked_for_exactly_one_result_per_case():
    seen: list[int] = []

    def recording(query, k):
        seen.append(k)
        return _hit(query, k)

    discrimination_filter([CASE], {"DENSE": recording})
    assert seen == [1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_gold_discriminate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.gold_discriminate'`

- [ ] **Step 3: Write the implementation**

```python
"""Validation stage 3: drop the queries that carry no experimental signal.

A query every retrieval channel already answers at rank 1 cannot distinguish
two methods — it will read 1.0 before a change and 1.0 after it, and its only
effect on the eval is to raise every number and shrink every visible
difference. That is precisely the flaw the old corpus had, and it is worth
spending model calls to generate cases that then get thrown away here rather
than shipping a gold set that scores well and measures nothing.

Note what is *not* dropped: a query no channel answers. Hard is not useless —
an unanswered case is the one an improvement can actually move.
"""

from __future__ import annotations

from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.span_metrics import span_recall_at_k
from ariostea.eval.wiki_gold import WikiGoldCase


def discrimination_filter(
    cases: list[WikiGoldCase], channels: dict[str, SpanSearchFn]
) -> tuple[list[WikiGoldCase], list[WikiGoldCase]]:
    """Split `cases` into `(kept, dropped)`.

    Raises on an empty `channels` map: `all()` over an empty sequence is
    vacuously true, so every case would be classified "answered by every
    channel" and the run would write an empty gold file while reporting
    success.
    """
    if not channels:
        raise ValueError("discrimination_filter needs at least one channel to judge against")

    kept: list[WikiGoldCase] = []
    dropped: list[WikiGoldCase] = []
    for case in cases:
        answered_by_all = all(
            span_recall_at_k(case.answer_spans, search_fn(case.query, 1), 1) == 1.0
            for search_fn in channels.values()
        )
        (dropped if answered_by_all else kept).append(case)
    return kept, dropped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_gold_discriminate.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/gold_discriminate.py tests/eval/test_gold_discriminate.py
git commit -m "feat(eval): discrimination filter drops signal-free gold cases"
```

---

### Task 10: nDCG in the harness

The design doc's harness section asks for "recall@k / MRR / nDCG at both note-level and span-level". Recall and MRR exist; nDCG does not.

**Files:**
- Modify: `src/ariostea/eval/metrics.py`
- Modify: `src/ariostea/eval/span_metrics.py`
- Modify: `src/ariostea/eval/spaneval.py`
- Test: `tests/eval/test_metrics.py`, `tests/eval/test_span_metrics.py`, `tests/eval/test_spaneval.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/eval/test_metrics.py`:

```python
import math

from ariostea.eval.metrics import ndcg_at_k


def test_ndcg_is_one_at_rank_1():
    assert ndcg_at_k({"a.md"}, ["a.md", "b.md"], k=5) == 1.0


def test_ndcg_discounts_logarithmically():
    assert ndcg_at_k({"b.md"}, ["a.md", "b.md"], k=5) == 1.0 / math.log2(3)


def test_ndcg_is_zero_when_the_hit_falls_outside_k():
    assert ndcg_at_k({"b.md"}, ["a.md", "b.md"], k=1) == 0.0


def test_ndcg_is_zero_with_no_hit():
    assert ndcg_at_k({"z.md"}, ["a.md", "b.md"], k=5) == 0.0
```

Append to `tests/eval/test_span_metrics.py`:

```python
import math

from ariostea.eval.span_metrics import span_ndcg_at_k
from ariostea.eval.wiki_gold import AnswerSpan

SPANS = (AnswerSpan(note="a.md", text="perfect fifths"),)


def test_span_ndcg_is_one_when_the_first_chunk_contains_the_span():
    assert span_ndcg_at_k(SPANS, [("a.md", "tuned in perfect fifths")], k=5) == 1.0


def test_span_ndcg_discounts_a_later_chunk():
    retrieved = [("a.md", "unrelated"), ("a.md", "tuned in perfect fifths")]
    assert span_ndcg_at_k(SPANS, retrieved, k=5) == 1.0 / math.log2(3)


def test_span_ndcg_ignores_a_containing_chunk_from_the_wrong_note():
    assert span_ndcg_at_k(SPANS, [("b.md", "tuned in perfect fifths")], k=5) == 0.0
```

Append to `tests/eval/test_spaneval.py`:

```python
def test_report_carries_ndcg_at_both_granularities():
    cases = [
        WikiGoldCase(
            query="q",
            query_lang="en",
            type="paraphrase",
            scenario="paraphrase",
            expected_notes=("a.md",),
            answer_spans=(AnswerSpan(note="a.md", text="perfect fifths"),),
        )
    ]

    def span_fn(query, k):
        return [("a.md", "tuned in perfect fifths")]

    report = evaluate_spans(cases, span_fn, k=5)
    assert report.overall.note_ndcg_at_k == 1.0
    assert report.overall.span_ndcg_at_k == 1.0
    assert "ndcg" in format_span_report(report)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_metrics.py tests/eval/test_span_metrics.py tests/eval/test_spaneval.py -v`
Expected: FAIL — `ImportError: cannot import name 'ndcg_at_k'`

- [ ] **Step 3: Add `ndcg_at_k` to `src/ariostea/eval/metrics.py`**

Add `import math` at the top of the file, then:

```python
def ndcg_at_k(expected: set[str], ranked: list[str], k: int) -> float:
    """Binary-relevance nDCG@k: `1/log2(rank + 1)` at the first hit inside the
    top k, else 0.0.

    With one relevant note per query — this gold set's shape, same as the rest
    of this module assumes — the ideal DCG is exactly 1.0, so nDCG reduces to
    the discounted gain of the first hit and needs no normalization term.

    Reported because the design doc asks for it, and worth having for its
    gentler discount: MRR drops from 0.33 to 0.20 between rank 3 and rank 5
    while nDCG only drops from 0.50 to 0.39, so mid-list movement stays
    visible. Under one-answer-per-query gold it is a monotone function of the
    same rank MRR reads, so it adds resolution, not independent information.
    """
    for index, path in enumerate(ranked[:k]):
        if path in expected:
            return 1.0 / math.log2(index + 2)
    return 0.0
```

- [ ] **Step 4: Add `span_ndcg_at_k` to `src/ariostea/eval/span_metrics.py`**

Add `import math` at the top of the file, then:

```python
def span_ndcg_at_k(
    spans: tuple[AnswerSpan, ...], retrieved: list[tuple[str, str]], k: int
) -> float:
    """Binary-relevance nDCG@k over chunks, using the same containment rule as
    `span_recall_at_k`: `1/log2(rank + 1)` at the first top-k chunk that is in
    an answer span's own note and contains that span, else 0.0."""
    for index, (note, text) in enumerate(retrieved[:k]):
        if _is_hit(spans, note, text):
            return 1.0 / math.log2(index + 2)
    return 0.0
```

- [ ] **Step 5: Wire both into `src/ariostea/eval/spaneval.py`**

Extend the imports:

```python
from ariostea.eval.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from ariostea.eval.span_metrics import span_ndcg_at_k, span_recall_at_k, span_reciprocal_rank
```

Extend `SpanScore` with two fields:

```python
@dataclass(frozen=True)
class SpanScore:
    type: str
    n: int
    note_recall_at_k: float
    note_mrr: float
    note_ndcg_at_k: float
    span_recall_at_k: float
    span_mrr: float
    span_ndcg_at_k: float
```

Each scored row becomes a 6-tuple `(note_recall, note_mrr, note_ndcg, span_recall, span_mrr, span_ndcg)`. Update `_aggregate`:

```python
# Each scored row is (note_recall, note_mrr, note_ndcg, span_recall, span_mrr, span_ndcg).
def _aggregate(type_: str, rows: list[tuple[float, ...]]) -> SpanScore:
    n = len(rows)
    if n == 0:
        return SpanScore(type_, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return SpanScore(
        type=type_,
        n=n,
        note_recall_at_k=sum(r[0] for r in rows) / n,
        note_mrr=sum(r[1] for r in rows) / n,
        note_ndcg_at_k=sum(r[2] for r in rows) / n,
        span_recall_at_k=sum(r[3] for r in rows) / n,
        span_mrr=sum(r[4] for r in rows) / n,
        span_ndcg_at_k=sum(r[5] for r in rows) / n,
    )
```

Update the row built inside `evaluate_spans`:

```python
        row = (
            recall_at_k(expected, notes, k),
            reciprocal_rank(expected, notes),
            ndcg_at_k(expected, notes, k),
            span_recall_at_k(case.answer_spans, retrieved, k),
            span_reciprocal_rank(case.answer_spans, retrieved),
            span_ndcg_at_k(case.answer_spans, retrieved, k),
        )
```

Update `format_span_report`:

```python
def format_span_report(report: SpanEvalReport) -> str:
    header = (
        f"{'type':<14} {'n':>3}  "
        f"{'note_r@' + str(report.k):<8} {'note_mrr':<8} {'note_ndcg':<9}  "
        f"{'span_r@' + str(report.k):<8} {'span_mrr':<8} {'span_ndcg'}"
    )
    lines = [header]
    for s in (*report.by_type, report.overall):
        lines.append(
            f"{s.type:<14} {s.n:>3}  "
            f"{s.note_recall_at_k:>8.3f} {s.note_mrr:>8.3f} {s.note_ndcg_at_k:>9.3f}  "
            f"{s.span_recall_at_k:>8.3f} {s.span_mrr:>8.3f} {s.span_ndcg_at_k:>9.3f}"
        )
    return "\n".join(lines)
```

- [ ] **Step 6: Run the full eval test suite to verify nothing else depended on the old shape**

Run: `uv run pytest tests/eval -v -m "not integration"`
Expected: all pass, including the pre-existing `test_spaneval.py` tests

- [ ] **Step 7: Commit**

```bash
git add src/ariostea/eval/metrics.py src/ariostea/eval/span_metrics.py \
        src/ariostea/eval/spaneval.py tests/eval/test_metrics.py \
        tests/eval/test_span_metrics.py tests/eval/test_spaneval.py
git commit -m "feat(eval): report nDCG at note and span granularity"
```

---

### Task 11: Difficulty guard

**Files:**
- Create: `src/ariostea/eval/difficulty.py`
- Test: `tests/eval/test_difficulty.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from ariostea.eval.difficulty import (
    ClusterBaseline,
    cluster_baselines,
    flag_easy_clusters,
    format_baselines,
)
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase


def _case(cluster: str, query: str) -> WikiGoldCase:
    note = f"{cluster}/note.md"
    return WikiGoldCase(
        query=query,
        query_lang="en",
        type="paraphrase",
        scenario="paraphrase",
        expected_notes=(note,),
        answer_spans=(AnswerSpan(note=note, text="perfect fifths"),),
    )


def test_baselines_are_grouped_by_cluster():
    cases = [_case("strings", "a"), _case("cheese", "b")]

    def span_fn(query, k):
        return [("strings/note.md", "tuned in perfect fifths")]

    baselines = cluster_baselines(cases, span_fn, k=5)
    assert [b.cluster for b in baselines] == ["cheese", "strings"]
    assert [b.n for b in baselines] == [1, 1]


def test_a_cluster_the_dense_channel_always_answers_scores_one():
    def span_fn(query, k):
        return [("strings/note.md", "tuned in perfect fifths")]

    (baseline,) = cluster_baselines([_case("strings", "a")], span_fn, k=5)
    assert baseline.note_recall_at_k == 1.0
    assert baseline.span_recall_at_k == 1.0


def test_a_cluster_the_dense_channel_never_answers_scores_zero():
    def span_fn(query, k):
        return [("other/note.md", "nothing relevant")]

    (baseline,) = cluster_baselines([_case("strings", "a")], span_fn, k=5)
    assert baseline.note_recall_at_k == 0.0


def test_cluster_baselines_rejects_a_case_with_no_expected_notes():
    empty = WikiGoldCase(
        query="q",
        query_lang="en",
        type="paraphrase",
        scenario="paraphrase",
        expected_notes=(),
        answer_spans=(),
    )
    with pytest.raises(ValueError, match="no expected_notes"):
        cluster_baselines([empty], lambda q, k: [], k=5)


def test_flag_easy_clusters_uses_note_recall_and_the_threshold():
    easy = ClusterBaseline("cheese", 10, 0.96, 0.90)
    hard = ClusterBaseline("strings", 10, 0.60, 0.40)
    assert flag_easy_clusters([easy, hard], threshold=0.95) == ("cheese",)


def test_flag_easy_clusters_is_inclusive_at_the_threshold():
    borderline = ClusterBaseline("cheese", 10, 0.95, 0.90)
    assert flag_easy_clusters([borderline], threshold=0.95) == ("cheese",)


def test_format_baselines_marks_the_flagged_clusters():
    text = format_baselines([ClusterBaseline("cheese", 10, 0.96, 0.90)], threshold=0.95)
    assert "cheese" in text and "TOO EASY" in text


def test_format_baselines_says_so_when_nothing_is_flagged():
    text = format_baselines([ClusterBaseline("strings", 10, 0.60, 0.40)], threshold=0.95)
    assert "TOO EASY" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_difficulty.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ariostea.eval.difficulty'`

- [ ] **Step 3: Write the implementation**

```python
"""The difficulty guard: a dense-only baseline per cluster, and a flag for the
clusters that are too easy to measure anything on.

The corpus this replaces failed for exactly one reason — with one plausible
target per query, every method scored near 1.0 and no improvement was visible.
Clusters are the unit because difficulty is a property of a cluster's internal
similarity, not of the corpus as a whole: `board-games` can be trivially easy
while `string-instruments` is hard, and an overall average hides that.

Dense-only is the baseline because it is the channel that needs no lexical
overlap. If a plain embedding lookup already finds the answer nine times out
of ten, nothing downstream — fusion, reranking, blurbs — has room to show a
difference on that cluster.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.harness import SpanSearchFn, dedupe
from ariostea.eval.metrics import recall_at_k
from ariostea.eval.span_metrics import span_recall_at_k
from ariostea.eval.wiki_gold import WikiGoldCase
from ariostea.eval.wiki_notes import cluster_of

# The design doc's "flag clusters scoring >= ~0.95 as too easy".
EASY_THRESHOLD = 0.95


@dataclass(frozen=True)
class ClusterBaseline:
    cluster: str
    n: int
    note_recall_at_k: float
    span_recall_at_k: float


def cluster_baselines(
    cases: list[WikiGoldCase], span_fn: SpanSearchFn, k: int, pool: int = 50
) -> tuple[ClusterBaseline, ...]:
    """One baseline per cluster, in cluster-name order.

    A case's cluster is the first path segment of its first expected note. A
    case with no expected notes raises rather than landing in a `""` bucket:
    `validate_wiki_gold` already rejects that shape, so reaching here means
    the gold was assembled by some path that skipped validation, and silently
    inventing an empty cluster would hide it.

    Both recalls are reported. Note-level is what the flag reads — it answers
    "can the dense channel find the right article at all", which is the sense
    in which the old corpus was too easy. Span-level is reported alongside
    because a cluster can be easy to find and still hard to answer, and that
    gap is worth seeing rather than averaging away.
    """
    grouped: dict[str, list[tuple[float, float]]] = {}
    for case in cases:
        if not case.expected_notes:
            raise ValueError(f"case {case.query!r} has no expected_notes to take a cluster from")
        retrieved = span_fn(case.query, pool)
        notes = dedupe([note for note, _ in retrieved])
        grouped.setdefault(cluster_of(case.expected_notes[0]), []).append(
            (
                recall_at_k(set(case.expected_notes), notes, k),
                span_recall_at_k(case.answer_spans, retrieved, k),
            )
        )
    return tuple(
        ClusterBaseline(
            cluster=cluster,
            n=len(rows),
            note_recall_at_k=sum(note for note, _ in rows) / len(rows),
            span_recall_at_k=sum(span for _, span in rows) / len(rows),
        )
        for cluster, rows in sorted(grouped.items())
    )


def flag_easy_clusters(
    baselines: list[ClusterBaseline] | tuple[ClusterBaseline, ...],
    threshold: float = EASY_THRESHOLD,
) -> tuple[str, ...]:
    """Clusters whose dense-only note recall reaches `threshold`.

    Inclusive at the threshold: a cluster sitting exactly on the line is the
    case the guard exists to surface, not one to wave through.
    """
    return tuple(b.cluster for b in baselines if b.note_recall_at_k >= threshold)


def format_baselines(
    baselines: list[ClusterBaseline] | tuple[ClusterBaseline, ...],
    threshold: float = EASY_THRESHOLD,
) -> str:
    flagged = set(flag_easy_clusters(baselines, threshold))
    lines = [f"{'cluster':<22} {'n':>3}  {'note_recall':>11}  {'span_recall':>11}"]
    for b in baselines:
        mark = "  <-- TOO EASY" if b.cluster in flagged else ""
        lines.append(
            f"{b.cluster:<22} {b.n:>3}  {b.note_recall_at_k:>11.3f}  "
            f"{b.span_recall_at_k:>11.3f}{mark}"
        )
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_difficulty.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/ariostea/eval/difficulty.py tests/eval/test_difficulty.py
git commit -m "feat(eval): per-cluster dense-only difficulty guard"
```

---

### Task 12: The generation runner

**Files:**
- Create: `eval/generate_gold.py`
- Test: `tests/eval/test_generate_gold.py`

- [ ] **Step 1: Write the failing tests**

The script lives outside the importable package, so the test loads it by path — the same pattern `tests/eval/test_build_wiki_corpus.py` already uses. **`sys.modules[spec.name] = module` before `exec_module` is required**, otherwise `@dataclass` cannot resolve the module's own annotations and import fails.

```python
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "eval" / "generate_gold.py"
_SPEC = importlib.util.spec_from_file_location("generate_gold", _PATH)
generate_gold = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = generate_gold
_SPEC.loader.exec_module(generate_gold)

from ariostea.eval.gold_passages import Passage  # noqa: E402
from ariostea.eval.wiki_gold import load_wiki_gold, validate_wiki_gold  # noqa: E402

NOTES = {
    "strings/violin.md": "# Violin\n\nThe violin is tuned in perfect fifths: G, D, A, E.\n"
}
TITLES = {"strings/violin.md": "Violin"}
PASSAGE = Passage(
    note="strings/violin.md",
    heading="Tuning",
    text="The violin is tuned in perfect fifths: G, D, A, E.",
    offset=10,
    note_chars=9000,
    rare_terms=("ponticello",),
)

GOOD_GENERATION = (
    '{"query": "which four notes does the instrument sound open", '
    '"answer_span": "tuned in perfect fifths: G, D, A, E"}'
)
APPROVAL = '{"answers": true, "unambiguous": true, "title_only": false, "reason": "ok"}'


class FakeChat:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)

    def complete(self, system: str, user: str) -> str:
        return self._responses.pop(0) if self._responses else "not json"


def test_a_good_candidate_becomes_a_gold_case():
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert rejections == []
    assert len(cases) == 1
    assert cases[0].expected_notes == ("strings/violin.md",)
    assert cases[0].answer_spans[0].text == "tuned in perfect fifths: G, D, A, E"


def test_a_cross_lingual_case_gets_the_arrow_scenario():
    generation = '{"query": "quali note produce lo strumento", "answer_span": "tuned in perfect fifths: G, D, A, E"}'
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(generation), FakeChat(APPROVAL), [("cross_lingual", PASSAGE)], NOTES, TITLES
    )
    assert rejections == []
    assert cases[0].query_lang in {"it", "es"}
    assert cases[0].scenario == f"en→{cases[0].query_lang}"


def test_an_unparseable_generation_is_recorded_as_a_generate_rejection():
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat("sorry"), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "generate"


def test_a_fabricated_span_is_recorded_as_an_automatic_rejection():
    bad = '{"query": "what tuning is used here", "answer_span": "tuned in perfect fourths"}'
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(bad), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "automatic" and "verbatim" in rejections[0].reason


def test_a_judge_veto_is_recorded_as_an_adversarial_rejection():
    veto = '{"answers": false, "unambiguous": true, "title_only": false, "reason": "no"}'
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(veto), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "adversarial"


def test_the_judge_is_not_called_when_stage_one_already_rejected():
    # Stage 2 costs a model call; spending it on a span that is not even in
    # the note is pure waste on a ~150-candidate run.
    class CountingJudge:
        calls = 0

        def complete(self, system, user):
            CountingJudge.calls += 1
            return APPROVAL

    bad = '{"query": "what tuning is used here", "answer_span": "not in the passage at all"}'
    generate_gold.generate_and_gate(
        FakeChat(bad), CountingJudge(), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert CountingJudge.calls == 0


def test_written_gold_reloads_and_validates(tmp_path):
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    path = tmp_path / "gold.json"
    generate_gold.write_gold(path, cases)
    reloaded = load_wiki_gold(path)
    assert validate_wiki_gold(reloaded, NOTES) == []


def test_review_markdown_samples_evenly_across_types():
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    text = generate_gold.review_markdown(cases, sample_size=1)
    assert "which four notes" in text
    assert "strings/violin.md" in text
    assert "[ ]" in text  # a checkbox the reviewer actually ticks


def test_rejection_summary_counts_by_stage_and_reason():
    summary = generate_gold.rejection_summary(
        [
            generate_gold.Rejection("automatic", "span is not verbatim in the cited note", "q", "n", "s", "paraphrase"),
            generate_gold.Rejection("automatic", "span is not verbatim in the cited note", "q", "n", "s", "buried"),
            generate_gold.Rejection("adversarial", "judge: query is ambiguous (x)", "q", "n", "s", "buried"),
        ]
    )
    assert "2" in summary and "automatic" in summary


def test_shortfall_is_reported_not_swallowed(capsys):
    generate_gold.report_shortfall({"paraphrase": 40}, [("paraphrase", PASSAGE)])
    assert "paraphrase" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_generate_gold.py -v`
Expected: FAIL — `FileNotFoundError` on `eval/generate_gold.py`

- [ ] **Step 3: Write the implementation**

```python
"""Generate the span-anchored gold set over the pinned Wikipedia corpus.

Usage:  uv run python eval/generate_gold.py [--limit N] [--no-discrimination]

Pipeline, per selected passage:

    select  ->  generate  ->  stage 1 automatic  ->  stage 2 adversarial judge
                                                        |
                          stage 3 discrimination  <-----+
                                    |
                        eval/wiki/gold.json

Point it at a running OpenAI-compatible endpoint with:
    ARIOSTEA_GOLD_BASE_URL    (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_GOLD_MODEL       (default qwen/qwen3.6-35b-a3b)
    ARIOSTEA_GOLD_JUDGE_MODEL (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_GOLD_API_KEY     (default empty)

The judge model must differ from the generator model. A model asked to audit
its own output agrees with itself, which turns stage 2 from a gate into a
rubber stamp -- the script refuses to run if the two names match.

Outputs, all under eval/wiki/:
    gold.json           the accepted cases, in the Plan 1 schema
    gold_rejected.json  every rejected candidate with its stage and reason
    gold.meta.json      which models produced this gold, and the counts
    gold_review.md      a sample rendered for stage 4, human spot-review

Reproducibility differs from the corpus build on purpose. `build_wiki_corpus.py`
reproduces byte-identical output from pinned revision ids; an LLM run cannot,
even at temperature 0, across model builds. So the *committed gold* is the
artifact of record and `gold.meta.json` records what produced it. Re-running
this script writes a new gold set; it does not verify the old one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ariostea.adapters.chat.openai_compat import ChatError, OpenAICompatChat
from ariostea.eval.gold_discriminate import discrimination_filter
from ariostea.eval.gold_generate import generate_case
from ariostea.eval.gold_passages import Passage, select_passages
from ariostea.eval.gold_validate import adversarial_gate, automatic_gate
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase
from ariostea.eval.wiki_index import index_wiki_corpus, wiki_channels
from ariostea.eval.wiki_notes import load_corpus_notes, note_titles
from ariostea.ports.chat import ChatProvider

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
GOLD = WIKI_DIR / "gold.json"
REJECTED = WIKI_DIR / "gold_rejected.json"
META = WIKI_DIR / "gold.meta.json"
REVIEW = WIKI_DIR / "gold_review.md"

# ~150 queries, per the design doc's Medium tier. cross_lingual is smaller
# because it is the most expensive to review by hand and the it/es notes it
# contends with are only 8 of the 79.
BUDGET = {"paraphrase": 40, "exact_term": 40, "buried": 40, "cross_lingual": 30}

# Cross-lingual queries alternate between the two languages the corpus has
# parallel articles in, so neither track is a footnote.
LANGUAGES = (("it", "Italian"), ("es", "Spanish"))

BASE_URL = os.environ.get("ARIOSTEA_GOLD_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("ARIOSTEA_GOLD_MODEL", "qwen/qwen3.6-35b-a3b")
JUDGE_MODEL = os.environ.get("ARIOSTEA_GOLD_JUDGE_MODEL", "qwen2.5-14b-instruct-mlx")
API_KEY = os.environ.get("ARIOSTEA_GOLD_API_KEY", "")
# Generous next to the 128-token default: the response carries a query and a
# span, and a reasoning model spends tokens on a <think> block first.
MAX_TOKENS = 512
REVIEW_SAMPLE = 20


@dataclass(frozen=True)
class Rejection:
    stage: str  # "generate" | "automatic" | "adversarial" | "discrimination"
    reason: str
    query: str
    note: str
    span: str
    type: str


def _scenario(query_type: str, query_lang: str) -> str:
    """Scenario label, matching the existing gold sets' vocabulary: the query
    type for same-language cases, an arrow for cross-lingual ones."""
    return f"en→{query_lang}" if query_type == "cross_lingual" else query_type


def to_gold_case(candidate) -> WikiGoldCase:
    """`Candidate` already carries the query language `generate_case` was told
    to use, so it is read from there rather than passed again -- two sources
    for one fact is two things that can disagree."""
    return WikiGoldCase(
        query=candidate.query,
        query_lang=candidate.query_lang,
        type=candidate.type,
        scenario=_scenario(candidate.type, candidate.query_lang),
        expected_notes=(candidate.note,),
        answer_spans=(AnswerSpan(note=candidate.note, text=candidate.span),),
    )
```

Continue the same file:

```python
def generate_and_gate(
    chat: ChatProvider,
    judge: ChatProvider,
    selected: list[tuple[str, Passage]],
    notes: dict[str, str],
    titles: dict[str, str],
) -> tuple[list[WikiGoldCase], list[Rejection]]:
    """Run generation and validation stages 1 and 2 over every selected passage.

    Stage 2 is only reached by candidates stage 1 accepted: the judge costs a
    model call, and there is nothing for it to judge about a span that is not
    in the note.

    A `ChatError` from the endpoint is recorded as a rejection rather than
    allowed to abort the run. One model call failing mid-run should cost one
    candidate, not the 140 already generated.
    """
    cases: list[WikiGoldCase] = []
    rejections: list[Rejection] = []
    cross_lingual_seen = 0

    for query_type, passage in selected:
        if query_type == "cross_lingual":
            query_lang, lang_name = LANGUAGES[cross_lingual_seen % len(LANGUAGES)]
            cross_lingual_seen += 1
        else:
            query_lang, lang_name = "en", "Italian"  # lang_name unused for en types

        title = titles[passage.note]
        try:
            candidate = generate_case(
                chat, passage, query_type, title=title, lang_name=lang_name,
                query_lang=query_lang,
            )
        except (ValueError, ChatError) as exc:
            rejections.append(
                Rejection("generate", str(exc), "", passage.note, "", query_type)
            )
            continue

        reason = automatic_gate(candidate, notes, titles)
        if reason:
            rejections.append(
                Rejection("automatic", reason, candidate.query, candidate.note,
                          candidate.span, query_type)
            )
            continue

        try:
            reason = adversarial_gate(judge, candidate, title=title)
        except ChatError as exc:
            reason = f"judge unreachable: {exc}"
        if reason:
            rejections.append(
                Rejection("adversarial", reason, candidate.query, candidate.note,
                          candidate.span, query_type)
            )
            continue

        cases.append(to_gold_case(candidate))

    return cases, rejections


def write_gold(path: Path, cases: list[WikiGoldCase]) -> None:
    """Write `cases` in the Plan 1 schema `load_wiki_gold` reads back."""
    rows = [
        {
            "query": case.query,
            "query_lang": case.query_lang,
            "type": case.type,
            "scenario": case.scenario,
            "expected_notes": list(case.expected_notes),
            "answer_spans": [{"note": s.note, "text": s.text} for s in case.answer_spans],
        }
        for case in cases
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_markdown(cases: list[WikiGoldCase], sample_size: int) -> str:
    """Render a sample for stage 4, human spot-review.

    Sampled evenly across query types by taking every nth case of each type,
    not the first n: the first cases of a type all come from the same handful
    of notes (selection is round-robin by note), so reviewing them would say
    more about those articles than about the gate.
    """
    by_type: dict[str, list[WikiGoldCase]] = {}
    for case in cases:
        by_type.setdefault(case.type, []).append(case)
    per_type = max(1, sample_size // max(1, len(by_type)))

    lines = [
        "# Gold spot-review sample",
        "",
        "Tick a case only if **all three** hold: the query is answerable, the span "
        "answers it, and the span is not the only sentence in the corpus that could.",
        "",
    ]
    for query_type in sorted(by_type):
        pool = by_type[query_type]
        step = max(1, len(pool) // per_type)
        lines.append(f"## {query_type}  ({len(pool)} cases, showing {len(pool[::step][:per_type])})")
        lines.append("")
        for case in pool[::step][:per_type]:
            span = case.answer_spans[0]
            lines += [
                f"- [ ] **{case.query}**  `{case.query_lang}`",
                f"  - note: `{span.note}`",
                f"  - span: {span.text}",
                "",
            ]
    return "\n".join(lines)


def rejection_summary(rejections: list[Rejection]) -> str:
    """Counts by stage, then by reason within each stage.

    Printed at the end of every run because the first real run is expected to
    reject a large fraction, and reading *why* is the only way to tell a model
    that is bad at the task from a threshold that is set wrong.
    """
    by_stage: dict[str, dict[str, int]] = {}
    for rejection in rejections:
        # Reasons carry a model's free text in parentheses; group on the part
        # before it so the counts are about causes, not phrasings.
        key = rejection.reason.split("(")[0].strip()
        by_stage.setdefault(rejection.stage, {}).setdefault(key, 0)
        by_stage[rejection.stage][key] += 1
    lines = []
    for stage in sorted(by_stage):
        total = sum(by_stage[stage].values())
        lines.append(f"  {total:4d}  {stage}")
        for reason, count in sorted(by_stage[stage].items(), key=lambda kv: -kv[1]):
            lines.append(f"        {count:4d}  {reason}")
    return "\n".join(lines)


def report_shortfall(budget: dict[str, int], selected: list[tuple[str, Passage]]) -> None:
    """Print any query type the corpus could not fill.

    A silent shortfall reads as "the corpus supports this design" when it does
    not — the same no-silent-caps rule the corpus build's dropped-template
    report follows.
    """
    for query_type, wanted in sorted(budget.items()):
        got = sum(1 for t, _ in selected if t == query_type)
        if got < wanted:
            print(
                f"  SHORTFALL {query_type}: {got}/{wanted} eligible passages in the corpus",
                file=sys.stderr,
            )
```

And the entry point:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the wiki eval gold set.")
    parser.add_argument("--limit", type=int, help="cap the total passages (for a smoke run)")
    parser.add_argument(
        "--no-discrimination",
        action="store_true",
        help="skip stage 3, which needs to index the corpus",
    )
    args = parser.parse_args()

    if MODEL == JUDGE_MODEL:
        print(
            f"generator and judge are both {MODEL!r}; stage 2 needs an independent "
            f"model. Set ARIOSTEA_GOLD_JUDGE_MODEL to something else.",
            file=sys.stderr,
        )
        return 2

    notes = load_corpus_notes(WIKI_DIR)
    titles = note_titles(notes)
    budget = BUDGET
    if args.limit:
        share = max(1, args.limit // len(BUDGET))
        budget = {t: share for t in BUDGET}

    selected = select_passages(notes, budget)
    report_shortfall(budget, selected)
    print(f"{len(selected)} passages selected from {len({p.note for _, p in selected})} notes")

    chat = OpenAICompatChat(
        base_url=BASE_URL, model=MODEL, api_key=API_KEY, timeout=180.0, max_tokens=MAX_TOKENS
    )
    judge = OpenAICompatChat(
        base_url=BASE_URL, model=JUDGE_MODEL, api_key=API_KEY, timeout=180.0,
        max_tokens=MAX_TOKENS,
    )
    print(f"generating with {MODEL}, judging with {JUDGE_MODEL} at {BASE_URL} ...")
    cases, rejections = generate_and_gate(chat, judge, selected, notes, titles)
    print(f"{len(cases)} candidates survived stages 1 and 2")

    dropped: list[WikiGoldCase] = []
    if not args.no_discrimination:
        with tempfile.TemporaryDirectory() as tmp:
            db = str(Path(tmp) / "eval.db")
            print("indexing the corpus for the discrimination filter ...")
            container = index_wiki_corpus(WIKI_DIR, db)
            cases, dropped = discrimination_filter(cases, wiki_channels(db, container))
        rejections += [
            Rejection(
                "discrimination", "every channel answers at rank 1", case.query,
                case.expected_notes[0], case.answer_spans[0].text, case.type,
            )
            for case in dropped
        ]
        print(f"{len(dropped)} dropped as too easy; {len(cases)} remain")

    write_gold(GOLD, cases)
    REJECTED.write_text(
        json.dumps([asdict(r) for r in rejections], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    META.write_text(
        json.dumps(
            {
                "generator_model": MODEL,
                "judge_model": JUDGE_MODEL,
                "base_url": BASE_URL,
                "passages_selected": len(selected),
                "accepted": len(cases),
                "rejected": len(rejections),
                "by_type": {t: sum(1 for c in cases if c.type == t) for t in sorted(BUDGET)},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REVIEW.write_text(review_markdown(cases, REVIEW_SAMPLE), encoding="utf-8")

    print("\nrejections by stage:")
    print(rejection_summary(rejections))
    print(f"\nwrote {len(cases)} cases to {GOLD}")
    print(f"spot-review sample: {REVIEW}")
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/eval/test_generate_gold.py -v`
Expected: 10 passed

- [ ] **Step 5: Verify the guard against a self-judging run**

Run: `ARIOSTEA_GOLD_JUDGE_MODEL="qwen/qwen3.6-35b-a3b" ARIOSTEA_GOLD_MODEL="qwen/qwen3.6-35b-a3b" uv run python eval/generate_gold.py --no-discrimination; echo "exit=$?"`
Expected: the "stage 2 needs an independent model" message and `exit=2`, with no model call attempted

- [ ] **Step 6: Commit**

```bash
git add eval/generate_gold.py tests/eval/test_generate_gold.py
git commit -m "feat(eval): gold generation runner with a four-stage gate"
```

---

### Task 13: The evaluation runner

**Files:**
- Create: `eval/run_wiki_eval.py`
- Test: `tests/eval/test_run_wiki_eval.py`

- [ ] **Step 1: Write the failing test**

```python
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "eval" / "run_wiki_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_wiki_eval", _PATH)
run_wiki_eval = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_wiki_eval
_SPEC.loader.exec_module(run_wiki_eval)


def test_missing_gold_exits_with_a_pointer_to_the_generator(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_wiki_eval, "GOLD", tmp_path / "absent.json")
    assert run_wiki_eval.main([]) == 2
    assert "generate_gold.py" in capsys.readouterr().err


def test_invalid_gold_is_reported_rather_than_evaluated(monkeypatch, tmp_path, capsys):
    # A span that no longer appears in its note means the corpus moved under
    # the gold. Scoring it would silently report a recall drop as a retrieval
    # regression.
    gold = tmp_path / "gold.json"
    gold.write_text(
        '[{"query": "q", "query_lang": "en", "type": "paraphrase", "scenario": "paraphrase",'
        ' "expected_notes": ["string-instruments/violin.md"],'
        ' "answer_spans": [{"note": "string-instruments/violin.md",'
        ' "text": "a span that is not in the corpus"}]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_wiki_eval, "GOLD", gold)
    assert run_wiki_eval.main([]) == 3
    assert "span text not found" in capsys.readouterr().err


@pytest.mark.integration
def test_a_real_run_prints_a_report_per_channel(capsys):
    assert run_wiki_eval.main([]) == 0
    out = capsys.readouterr().out
    for channel in ("DENSE", "SPARSE", "HYBRID"):
        assert channel in out
    assert "cluster" in out  # the difficulty guard table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/eval/test_run_wiki_eval.py -v -m "not integration"`
Expected: FAIL — `FileNotFoundError` on `eval/run_wiki_eval.py`

- [ ] **Step 3: Write the implementation**

```python
"""Evaluate retrieval against the span-anchored Wikipedia gold set.

Usage:  uv run python eval/run_wiki_eval.py [k]

Indexes eval/wiki/ into a throwaway database, then prints, for each retrieval
channel, note-level and span-level recall@k / MRR / nDCG broken down by query
type — plus the per-cluster dense-only difficulty guard.

The per-type breakdown is the point. An aggregate number cannot tell you that
a tokenizer change helped `exact_term` and hurt nothing else, and that
attribution is the whole reason this corpus exists.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.eval.difficulty import (
    EASY_THRESHOLD,
    cluster_baselines,
    flag_easy_clusters,
    format_baselines,
)
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import load_wiki_gold, validate_wiki_gold
from ariostea.eval.wiki_index import CHUNK_POOL, index_wiki_corpus, wiki_channels
from ariostea.eval.wiki_notes import load_corpus_notes

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
GOLD = WIKI_DIR / "gold.json"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    k = int(argv[0]) if argv else 5

    if not GOLD.exists():
        print(
            f"no gold set at {GOLD}; run `uv run python eval/generate_gold.py` first",
            file=sys.stderr,
        )
        return 2

    notes = load_corpus_notes(WIKI_DIR)
    cases = load_wiki_gold(GOLD)
    errors = validate_wiki_gold(cases, notes)
    if errors:
        # Refuse to score gold the corpus no longer supports: a span that has
        # drifted out of its note scores zero for every channel, which reads
        # as a retrieval regression rather than as stale data.
        print(f"gold set does not match the corpus ({len(errors)} problems):", file=sys.stderr)
        for error in errors[:20]:
            print(f"  {error}", file=sys.stderr)
        return 3

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        print(f"indexing {len(notes)} notes ...")
        container = index_wiki_corpus(WIKI_DIR, db)
        channels = wiki_channels(db, container)

        for label, span_fn in channels.items():
            print(f"\n=== {label} ===")
            print(format_span_report(evaluate_spans(cases, span_fn, k=k, pool=CHUNK_POOL)))

        print(f"\n=== difficulty guard (dense-only baseline, k={k}) ===")
        baselines = cluster_baselines(cases, channels["DENSE"], k=k, pool=CHUNK_POOL)
        print(format_baselines(baselines, threshold=EASY_THRESHOLD))
        easy = flag_easy_clusters(baselines, threshold=EASY_THRESHOLD)
        if easy:
            print(
                f"\n{len(easy)} cluster(s) at or above {EASY_THRESHOLD}: {', '.join(easy)}. "
                f"No experiment can show an improvement there — densify the cluster or "
                f"regenerate its queries."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the fast tests**

Run: `uv run pytest tests/eval/test_run_wiki_eval.py -v -m "not integration"`
Expected: 2 passed, 1 deselected

- [ ] **Step 5: Commit**

```bash
git add eval/run_wiki_eval.py tests/eval/test_run_wiki_eval.py
git commit -m "feat(eval): span-level eval runner over the wiki corpus"
```

---

### Task 14: Generate the real gold set, review it, commit it

This is the task that needs a human and a running model. Everything before it is testable offline; this is not.

**Files:**
- Create: `eval/wiki/gold.json`, `eval/wiki/gold_rejected.json`, `eval/wiki/gold.meta.json`, `eval/wiki/gold_review.md`
- Modify: `README.md`

- [ ] **Step 1: Start the model server and confirm both models answer**

```bash
lms server start
curl -s http://localhost:1234/v1/models | python3 -m json.tool | grep '"id"'
```
Expected: both `qwen/qwen3.6-35b-a3b` and `qwen2.5-14b-instruct-mlx` listed.

- [ ] **Step 2: Smoke-run the pipeline on a handful of passages**

Run: `uv run python eval/generate_gold.py --limit 8 --no-discrimination`
Expected: eight passages selected, a mix of accepted and rejected, and a printed rejection summary. **Read the summary before continuing.** If nearly everything is rejected at the `automatic` stage for "span is not verbatim", the model is paraphrasing rather than copying — strengthen rule 1 in `GENERATION_SYSTEM` and re-run before spending a full run on it.

- [ ] **Step 3: Inspect a few accepted cases by hand**

Run: `python3 -m json.tool eval/wiki/gold.json | head -60`
Confirm the queries read like something a person would type, and that `answer_span` values are contiguous quotes rather than stitched fragments.

- [ ] **Step 4: Run the full generation**

Run: `uv run python eval/generate_gold.py 2>&1 | tee /tmp/gold-run.log`
Expected: ~150 passages selected, the stage-by-stage counts, the discrimination filter's drop count, and the four output files written. This takes a while — 150 generation calls plus one judge call each, against a local model.

- [ ] **Step 5: Check the shape of what survived**

Run:
```bash
python3 -m json.tool eval/wiki/gold.meta.json
uv run python -c "
import json, collections
rows = json.load(open('eval/wiki/gold.json'))
print(len(rows), 'cases')
print(collections.Counter(r['type'] for r in rows))
print(collections.Counter(r['expected_notes'][0].split('/')[0] for r in rows))
print('distinct notes:', len({r['expected_notes'][0] for r in rows}))
"
```
Expected: every type represented, every cluster represented, and cases spread over a good fraction of the 79 notes. If one type collapsed to near zero, read `gold_rejected.json` for that type before proceeding — the fix belongs in the prompt or the gate, not in lowering the bar.

- [ ] **Step 6: Human spot-review (validation stage 4)**

Open `eval/wiki/gold_review.md` and work through the sample. For each case check that the query is answerable, that the span answers it, and that the span is not the only sentence in the corpus that could plausibly answer — an unfair query is as useless as a trivial one.

**This step needs the repository owner, not an agent.** Stop here and hand the file over. Record the outcome — how many of the sample were sound — in the commit message, since it is the only evidence stage 4 ran at all.

- [ ] **Step 7: Run the evaluation and read the difficulty guard**

Run: `uv run python eval/run_wiki_eval.py 5 2>&1 | tee /tmp/wiki-eval.log`
Expected: a per-type table for DENSE, SPARSE and HYBRID, then the per-cluster guard. Note in the commit message which clusters, if any, were flagged as too easy.

This run is also the first real measurement of the retrieval stack against a hard corpus. Expect it to look worse than the old eval. That is the instrument working.

- [ ] **Step 8: Confirm no production code changed**

Run: `git diff --name-only master -- src/ariostea | grep -v '^src/ariostea/eval/' ; echo "exit=$?"`
Expected: no output and `exit=1` (grep found nothing). Any line printed here is a plan violation — this phase builds evaluation capability only.

- [ ] **Step 9: Full suite, lint, format**

```bash
uv run pytest -m "not integration"
uv run ruff check .
uv run ruff format --check .
```
Expected: all pass.

- [ ] **Step 10: Document the gold set in the README**

Add to `README.md`, immediately after the existing "### Evaluation corpus" subsection:

```markdown
The gold set (`eval/wiki/gold.json`) is LLM-generated and validated by four
gates: automatic span verification, an adversarial second-model check, a
discrimination filter that drops queries every retrieval channel already
answers at rank 1, and human spot-review. `eval/wiki/gold.meta.json` records
which models produced it and `eval/wiki/gold_rejected.json` records every
candidate that was thrown out, with the reason. Unlike the corpus, the gold
set is not byte-reproducible — an LLM run is not — so the committed file is
the artifact of record:

    uv run python eval/run_wiki_eval.py     # evaluate against the committed gold
    uv run python eval/generate_gold.py     # regenerate it (needs a local model)
```

- [ ] **Step 11: Commit and open the PR**

```bash
git add eval/wiki/gold.json eval/wiki/gold_rejected.json eval/wiki/gold.meta.json \
        eval/wiki/gold_review.md README.md
git commit -m "feat(eval): committed span-anchored gold set for the wiki corpus"
git push -u origin feat/eval-gold-generation
gh pr create --title "eval: LLM gold generation pipeline (Plan 3/3)" --body "..."
```

The PR body must record: the two models used, the accept/reject counts by stage, how many spot-review cases were checked and how many were sound, which clusters the difficulty guard flagged, and the first per-type numbers from `run_wiki_eval.py`. Those numbers are the baseline every later experiment is measured against — a PR that ships the instrument without recording its first reading makes the next change unattributable.

---

## Self-review

**Spec coverage.** Design doc section 2 — `eval/generate_gold.py` over `OpenAICompatChat` (Task 12), four query types (Tasks 3, 4), validation gate stages 1–4 (Tasks 6, 7, 9, and Task 14 step 6), gold committed and network-free at eval time (Task 14). Section 3 — the schema is Plan 1's; `write_gold` emits it and Task 12 tests the round-trip through `load_wiki_gold`/`validate_wiki_gold`. Section 4 — span containment and per-type breakdown are Plan 1's; nDCG is Task 10; the difficulty guard is Task 11; both granularities are already in `SpanScore`.

**Two spec items deliberately not built.** Multi-hop/graph gold is out of scope by the design doc's own deferral. Section 4's "report … at both note-level and span-level" is satisfied by Plan 1's existing `SpanScore`, so no task re-implements it.

**Known judgement calls an implementer may need to revisit:**

- `RARE_MAX_NOTES = 2` and `BURIED_MIN_NOTE_CHARS = 8000` are guesses about a corpus nobody has measured for this purpose. Task 2 step 5 and Task 3 step 5 measure them against the real 79 notes before any model call is spent; both steps say what to change if the counts come up short.
- `TITLE_OVERLAP = 0.8` will reject some legitimate queries. That is the right direction to err — a title-answerable query in the gold set inflates every channel equally and measures nothing, while a rejected good query costs one of ~600 candidates.
- The generator/judge split assumes both models are served from one LM Studio instance. If they are served separately, `ARIOSTEA_GOLD_BASE_URL` covers only one; add a second URL variable rather than pointing both at one model, which the script already refuses.
