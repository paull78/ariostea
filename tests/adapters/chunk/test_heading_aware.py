from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.domain.models import Note


def _note():
    return Note(
        path="n.md", title="N", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=0.0
    )


def test_splits_on_headings_and_tracks_heading_path():
    body = (
        "# Title\nIntro paragraph.\n\n## Section A\nAlpha content.\n\n## Section B\nBeta content.\n"
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


# --- policy: overlap, cost, fingerprint -------------------------------------

import hashlib  # noqa: E402
from pathlib import Path  # noqa: E402

from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser  # noqa: E402

WIKI = Path(__file__).resolve().parents[3] / "eval" / "wiki"


def _words(chunk):
    return chunk.text.split()


def test_default_output_is_unchanged_on_the_wiki_corpus():
    # Pinned from the chunker as it stood before overlap and cost functions
    # were added. The defaults must stay inert: any drift here silently
    # re-chunks every existing vault and invalidates the recorded baseline.
    parser, chunker = ObsidianMarkdownParser(), HeadingAwareChunker()
    digest, count = hashlib.sha256(), 0
    for path in sorted(WIKI.glob("*/*.md")):
        rel = f"{path.parent.name}/{path.name}"
        note, body = parser.parse(rel, path.read_text(encoding="utf-8"), 0.0)
        for c in chunker.chunk(note, body):
            digest.update(
                f"{c.note_path}\x00{c.ordinal}\x00{'/'.join(c.heading_path)}\x00"
                f"{c.token_count}\x00{c.text}\x01".encode()
            )
            count += 1
    assert count == 1549
    assert digest.hexdigest() == "1331695e9b3a11d6f3cc85259e45a34d2fe75cae14b9b4e44af9b9a55972c56a"


def test_overlap_repeats_the_tail_of_each_piece():
    body = "# T\n" + " ".join(f"w{i}" for i in range(30))
    chunks = HeadingAwareChunker(max_tokens=10, overlap=3).chunk(_note(), body)
    for before, after in zip(chunks, chunks[1:]):
        assert _words(before)[-3:] == _words(after)[:3]
    assert _words(chunks[-1])[-1] == "w29"


def test_overlap_never_crosses_a_heading():
    body = (
        "# A\n"
        + " ".join(f"a{i}" for i in range(12))
        + "\n\n# B\n"
        + " ".join(f"b{i}" for i in range(12))
    )
    chunks = HeadingAwareChunker(max_tokens=8, overlap=3).chunk(_note(), body)
    for c in chunks:
        assert not (
            any(w.startswith("a") for w in _words(c)) and any(w.startswith("b") for w in _words(c))
        )


def test_overlap_keeps_a_boundary_straddling_span_whole():
    # The reason overlap exists: without it this span is in no chunk at all.
    words = [f"w{i}" for i in range(20)]
    body = "# T\n" + " ".join(words)
    # "# T" plus w0..w7 fill the first 10-word window, so the cut falls
    # between w7 and w8 and this span straddles it.
    span = "w6 w7 w8 w9"

    def contains(chunks):
        return any(span in c.text for c in chunks)

    assert not contains(HeadingAwareChunker(max_tokens=10).chunk(_note(), body))
    assert contains(HeadingAwareChunker(max_tokens=10, overlap=4).chunk(_note(), body))


def test_a_cost_function_caps_each_piece():
    # Each word costs its length, standing in for a subword tokenizer.
    body = "# T\n" + " ".join(["aaaa", "bb", "cccccc", "d", "eeeee", "ff", "ggg"] * 4)
    chunks = HeadingAwareChunker(max_tokens=10, count=len).chunk(_note(), body)
    assert len(chunks) > 1
    for c in chunks:
        cost = sum(len(w) for w in _words(c) if w not in {"#", "T"})
        assert cost <= 10 or len(_words(c)) == 1


def test_a_word_costlier_than_the_cap_still_advances():
    body = "# T\n" + "tiny " + "x" * 50 + " tiny"
    chunks = HeadingAwareChunker(max_tokens=10, overlap=2, count=len).chunk(_note(), body)
    assert any("x" * 50 in c.text for c in chunks)
    assert _words(chunks[-1])[-1] == "tiny"


def test_fingerprint_is_empty_for_the_default_policy():
    # Empty so an existing index keeps its stored fingerprint on upgrade.
    assert HeadingAwareChunker().fingerprint == ""


def test_fingerprint_names_a_non_default_policy():
    assert HeadingAwareChunker(max_tokens=128).fingerprint != ""
    assert (
        HeadingAwareChunker(max_tokens=128).fingerprint
        != HeadingAwareChunker(max_tokens=128, overlap=32).fingerprint
    )
    assert (
        HeadingAwareChunker(max_tokens=128).fingerprint
        != HeadingAwareChunker(max_tokens=128, count=len, unit="model_tokens").fingerprint
    )
