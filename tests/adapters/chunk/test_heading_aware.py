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
        assert sum(len(w) for w in _words(c)) <= 10


def test_a_word_costlier_than_the_cap_is_split_to_fit():
    body = "# T\n" + "tiny " + "x" * 50 + " tiny"
    chunks = HeadingAwareChunker(max_tokens=10, overlap=2, count=len).chunk(_note(), body)
    assert all(sum(len(w) for w in _words(c)) <= 10 for c in chunks)
    assert "".join("".join(_words(c)) for c in chunks).count("x") >= 50
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


# --- every piece fits, and every window adds new text (AC-08)


def _shared(before, after):
    """How many units `after` repeats from the end of `before`."""
    for k in range(min(len(before), len(after)), 0, -1):
        if before[-k:] == after[:k]:
            return k
    return 0


def test_fit_property():
    import random

    for seed in range(500):
        rng = random.Random(seed)
        max_tokens = rng.randint(2, 40)
        overlap = rng.randint(0, max_tokens - 1)
        k = rng.randint(1, 4)

        def cost(text, k=k):  # prices any substring, like a subword tokenizer
            return len(text) // k + 1

        # Every character is a distinct CJK ideograph, so two units are equal
        # only when the chunker really repeated one: overlap is unambiguous.
        chars = (chr(0x4E00 + i) for i in range(20_000))
        words = []
        for _ in range(rng.randint(1, 30)):
            length = rng.randint(1, 3 * max_tokens * k)
            words.append("".join(next(chars) for _ in range(length)))
        chunker = HeadingAwareChunker(max_tokens=max_tokens, overlap=overlap, count=cost)
        pieces = [_words(c) for c in chunker.chunk(_note(), " ".join(words))]

        context = f"seed {seed}: max {max_tokens}, overlap {overlap}, k {k}"
        rebuilt = list(pieces[0])
        for piece in pieces:
            assert sum(cost(u) for u in piece) <= max_tokens, context
        for before, after in zip(pieces, pieces[1:]):
            shared = _shared(before, after)
            assert sum(cost(u) for u in after[:shared]) <= overlap, context
            assert len(after) > shared, context  # something new in every window
            rebuilt.extend(after[shared:])
        assert "".join(rebuilt) == "".join(words), context


def test_oversized_word_is_split_to_fit():
    word = "".join(chr(ord("a") + (i * 7) % 26) for i in range(20_000))

    def cost(text):
        return len(text) // 3 + 1

    chunks = HeadingAwareChunker(max_tokens=256, overlap=64, count=cost).chunk(_note(), word)
    assert len(chunks) > 1
    assert all(cost(c.text) <= 256 for c in chunks)
    assert "".join(c.text for c in chunks) == word


def test_no_shrinking_duplicates_before_an_oversized_word():
    # The web clip from the hands-on test: a short heading line, then one
    # whitespace-free string far over the budget. The old chunker emitted
    # windows that were each a suffix of the one before.
    body = "# Web clip\nArticle saved from the web " + "QUJD" * 4_500 + " end of clip"

    def cost(text):
        return len(text) // 4 + 1

    chunks = HeadingAwareChunker(max_tokens=256, overlap=64, count=cost).chunk(_note(), body)
    texts = [c.text for c in chunks]
    for a, b in zip(texts, texts[1:]):
        assert not a.endswith(b) and not b.endswith(a)
    assert all(sum(cost(w) for w in t.split()) <= 256 for t in texts)


def test_short_sections_untouched():
    body = "# T\nline one\n\n  line two  \n- item"
    chunks = HeadingAwareChunker(max_tokens=50, overlap=10, count=len).chunk(_note(), body)
    assert [c.text for c in chunks] == ["# T\nline one\n\n  line two  \n- item"]


def test_words_unit_never_splits_a_word():
    body = "# T\n" + "x" * 5_000 + " tail"
    chunks = HeadingAwareChunker(max_tokens=512).chunk(_note(), body)
    assert [c.text for c in chunks] == [body]


# --- code fences do not split sections (AC-11)


def test_chunks_inside_fences_have_empty_heading_path():
    body = '```python\n# This comment is not a title\nprint("hello")\n```\n\nBody after.'
    chunks = HeadingAwareChunker(max_tokens=200).chunk(_note(), body)
    assert len(chunks) == 1
    assert chunks[0].heading_path == ()
    assert chunks[0].text == body


def test_fenced_hash_does_not_open_a_section():
    body = "# Setup\nInstall it.\n```bash\n# step one\nmake\n```\nThen run it."
    chunks = HeadingAwareChunker(max_tokens=200).chunk(_note(), body)
    assert [c.heading_path for c in chunks] == [("Setup",)]
