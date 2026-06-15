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
