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
