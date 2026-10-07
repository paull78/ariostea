import dataclasses

import pytest

from ariostea.domain.models import Chunk, IndexStats, Note, Query, RetrievedChunk, contextual_text


def test_note_holds_metadata_and_is_frozen():
    note = Note(
        path="ideas/rag.md",
        title="RAG",
        frontmatter={"status": "draft"},
        tags=("ml", "search"),
        wikilinks=("Embeddings",),
        content_hash="abc123",
        mtime=1.0,
    )
    assert note.tags == ("ml", "search")
    with pytest.raises(dataclasses.FrozenInstanceError):
        note.path = "other.md"


def test_contextual_text_prepends_blurb():
    assert contextual_text("The double bass.", "It has four strings.") == (
        "The double bass.\n\nIt has four strings."
    )


def test_contextual_text_without_blurb_returns_text_alone():
    assert contextual_text(None, "It has four strings.") == "It has four strings."


def test_contextual_text_with_empty_blurb_returns_text_alone():
    assert contextual_text("", "It has four strings.") == "It has four strings."


def test_chunk_and_retrieved_chunk_compose():
    chunk = Chunk(
        note_path="ideas/rag.md", ordinal=0, heading_path=("RAG",), text="hello", token_count=1
    )
    rc = RetrievedChunk(chunk=chunk, score=0.9, dense_rank=0, sparse_rank=None)
    assert rc.chunk.text == "hello"
    assert rc.score == 0.9


def test_retrieved_chunk_blurb_defaults_to_none():
    chunk = Chunk(note_path="a.md", ordinal=0, heading_path=(), text="t", token_count=1)
    rc = RetrievedChunk(chunk=chunk, score=1.0)
    assert rc.context_blurb is None
    assert RetrievedChunk(chunk=chunk, score=1.0, context_blurb="b").context_blurb == "b"


def test_query_defaults():
    q = Query(text="what is rag")
    assert q.k == 10 and q.filters is None


def test_index_stats_fields():
    s = IndexStats(notes=2, chunks=5, last_indexed=1.0, config_fingerprint="fp")
    assert s.notes == 2 and s.config_fingerprint == "fp"
