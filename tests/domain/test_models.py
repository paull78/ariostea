import dataclasses

import pytest

from ariostea.domain.models import Chunk, IndexStats, Note, Query, RetrievedChunk


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


def test_chunk_and_retrieved_chunk_compose():
    chunk = Chunk(
        note_path="ideas/rag.md", ordinal=0, heading_path=("RAG",), text="hello", token_count=1
    )
    rc = RetrievedChunk(chunk=chunk, score=0.9, dense_rank=0, sparse_rank=None)
    assert rc.chunk.text == "hello"
    assert rc.score == 0.9


def test_query_defaults():
    q = Query(text="what is rag")
    assert q.k == 10 and q.filters is None


def test_index_stats_fields():
    s = IndexStats(notes=2, chunks=5, last_indexed=1.0, config_fingerprint="fp")
    assert s.notes == 2 and s.config_fingerprint == "fp"
