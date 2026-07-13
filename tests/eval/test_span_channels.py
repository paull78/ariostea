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


def test_dense_chunk_fn_does_not_dedupe_repeated_notes():
    hits = [_rc("violin.md", "first chunk"), _rc("violin.md", "second chunk", ordinal=1)]
    fn = make_dense_chunk_fn(_FakeEmbeddings(), _FakeRetriever(hits), pool=10)
    assert fn("q", k=2) == [("violin.md", "first chunk"), ("violin.md", "second chunk")]
