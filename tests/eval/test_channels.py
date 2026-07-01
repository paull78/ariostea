from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.eval.channels import (
    make_dense_search_fn,
    make_hybrid_search_fn,
    make_sparse_search_fn,
)


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
        return [_rc("c.md", 0), _rc("c.md", 1), _rc("d.md", 2)]  # two chunks of c.md


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

    assert fn("dice", 5) == ["c.md", "d.md"]
    assert ret.sparse_call == ("dice", 30, None)  # raw text + pool, no embedding


def test_sparse_search_fn_truncates_to_k():
    fn = make_sparse_search_fn(FakeRetriever(), pool=30)
    assert fn("dice", 1) == ["c.md"]


class _Chunk:
    def __init__(self, note_path):
        self.note_path = note_path
        self.heading_path = ()
        self.text = "t"


class _Ranked:
    def __init__(self, note_path):
        self.chunk = _Chunk(note_path)
        self.score = 1.0


class _Result:
    def __init__(self, chunks):
        self.chunks = chunks


class _Searcher:
    def __init__(self, ranked):
        self._ranked = ranked
        self.last = None

    def search(self, query):
        self.last = query
        return _Result(self._ranked)


class _Container:
    def __init__(self, ranked):
        self.searcher = _Searcher(ranked)


def test_hybrid_search_fn_dedupes_to_notes_and_truncates():
    c = _Container([_Ranked("a.md"), _Ranked("a.md"), _Ranked("b.md")])
    fn = make_hybrid_search_fn(c, pool=50)

    assert fn("q", 5) == ["a.md", "b.md"]
    assert fn("q", 1) == ["a.md"]
    assert c.searcher.last.text == "q"
    assert c.searcher.last.k == 50
