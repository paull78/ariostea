from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.eval.channels import make_dense_search_fn, make_sparse_search_fn


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
        return [_rc("c.md", 0), _rc("c.md", 1)]


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

    assert fn("dice", 5) == ["c.md"]
    assert ret.sparse_call == ("dice", 30, None)  # raw text + pool, no embedding
