from ariostea.adapters.rerank.noop import NoopReranker
from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal):
    chunk = Chunk(
        note_path="a.md", ordinal=ordinal, heading_path=("H",), text=f"c{ordinal}", token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=0.0, dense_rank=ordinal)


def test_noop_preserves_order_and_truncates():
    candidates = [_rc(0), _rc(1), _rc(2)]
    out = NoopReranker().rerank("any query", candidates, top_n=2)
    assert [rc.chunk.ordinal for rc in out] == [0, 1]


def test_noop_handles_empty():
    assert NoopReranker().rerank("q", [], top_n=5) == []
