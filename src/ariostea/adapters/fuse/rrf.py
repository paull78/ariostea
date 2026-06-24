from dataclasses import dataclass

from ariostea.domain.models import Chunk, RetrievedChunk
from ariostea.ports.fusion import Fuser


@dataclass
class _Entry:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class RRFFuser(Fuser):
    """Reciprocal Rank Fusion: combine ranked lists by position, not by raw
    score, so cosine-distance and BM25 scales never need reconciling."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def fuse(self, dense, sparse, k) -> list[RetrievedChunk]:
        table: dict[tuple[str, int], _Entry] = {}

        def absorb(results: list[RetrievedChunk], which: str) -> None:
            for rank, rc in enumerate(results):
                key = (rc.chunk.note_path, rc.chunk.ordinal)
                entry = table.get(key)
                if entry is None:
                    entry = _Entry(chunk=rc.chunk, score=0.0, dense_rank=None, sparse_rank=None)
                    table[key] = entry
                entry.score += 1.0 / (self.rrf_k + rank + 1)
                if which == "dense":
                    entry.dense_rank = rank
                else:
                    entry.sparse_rank = rank

        absorb(dense, "dense")
        absorb(sparse, "sparse")

        ranked = sorted(table.values(), key=lambda e: e.score, reverse=True)
        return [
            RetrievedChunk(
                chunk=e.chunk,
                score=e.score,
                dense_rank=e.dense_rank,
                sparse_rank=e.sparse_rank,
            )
            for e in ranked[:k]
        ]
