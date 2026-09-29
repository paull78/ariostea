from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal, score, dense_rank=None, sparse_rank=None, path="a.md"):
    chunk = Chunk(
        note_path=path, ordinal=ordinal, heading_path=("H",), text=f"c{ordinal}", token_count=1
    )
    return RetrievedChunk(chunk=chunk, score=score, dense_rank=dense_rank, sparse_rank=sparse_rank)


def test_chunk_in_both_lists_outranks_chunk_in_one():
    # A: top of both lists. B: only dense. C: only sparse.
    dense = [_rc(0, 0.9, dense_rank=0), _rc(1, 0.8, dense_rank=1)]  # A, B
    sparse = [_rc(0, 5.0, sparse_rank=0), _rc(2, 4.0, sparse_rank=1)]  # A, C

    fused = RRFFuser().fuse(dense, sparse, k=10)

    assert fused[0].chunk.ordinal == 0  # A wins — present in both
    assert fused[0].score > fused[1].score
    # the fused A carries both ranks; B/C carry only their own
    assert fused[0].dense_rank == 0 and fused[0].sparse_rank == 0


def test_dedupes_on_note_path_and_ordinal():
    dense = [_rc(0, 0.9, dense_rank=0)]
    sparse = [_rc(0, 5.0, sparse_rank=0)]
    fused = RRFFuser().fuse(dense, sparse, k=10)
    assert len(fused) == 1
    # same chunk seen via both routes keeps both rank annotations
    assert fused[0].dense_rank == 0 and fused[0].sparse_rank == 0


def test_truncates_to_k():
    dense = [_rc(i, 1.0 / (i + 1), dense_rank=i) for i in range(5)]
    fused = RRFFuser().fuse(dense, [], k=2)
    assert len(fused) == 2


def test_fused_chunks_keep_their_blurb_from_either_channel():
    def with_blurb(rc, blurb):
        return RetrievedChunk(
            chunk=rc.chunk,
            score=rc.score,
            dense_rank=rc.dense_rank,
            sparse_rank=rc.sparse_rank,
            context_blurb=blurb,
        )

    dense = [with_blurb(_rc(0, 0.9, dense_rank=0), "note A")]  # A, dense only
    sparse = [with_blurb(_rc(1, 5.0, sparse_rank=0), "note B")]  # B, sparse only

    fused = {rc.chunk.ordinal: rc for rc in RRFFuser().fuse(dense, sparse, k=10)}

    assert fused[0].context_blurb == "note A"
    assert fused[1].context_blurb == "note B"


def test_rrf_constant_changes_weighting_but_not_presence():
    dense = [_rc(0, 0.9, dense_rank=0), _rc(1, 0.8, dense_rank=1)]
    sparse = [_rc(1, 5.0, sparse_rank=0), _rc(0, 4.0, sparse_rank=1)]
    fused = RRFFuser(rrf_k=60).fuse(dense, sparse, k=10)
    # 0 is rank0+rank1, 1 is rank1+rank0 — symmetric, both present
    assert {c.chunk.ordinal for c in fused} == {0, 1}
