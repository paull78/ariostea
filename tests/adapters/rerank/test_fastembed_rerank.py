import pytest

from ariostea.domain.models import Chunk, RetrievedChunk


def _rc(ordinal, text):
    chunk = Chunk(
        note_path=f"{ordinal}.md",
        ordinal=ordinal,
        heading_path=("H",),
        text=text,
        token_count=1,
    )
    # Deliberately bad fused order: the relevant chunk starts last.
    return RetrievedChunk(chunk=chunk, score=0.0, dense_rank=ordinal)


@pytest.mark.integration
def test_fastembed_reranker_promotes_relevant_passage():
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    candidates = [
        _rc(0, "A recipe for boiling pasta with salt and water."),
        _rc(1, "The weather forecast predicts rain over the weekend."),
        _rc(2, "Rolling dice and moving tokens on a board game."),
    ]
    out = FastEmbedReranker().rerank("how do board games use dice", candidates, top_n=2)

    assert len(out) == 2
    # The dice passage must be promoted to the top despite starting last.
    assert out[0].chunk.text.startswith("Rolling dice")
    # Scores are reranker relevance scores in descending order.
    assert out[0].score >= out[1].score
