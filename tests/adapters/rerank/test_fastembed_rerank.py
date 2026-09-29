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


class _RecordingEncoder:
    """Stands in for fastembed's cross-encoder: scores passages by position
    (later is better) and remembers what it was given."""

    def __init__(self, model_name):
        self.seen: list[str] = []

    def rerank(self, query, documents):
        self.seen = list(documents)
        return [float(i) for i in range(len(documents))]


@pytest.fixture
def fake_encoder(monkeypatch):
    from ariostea.adapters.rerank import fastembed_rerank

    monkeypatch.setattr(fastembed_rerank, "TextCrossEncoder", _RecordingEncoder)


def _blurbed(ordinal, text, blurb):
    rc = _rc(ordinal, text)
    return RetrievedChunk(chunk=rc.chunk, score=0.0, dense_rank=ordinal, context_blurb=blurb)


def test_use_context_scores_blurb_then_text(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker(use_context=True)
    reranker.rerank("q", [_blurbed(0, "It has four strings.", "The double bass.")], top_n=1)

    assert reranker._model.seen == ["The double bass.\n\nIt has four strings."]


def test_use_context_falls_back_to_text_without_a_blurb(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker(use_context=True)
    reranker.rerank("q", [_blurbed(0, "plain", None), _blurbed(1, "empty", "")], top_n=2)

    assert reranker._model.seen == ["plain", "empty"]


def test_default_scores_raw_text_and_returns_it_unchanged(fake_encoder):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    reranker = FastEmbedReranker()
    out = reranker.rerank("q", [_blurbed(0, "first", "B"), _blurbed(1, "second", "B")], top_n=2)

    assert reranker._model.seen == ["first", "second"]
    assert [rc.chunk.text for rc in out] == ["second", "first"]  # fake prefers later
    assert all(rc.context_blurb == "B" for rc in out)
