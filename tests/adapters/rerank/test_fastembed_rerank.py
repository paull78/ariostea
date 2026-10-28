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

    def __init__(self, model_name, cache_dir=None):
        self.seen: list[str] = []
        self.cache_dir = cache_dir

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
    out = reranker.rerank("q", [_blurbed(0, "It has four strings.", "The double bass.")], top_n=1)

    assert reranker._model.seen == ["The double bass.\n\nIt has four strings."]
    assert out[0].chunk.text == "It has four strings."
    assert out[0].context_blurb == "The double bass."


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


def test_reranker_passes_the_cache_dir(fake_encoder, monkeypatch, tmp_path):
    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    assert FastEmbedReranker()._model.cache_dir == str(tmp_path)


MMINILM = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _texts(*texts):
    return [_rc(i, text) for i, text in enumerate(texts)]


@pytest.mark.integration
def test_mminilm_loads_into_an_empty_cache_and_ranks(monkeypatch, tmp_path):
    from huggingface_hub import constants

    from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker

    monkeypatch.setenv("FASTEMBED_CACHE_PATH", str(tmp_path))
    # The model is public: test the fetch, not whether the machine's stored
    # Hugging Face token is still valid (an expired one gets a 401).
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_IMPLICIT_TOKEN", True)
    reranker = FastEmbedReranker(model_name=MMINILM)

    fetched = {path.name for path in tmp_path.rglob("*") if path.is_file() or path.is_symlink()}
    assert {"model.onnx", "config.json", "tokenizer.json"} <= fetched

    query = "What is the capital of France?"
    for answer in ("Paris is the capital of France.", "Parigi è la capitale della Francia."):
        out = reranker.rerank(
            query,
            _texts("Bananas are rich in potassium.", "The Moon orbits the Earth.", answer),
            top_n=3,
        )
        assert out[0].chunk.text == answer


@pytest.mark.integration
def test_container_builds_mminilm_without_fallback(tmp_path):
    from ariostea.config.container import build_container
    from ariostea.config.schema import Config, RerankCfg, StoreCfg, VaultCfg
    from ariostea.eval.rerank_sweep import guarded

    (tmp_path / "vault").mkdir()
    config = Config(
        vault=VaultCfg(path=str(tmp_path / "vault")),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
        rerank=RerankCfg(model=MMINILM),
    )
    guarded(lambda: build_container(config))


def test_registration_is_idempotent(monkeypatch):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    from ariostea.adapters.rerank import fastembed_rerank

    registered = []
    monkeypatch.setattr(
        TextCrossEncoder, "list_supported_models", classmethod(lambda cls: registered)
    )
    monkeypatch.setattr(
        TextCrossEncoder,
        "add_custom_model",
        classmethod(lambda cls, model, **spec: registered.append({"model": model, **spec})),
    )
    fastembed_rerank._register_if_custom(MMINILM)
    fastembed_rerank._register_if_custom(MMINILM)
    fastembed_rerank._register_if_custom("jinaai/jina-reranker-v2-base-multilingual")
    assert [m["model"] for m in registered] == [MMINILM]
    assert registered[0]["model_file"] == "onnx/model.onnx"
