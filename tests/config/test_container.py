from __future__ import annotations

from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.config.container import _build_contextualizer
from ariostea.config.schema import ContextualCfg


def test_build_contextualizer_noop_when_disabled():
    ctx = _build_contextualizer(ContextualCfg(enabled=False))
    assert isinstance(ctx, NoopContextualizer)


def test_build_contextualizer_llm_when_enabled():
    ctx = _build_contextualizer(ContextualCfg(enabled=True, model="m", base_url="http://x/v1"))
    assert isinstance(ctx, LLMContextualizer)
    assert ctx.fingerprint == "llm:m"


class _Tokenizing:
    """Stands in for an embeddings adapter that can count model tokens."""

    def count_tokens(self, text: str) -> int:
        return len(text)  # one "token" per character


class _NoTokenizer:
    pass


def test_build_chunker_by_words_ignores_the_tokenizer():
    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg

    chunker = build_chunker(ChunkingCfg(), _Tokenizing())
    assert chunker.fingerprint == ""  # the default policy, unchanged


def test_build_chunker_by_model_tokens_counts_through_the_tokenizer():
    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg
    from ariostea.domain.models import Note

    chunker = build_chunker(ChunkingCfg(max_tokens=12, unit="model_tokens"), _Tokenizing())
    note = Note(
        path="n.md", title="N", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=0.0
    )
    chunks = chunker.chunk(note, "abcd efgh ijkl mnop qrst")
    # 12 tokens less the two the model adds for its start and end markers:
    # two four-character words fit, three do not.
    assert [c.text for c in chunks] == ["abcd efgh", "ijkl mnop", "qrst"]
    assert "model_tokens" in chunker.fingerprint


def test_build_chunker_by_model_tokens_needs_a_tokenizer():
    import pytest

    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg

    with pytest.raises(ValueError, match="model_tokens"):
        build_chunker(ChunkingCfg(unit="model_tokens"), _NoTokenizer())
