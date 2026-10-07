from __future__ import annotations

from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.adapters.contextualize.llm_chunk import LLMChunkContextualizer
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


def test_build_contextualizer_chunk_when_granularity_is_chunk():
    ctx = _build_contextualizer(
        ContextualCfg(enabled=True, model="m", base_url="http://x/v1", granularity="chunk")
    )
    assert isinstance(ctx, LLMChunkContextualizer)
    assert ctx.fingerprint == "llm-chunk:m"


def test_build_contextualizer_applies_wrap_chat():
    calls = []

    class _RecordingWrappedChat:
        def __init__(self, chat):
            self._chat = chat

        def complete(self, system, user):
            return self._chat.complete(system, user)

    def wrap_chat(chat):
        calls.append(chat)
        return _RecordingWrappedChat(chat)

    ctx = _build_contextualizer(
        ContextualCfg(enabled=True, model="m", base_url="http://x/v1"), wrap_chat=wrap_chat
    )

    assert isinstance(ctx, LLMContextualizer)
    assert len(calls) == 1
    assert isinstance(ctx._chat, _RecordingWrappedChat)
    assert ctx._chat._chat is calls[0]


class _Tokenizing:
    """Stands in for an embeddings adapter that can count model tokens."""

    def count_tokens(self, text: str) -> int:
        return len(text)  # one "token" per character


class _NoTokenizer:
    pass


def test_build_chunker_by_words_ignores_the_tokenizer():
    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg

    chunker = build_chunker(ChunkingCfg(max_tokens=512, overlap=0, unit="words"), _Tokenizing())
    assert chunker.fingerprint == ""  # the original policy: existing indexes keep their fingerprint


def test_build_chunker_by_model_tokens_counts_through_the_tokenizer():
    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg
    from ariostea.domain.models import Note

    chunker = build_chunker(
        ChunkingCfg(max_tokens=12, overlap=0, unit="model_tokens"), _Tokenizing()
    )
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


def test_the_default_policy_counts_model_tokens():
    # A changed fingerprint is what makes every existing vault re-chunk once
    # on upgrade instead of keeping 512-word chunks next to new ones.
    from ariostea.config.container import build_chunker
    from ariostea.config.schema import ChunkingCfg

    chunker = build_chunker(ChunkingCfg(), _Tokenizing())
    assert "model_tokens" in chunker.fingerprint


def test_build_reranker_passes_use_context(monkeypatch):
    from ariostea.config import container
    from ariostea.config.schema import RerankCfg

    built = {}

    class _Fake:
        def __init__(self, model_name, use_context=False):
            built.update(model_name=model_name, use_context=use_context)

    monkeypatch.setattr(container, "FastEmbedReranker", _Fake)
    container._build_reranker(RerankCfg(use_context=True))

    assert built["use_context"] is True
    assert built["model_name"] == RerankCfg().model
