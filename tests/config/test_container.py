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
