from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from ariostea.adapters.chat.openai_compat import OpenAICompatChat
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.adapters.contextualize.noop import NoopContextualizer
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.fuse.rrf import RRFFuser
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.rerank.fastembed_rerank import FastEmbedReranker
from ariostea.adapters.rerank.noop import NoopReranker
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.schema import Config, ContextualCfg, RerankCfg
from ariostea.indexing.index_vault import IndexVault
from ariostea.ports.embedding import EmbeddingProvider
from ariostea.ports.pipeline import Contextualizer
from ariostea.ports.rerank import Reranker
from ariostea.ports.store import DocumentReader, IndexAdmin
from ariostea.search.search_knowledge import SearchKnowledge
from ariostea.search.search_sources import SearchSources

logger = logging.getLogger(__name__)


@dataclass
class Container:
    """Assembled application: config, the use cases consumers call, and the
    admin port for status. Concrete adapters (embeddings, store) are wiring
    internals of build_container and are deliberately not exposed here."""

    config: Config
    indexer: IndexVault
    searcher: SearchKnowledge
    admin: IndexAdmin
    sources: SearchSources
    reader: DocumentReader


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def _build_reranker(cfg: RerankCfg) -> Reranker:
    """Build the configured reranker, degrading to NoopReranker (fused order)
    with a warning if the model cannot be loaded — a degraded ranking, never a
    failed search."""
    if not cfg.enabled:
        return NoopReranker()
    try:
        return FastEmbedReranker(model_name=cfg.model)
    except Exception as exc:  # model missing/offline/unsupported
        logger.warning("reranker unavailable (%s); falling back to fused order", exc)
        return NoopReranker()


def _build_contextualizer(cfg: ContextualCfg) -> Contextualizer:
    """Build the configured contextualizer. When disabled, returns a
    NoopContextualizer (plain chunks) silently; when enabled but the chat client
    can't be built, degrades to NoopContextualizer with a warning."""
    if not cfg.enabled:
        return NoopContextualizer()
    try:
        chat = OpenAICompatChat(
            base_url=cfg.base_url,
            model=cfg.model,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_tokens=cfg.max_tokens,
        )
        return LLMContextualizer(chat, model_name=cfg.model)
    except Exception as exc:  # misconfiguration
        logger.warning("contextualizer unavailable (%s); indexing plain chunks", exc)
        return NoopContextualizer()


def build_container(config: Config) -> Container:
    # Embedding provider — local fastembed for the walking skeleton.
    embeddings: EmbeddingProvider = FastEmbedEmbeddings(model_name=config.embedding.local_model)

    store_path = _expand(config.store.path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path=store_path, dim=embeddings.dimension)

    parser = ObsidianMarkdownParser()
    chunker = HeadingAwareChunker()

    # The store is injected into each use case as its narrow role
    # (DocumentWriter for indexing, ChunkRetriever for search); only its
    # IndexAdmin face is re-exposed on the Container for status.
    indexer = IndexVault(
        parser=parser,
        chunker=chunker,
        embeddings=embeddings,
        store=store,
        contextualizer=_build_contextualizer(config.contextual),
    )
    searcher = SearchKnowledge(
        embeddings=embeddings,
        retriever=store,
        fuser=RRFFuser(),
        reranker=_build_reranker(config.rerank),
        k_dense=config.search.k_dense,
        k_sparse=config.search.k_sparse,
        pool=config.rerank.pool,
    )
    sources = SearchSources(searcher=searcher, reader=store)

    return Container(
        config=config,
        indexer=indexer,
        searcher=searcher,
        admin=store,
        sources=sources,
        reader=store,
    )
