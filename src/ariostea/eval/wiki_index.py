"""Index the committed Wikipedia corpus into a throwaway database and expose
the three retrieval channels over it.

Shared by both runners rather than duplicated. `generate_gold.py` needs
channels for the discrimination filter and `run_wiki_eval.py` needs them to
report; if the two built their indexes differently -- a different embedding
model, contextualization on in one and off in the other -- the filter would
drop cases as "too easy" for a pipeline the evaluation never actually runs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import Container, build_container
from ariostea.config.schema import (
    ChunkingCfg,
    Config,
    ContextualCfg,
    EmbeddingCfg,
    RerankCfg,
    StoreCfg,
    VaultCfg,
)
from ariostea.eval.channels import (
    make_dense_chunk_fn,
    make_hybrid_chunk_fn,
    make_sparse_chunk_fn,
)
from ariostea.eval.harness import SpanSearchFn
from ariostea.mcp.handlers import reindex_payload
from ariostea.ports.chat import ChatProvider

# The corpus is deliberately multilingual; an English-only embedding model
# would make the cross_lingual track measure the model rather than the
# pipeline. Also the production default since 2026-10, but pinned here so a
# later change of default cannot silently move the eval's numbers.
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_POOL = 50


def wiki_config(
    corpus: Path,
    db: str,
    contextual: ContextualCfg | None = None,
    chunking: ChunkingCfg | None = None,
    model: str = MULTILINGUAL_MODEL,
    normalize: bool = False,
) -> Config:
    """Config for indexing `corpus` into `db`.

    `ignore=[]` overrides the default `.obsidian/` skip: the eval corpus has
    no such directory, and every file in it is a note under test. `chunking`
    defaults to the production default, so an eval that does not ask for a
    policy measures the one users get. `model` defaults to the pinned
    multilingual model; the embedding sweep passes a candidate. `normalize`
    asks for unit-length vectors, off by default like production.
    """
    return Config(
        vault=VaultCfg(path=str(corpus), ignore=[]),
        embedding=EmbeddingCfg(local_model=model, normalize=normalize),
        store=StoreCfg(backend="sqlite", path=db),
        contextual=contextual or ContextualCfg(enabled=False),
        chunking=chunking or ChunkingCfg(),
    )


def index_wiki_corpus(
    corpus: Path,
    db: str,
    contextual: ContextualCfg | None = None,
    chunking: ChunkingCfg | None = None,
    wrap_chat: Callable[[ChatProvider], ChatProvider] | None = None,
    model: str = MULTILINGUAL_MODEL,
    normalize: bool = False,
) -> Container:
    """Build the index and return the container that owns it. `wrap_chat`
    wraps the contextualizer's chat provider, e.g. in a response cache."""
    config = wiki_config(corpus, db, contextual, chunking, model=model, normalize=normalize)
    container = build_container(config, wrap_chat=wrap_chat)
    reindex_payload(container)
    return container


def wiki_channels(db: str, container: Container) -> dict[str, SpanSearchFn]:
    """The three chunk-level channels over an already-indexed `db`.

    Opens a second store handle rather than reaching into the container: the
    `Container` deliberately exposes ports and use cases, never the concrete
    `SqliteStore`, and the raw dense/sparse channels need the adapter. A
    second handle over the same file is the cheapest way to keep that boundary
    intact -- the same trick `eval/run_eval.py` already uses.

    The dense channel embeds queries with the model and normalisation the
    container indexed with, read from its config, so a candidate's index is never queried with
    another model's vectors.
    """
    embedding = container.config.embedding
    embeddings = FastEmbedEmbeddings(
        model_name=embedding.local_model, normalize=embedding.normalize
    )
    store = SqliteStore(path=db, dim=embeddings.dimension)
    return {
        "DENSE": make_dense_chunk_fn(embeddings, store, CHUNK_POOL),
        "SPARSE": make_sparse_chunk_fn(store, CHUNK_POOL),
        "HYBRID": make_hybrid_chunk_fn(container, CHUNK_POOL),
    }


def fused_config(config: Config) -> Config:
    """`config` with the reranker switched off and nothing else changed."""
    return config.model_copy(update={"rerank": RerankCfg(enabled=False)})


def fused_channel(container: Container) -> SpanSearchFn:
    """Production search without the reranker, over `container`'s index.

    The same dense and sparse retrieval and the same RRF fusion, returned in
    fused order: what users get when reranking is disabled, or when its model
    fails to load and the server falls back silently. Deliberately not part of
    `wiki_channels`: gold generation's discrimination filter drops a case only
    when every channel there answers it, so adding a channel would change what
    the filter keeps.

    Builds a second container over the same database without re-indexing it.
    """
    return make_hybrid_chunk_fn(build_container(fused_config(container.config)), CHUNK_POOL)


def context_rerank_config(config: Config) -> Config:
    """`config` with the reranker scoring blurb plus chunk, nothing else changed."""
    rerank = config.rerank.model_copy(update={"use_context": True})
    return config.model_copy(update={"rerank": rerank})


def context_rerank_channel(container: Container) -> SpanSearchFn:
    """Production search over `container`'s index with the blurb-aware
    reranker. Like `fused_channel`, a second container over the same database,
    not re-indexed, and not part of `wiki_channels` for the same reason."""
    return make_hybrid_chunk_fn(
        build_container(context_rerank_config(container.config)), CHUNK_POOL
    )
