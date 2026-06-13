from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ariostea.config.schema import Config
from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.indexing.index_vault import IndexVault
from ariostea.search.search_knowledge import SearchKnowledge
from ariostea.ports.store import IndexAdmin


@dataclass
class Container:
    """Assembled application: config, the use cases consumers call, and the
    admin port for status. Concrete adapters (embeddings, store) are wiring
    internals of build_container and are deliberately not exposed here."""

    config: Config
    indexer: IndexVault
    searcher: SearchKnowledge
    admin: IndexAdmin


def _expand(p: str) -> str:
    return os.path.expanduser(p)


def build_container(config: Config) -> Container:
    # Embedding provider — local fastembed for the walking skeleton.
    embeddings = FastEmbedEmbeddings(model_name=config.embedding.local_model)

    store_path = _expand(config.store.path)
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(path=store_path, dim=embeddings.dimension)

    parser = ObsidianMarkdownParser()
    chunker = HeadingAwareChunker()

    # The store is injected into each use case as its narrow role
    # (DocumentWriter for indexing, ChunkRetriever for search); only its
    # IndexAdmin face is re-exposed on the Container for status.
    indexer = IndexVault(parser=parser, chunker=chunker, embeddings=embeddings, store=store)
    searcher = SearchKnowledge(embeddings=embeddings, retriever=store)

    return Container(config=config, indexer=indexer, searcher=searcher, admin=store)
