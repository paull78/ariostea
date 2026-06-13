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


@dataclass
class Container:
    config: Config
    embeddings: FastEmbedEmbeddings
    store: SqliteStore
    indexer: IndexVault
    searcher: SearchKnowledge

    @property
    def admin(self) -> SqliteStore:
        return self.store


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

    indexer = IndexVault(parser=parser, chunker=chunker, embeddings=embeddings, store=store)
    searcher = SearchKnowledge(embeddings=embeddings, retriever=store)

    return Container(
        config=config,
        embeddings=embeddings,
        store=store,
        indexer=indexer,
        searcher=searcher,
    )
