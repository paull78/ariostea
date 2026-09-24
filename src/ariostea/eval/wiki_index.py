"""Index the committed Wikipedia corpus into a throwaway database and expose
the three retrieval channels over it.

Shared by both runners rather than duplicated. `generate_gold.py` needs
channels for the discrimination filter and `run_wiki_eval.py` needs them to
report; if the two built their indexes differently -- a different embedding
model, contextualization on in one and off in the other -- the filter would
drop cases as "too easy" for a pipeline the evaluation never actually runs.
"""

from __future__ import annotations

import os
from pathlib import Path

# fastembed defaults its model cache to `tempfile.gettempdir()/fastembed_cache`,
# which on macOS is /private/tmp -- a directory the OS periodically purges. The
# purge deletes the downloaded blobs but leaves the snapshot symlinks pointing
# at them, so the next run finds a cache that looks populated and is not: either
# onnxruntime raises NoSuchFile on a dangling `model.onnx`, or huggingface_hub
# re-fetches under a stale lock and deadlocks with no timeout (observed hung for
# three days on a zero-byte blob). Both failures happen *after* the expensive
# LLM stages, so they waste a whole run. Pinning the cache under $HOME keeps it
# outside the reaper's reach. `setdefault`, so an explicit env var still wins.
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path.home() / ".cache" / "fastembed"))

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import Container, build_container
from ariostea.config.schema import (
    ChunkingCfg,
    Config,
    ContextualCfg,
    EmbeddingCfg,
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

# The corpus is deliberately multilingual; an English-only embedding model
# would make the cross_lingual track measure the model rather than the
# pipeline. Same model the existing eval runners use, so numbers compare.
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_POOL = 50


def wiki_config(
    corpus: Path,
    db: str,
    contextual: ContextualCfg | None = None,
    chunking: ChunkingCfg | None = None,
) -> Config:
    """Config for indexing `corpus` into `db`.

    `ignore=[]` overrides the default `.obsidian/` skip: the eval corpus has
    no such directory, and every file in it is a note under test. `chunking`
    defaults to the production default, so an eval that does not ask for a
    policy measures the one users get.
    """
    return Config(
        vault=VaultCfg(path=str(corpus), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=db),
        contextual=contextual or ContextualCfg(enabled=False),
        chunking=chunking or ChunkingCfg(),
    )


def index_wiki_corpus(
    corpus: Path,
    db: str,
    contextual: ContextualCfg | None = None,
    chunking: ChunkingCfg | None = None,
) -> Container:
    """Build the index and return the container that owns it."""
    container = build_container(wiki_config(corpus, db, contextual, chunking))
    reindex_payload(container)
    return container


def wiki_channels(db: str, container: Container) -> dict[str, SpanSearchFn]:
    """The three chunk-level channels over an already-indexed `db`.

    Opens a second store handle rather than reaching into the container: the
    `Container` deliberately exposes ports and use cases, never the concrete
    `SqliteStore`, and the raw dense/sparse channels need the adapter. A
    second handle over the same file is the cheapest way to keep that boundary
    intact -- the same trick `eval/run_eval.py` already uses.
    """
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
    store = SqliteStore(path=db, dim=embeddings.dimension)
    return {
        "DENSE": make_dense_chunk_fn(embeddings, store, CHUNK_POOL),
        "SPARSE": make_sparse_chunk_fn(store, CHUNK_POOL),
        "HYBRID": make_hybrid_chunk_fn(container, CHUNK_POOL),
    }
