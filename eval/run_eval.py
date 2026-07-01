"""Run the cross-lingual eval harness against the committed fixture vault.

Usage:  uv run python eval/run_eval.py [k]

Loads the multilingual embedding model (downloads on first run), indexes the
fixture corpus into a throwaway database, and prints a recall@k / MRR table
per scenario for each retrieval channel (dense, sparse, hybrid).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import (
    make_dense_search_fn,
    make_hybrid_search_fn,
    make_sparse_search_fn,
)
from ariostea.eval.harness import evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
GOLD = EVAL_DIR / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Pull a generous chunk pool, then dedupe to notes before taking the top k.
CHUNK_POOL = 50


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        cfg = Config(
            vault=VaultCfg(path=str(CORPUS), ignore=[]),
            embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
            store=StoreCfg(backend="sqlite", path=db),
        )
        container = build_container(cfg)
        reindex_payload(container)

        # Raw channels read the same indexed DB through a second handle, so the
        # production Container stays ports-only (no adapter leakage).
        embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
        store = SqliteStore(path=db, dim=embeddings.dimension)
        channels = {
            "DENSE": make_dense_search_fn(embeddings, store, CHUNK_POOL),
            "SPARSE": make_sparse_search_fn(store, CHUNK_POOL),
            "HYBRID": make_hybrid_search_fn(container, CHUNK_POOL),
        }

        for label, search_fn in channels.items():
            report = evaluate(cases, search_fn, k=k)
            print(f"\n=== {label} ===")
            print(format_report(report))


if __name__ == "__main__":
    main()
