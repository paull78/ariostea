"""Measure the Contextual Retrieval (Phase 5) lift against a local Ollama.

Usage:  uv run python eval/run_contextual_eval.py [k]

Indexes the context-dependent corpus twice — contextualization OFF then ON
(against a real OpenAI-compatible chat endpoint) — and prints recall@k / MRR
per channel for each, plus an OFF->ON delta table.

Point it at a running endpoint with:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:11434/v1)
    ARIOSTEA_CTX_MODEL     (default llama3.1)
    ARIOSTEA_CTX_API_KEY   (default empty)

Requires the chat endpoint to be reachable: if any note fails to get a blurb
the run aborts, so a partial ON index can never masquerade as "no lift".
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, ContextualCfg, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import (
    make_dense_search_fn,
    make_hybrid_search_fn,
    make_sparse_search_fn,
)
from ariostea.eval.contextual import find_uncontextualized_notes, format_delta, read_blurb_rows
from ariostea.eval.harness import evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "contextual_corpus"
GOLD = EVAL_DIR / "contextual_gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
CHUNK_POOL = 50

BASE_URL = os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("ARIOSTEA_CTX_MODEL", "llama3.1")
API_KEY = os.environ.get("ARIOSTEA_CTX_API_KEY", "")


def _config(db: str, contextual: ContextualCfg) -> Config:
    return Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=db),
        contextual=contextual,
    )


def _build_index(db: str, contextual: ContextualCfg) -> None:
    container = build_container(_config(db, contextual))
    reindex_payload(container)


def _channels(db: str, embeddings: FastEmbedEmbeddings) -> dict:
    # A second store handle over the indexed DB keeps the production Container
    # ports-only while the eval reads the dense/sparse channels directly.
    store = SqliteStore(path=db, dim=embeddings.dimension)
    container = build_container(_config(db, ContextualCfg(enabled=False)))
    return {
        "DENSE": make_dense_search_fn(embeddings, store, CHUNK_POOL),
        "SPARSE": make_sparse_search_fn(store, CHUNK_POOL),
        "HYBRID": make_hybrid_search_fn(container, CHUNK_POOL),
    }


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)

    with tempfile.TemporaryDirectory() as tmp:
        off_db = str(Path(tmp) / "off.db")
        on_db = str(Path(tmp) / "on.db")

        print("Indexing OFF (contextualization disabled) ...")
        _build_index(off_db, ContextualCfg(enabled=False))

        print(f"Indexing ON  (contextualization via {MODEL} at {BASE_URL}) ...")
        _build_index(
            on_db,
            ContextualCfg(enabled=True, base_url=BASE_URL, model=MODEL, api_key=API_KEY),
        )

        rows = read_blurb_rows(on_db)
        missed = find_uncontextualized_notes(rows)
        if missed:
            total = len({path for path, _ in rows})
            raise SystemExit(
                f"contextualization incomplete — {len(missed)}/{total} notes produced "
                f"no blurb (is the chat endpoint running at {BASE_URL}?): {', '.join(missed)}"
            )

        off_channels = _channels(off_db, embeddings)
        on_channels = _channels(on_db, embeddings)

        for label in ("DENSE", "SPARSE", "HYBRID"):
            off_report = evaluate(cases, off_channels[label], k=k)
            on_report = evaluate(cases, on_channels[label], k=k)
            print(f"\n=== {label} — OFF ===")
            print(format_report(off_report))
            print(f"\n=== {label} — ON ===")
            print(format_report(on_report))
            print(f"\n=== {label} — Δ (OFF → ON) ===")
            print(format_delta(off_report, on_report))


if __name__ == "__main__":
    main()
