"""Run the cross-lingual eval harness against the committed fixture vault.

Usage:  uv run python eval/run_eval.py [k]

Loads the multilingual embedding model (downloads on first run), indexes the
fixture corpus into a throwaway database, and prints a recall@k / MRR table
broken down by language direction.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.harness import dedupe, evaluate, format_report, load_gold
from ariostea.mcp.handlers import reindex_payload, search_payload

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
GOLD = EVAL_DIR / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Pull a generous chunk pool, then dedupe to notes before taking the top k.
CHUNK_POOL = 50


def make_search_fn(container):
    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=CHUNK_POOL)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    return search_fn


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cases = load_gold(GOLD)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(
            vault=VaultCfg(path=str(CORPUS), ignore=[]),
            embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
            store=StoreCfg(backend="sqlite", path=str(Path(tmp) / "eval.db")),
        )
        container = build_container(cfg)
        reindex_payload(container)
        report = evaluate(cases, make_search_fn(container), k=k)
    print(format_report(report))


if __name__ == "__main__":
    main()
