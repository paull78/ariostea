from pathlib import Path

import pytest

from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.harness import dedupe, evaluate, load_gold
from ariostea.mcp.handlers import reindex_payload, search_payload

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
GOLD = REPO / "eval" / "gold.json"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@pytest.mark.integration
def test_eval_harness_runs_and_same_language_is_perfect(tmp_path):
    cfg = Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "eval.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    cases = load_gold(GOLD)

    def search_fn(query: str, k: int) -> list[str]:
        payload = search_payload(container, query=query, k=50)
        return dedupe([r["note_path"] for r in payload["results"]])[:k]

    report = evaluate(cases, search_fn, k=3)

    # The harness produces a complete report over every gold case...
    assert report.overall.n == len(cases)
    by = {d.direction: d for d in report.by_direction}
    assert set(by) == {"same", "en→it", "it→en"}

    # ...and same-language retrieval on the fixture is a stable 1.0 floor.
    # Cross-lingual directions are reported but NOT asserted — lifting those
    # numbers is exactly what the later multilingual-reranking work is for.
    assert by["same"].recall_at_k == 1.0
