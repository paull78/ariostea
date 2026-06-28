from pathlib import Path

import pytest

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.store.sqlite_store import SqliteStore
from ariostea.config.container import build_container
from ariostea.config.schema import Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.channels import make_dense_search_fn, make_sparse_search_fn
from ariostea.mcp.handlers import reindex_payload

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


@pytest.mark.integration
def test_accent_hits_sparse_and_inflection_needs_dense(tmp_path):
    db = str(tmp_path / "eval.db")
    cfg = Config(
        vault=VaultCfg(path=str(CORPUS), ignore=[]),
        embedding=EmbeddingCfg(local_model=MULTILINGUAL_MODEL),
        store=StoreCfg(backend="sqlite", path=db),
    )
    container = build_container(cfg)
    reindex_payload(container)

    # A second read-only handle on the same indexed DB for raw channel access.
    embeddings = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
    store = SqliteStore(path=db, dim=embeddings.dimension)
    dense = make_dense_search_fn(embeddings, store, pool=50)
    sparse = make_sparse_search_fn(store, pool=50)

    # Accent: the diacritic fix means an accented keyword matches on sparse.
    assert "astronomia_it.md" in sparse("città", 5)
    assert "ciclismo_es.md" in sparse("montaña", 5)

    # Inflection: FTS has no stemming, so a plural query misses the singular
    # note on sparse — but multilingual dense embeddings recover it. This is
    # the evidence that keeps multilingual FTS stemming a YAGNI backlog item.
    assert "cucito_it.md" not in sparse("bottoni", 5)
    assert "cucito_it.md" in dense("bottoni", 5)
    assert "alfareria_es.md" not in sparse("vasijas", 5)
    assert "alfareria_es.md" in dense("vasijas", 5)
