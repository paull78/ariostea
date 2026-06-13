import pytest

from ariostea.config.schema import Config, VaultCfg, StoreCfg
from ariostea.config.container import build_container
from ariostea.mcp.handlers import search_payload, reindex_payload, status_payload


@pytest.mark.integration
def test_index_and_search_end_to_end(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "rag.md").write_text("# Retrieval\nVector databases store embeddings for semantic search.")
    (vault / "cooking.md").write_text("# Pasta\nBoil water, add salt, cook the pasta al dente.")

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)

    reindex_payload(container)  # full index
    assert status_payload(container.admin)["notes"] == 2

    payload = search_payload(container, query="how are embeddings stored", k=1)
    assert payload["results"]
    assert payload["results"][0]["note_path"] == "rag.md"
