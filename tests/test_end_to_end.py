import pytest

from ariostea.config.container import build_container
from ariostea.config.schema import Config, StoreCfg, VaultCfg
from ariostea.mcp.handlers import reindex_payload, search_payload, status_payload


@pytest.mark.integration
def test_index_and_search_end_to_end(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "rag.md").write_text(
        "# Retrieval\nVector databases store embeddings for semantic search."
    )
    (vault / "cooking.md").write_text("# Pasta\nBoil water, add salt, cook the pasta al dente.")

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)

    reindex_payload(container)  # full index
    assert status_payload(container)["notes"] == 2

    payload = search_payload(container, query="how are embeddings stored", k=1)
    assert payload["results"]
    assert payload["results"][0]["note_path"] == "rag.md"


@pytest.mark.integration
def test_hybrid_finds_exact_keyword_dense_would_miss(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    # A rare, semantically-empty identifier that only exact (BM25) match can target.
    (vault / "config.md").write_text(
        "# Settings\nThe deployment uses token ZK7QWASDF as its rotation key."
    )
    (vault / "prose.md").write_text(
        "# Overview\nA general discussion of deployments, settings, and keys in systems."
    )

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    payload = search_payload(container, query="ZK7QWASDF", k=2)
    assert payload["results"]
    assert payload["results"][0]["note_path"] == "config.md"


@pytest.mark.integration
def test_search_sources_and_get_note_end_to_end(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "a.md").write_text("# Alpha\nVector databases store embeddings for search.")
    (vault / "b.md").write_text("# Beta\nEmbeddings power semantic vector search too.")

    cfg = Config(
        vault=VaultCfg(path=str(vault), ignore=[]),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)
    reindex_payload(container)

    from ariostea.mcp.handlers import get_note_payload, search_sources_payload

    sources = search_sources_payload(container, query="embeddings vector search", k=5)
    paths = {s["note_path"] for s in sources["sources"]}
    assert {"a.md", "b.md"} <= paths  # the concept appears in both notes
    assert all(s["hit_count"] >= 1 and s["title"] for s in sources["sources"])

    note = get_note_payload(container, path="a.md")
    assert note["found"] is True
    assert "Vector databases" in note["text"]

    assert get_note_payload(container, path="does-not-exist.md")["found"] is False


@pytest.mark.integration
def test_get_note_returns_the_file_of_a_long_note(tmp_path):  # AC-12
    # Long enough for overlapping windows: rebuilt from chunks, the text
    # repeated every overlap and lost its line breaks and frontmatter.
    vault = tmp_path / "vault"
    vault.mkdir()
    text = (
        "---\ntags: [history]\n---\n# Lighthouse\n\n"
        + "\n\n".join(
            f"Entry {i}: the keeper logged ships passing the strait, " + "and the weather " * 60
            for i in range(10)
        )
        + "\n\n```text\n# not a heading\n```\n"
    )
    (vault / "long.md").write_text(text)

    cfg = Config(
        vault=VaultCfg(path=str(vault)),
        store=StoreCfg(backend="sqlite", path=str(tmp_path / "index.db")),
    )
    container = build_container(cfg)
    assert reindex_payload(container)["chunks"] > 3

    from ariostea.mcp.handlers import get_note_payload

    note = get_note_payload(container, path="long.md")
    assert note["found"] is True
    assert note["text"] == text
    assert get_note_payload(container, path="../vault/long.md")["found"] is False
