from ariostea.config.schema import load_config


def test_minimal_config_applies_defaults(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.vault.path == "~/Vault"
    assert cfg.embedding.provider == "local"  # default
    assert cfg.store.backend == "sqlite"  # default
    assert cfg.search.top_k == 10  # default
    assert cfg.search.k_sparse == 50  # default


def test_full_config_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(
        """
[vault]
path = "/notes"
ignore = [".obsidian/"]

[embedding]
provider = "openai_compat"
base_url = "http://localhost:11434/v1"
model = "nomic-embed-text"

[store]
backend = "sqlite"
path = "/tmp/index.db"

[search]
k_dense = 40
top_k = 8
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.embedding.base_url == "http://localhost:11434/v1"
    assert cfg.embedding.model == "nomic-embed-text"
    assert cfg.vault.ignore == [".obsidian/"]
    assert cfg.search.k_dense == 40 and cfg.search.top_k == 8


def test_rerank_defaults(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.enabled is True
    assert cfg.rerank.model == "jinaai/jina-reranker-v2-base-multilingual"
    assert cfg.rerank.pool == 100


def test_rerank_can_be_disabled(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[rerank]\nenabled = false\npool = 40\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.enabled is False
    assert cfg.rerank.pool == 40


def test_contextual_defaults_off():
    from ariostea.config.schema import Config, VaultCfg

    cfg = Config(vault=VaultCfg(path="/v"))
    assert cfg.contextual.enabled is False
    assert cfg.contextual.base_url == "http://localhost:11434/v1"
    assert cfg.contextual.model == "llama3.1"
    assert cfg.contextual.max_tokens == 128
