import pytest

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
    assert cfg.rerank.use_context is False


def test_rerank_use_context_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[rerank]\nuse_context = true\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.use_context is True


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


def test_contextual_granularity_defaults_to_note():
    from ariostea.config.schema import Config, VaultCfg

    cfg = Config(vault=VaultCfg(path="/v"))
    assert cfg.contextual.granularity == "note"


def test_contextual_granularity_chunk_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[contextual]\ngranularity = "chunk"\n')
    cfg = load_config(cfg_file)
    assert cfg.contextual.granularity == "chunk"


def test_contextual_granularity_rejects_unknown_value():
    from pydantic import ValidationError

    from ariostea.config.schema import ContextualCfg

    with pytest.raises(ValidationError):
        ContextualCfg(granularity="paragraph")


def test_server_defaults_are_localhost_8000(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8000


def test_server_section_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[server]\nhost = "0.0.0.0"\nport = 9001\n')
    cfg = load_config(cfg_file)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 9001


def test_chunking_defaults_to_the_measured_policy(tmp_path):
    # 160 model tokens with 40 of overlap: the policy the chunking sweep
    # adopted (docs/retrieval-tuning.md). The old 512-word cap overran the
    # multilingual embedding model's input window on one chunk in five.
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.chunking.max_tokens == 160
    assert cfg.chunking.overlap == 40
    assert cfg.chunking.unit == "model_tokens"


def test_the_original_chunking_policy_is_still_selectable(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(
        '[vault]\npath = "~/Vault"\n\n[chunking]\nmax_tokens = 512\noverlap = 0\nunit = "words"\n'
    )
    cfg = load_config(cfg_file)
    assert (cfg.chunking.max_tokens, cfg.chunking.overlap, cfg.chunking.unit) == (512, 0, "words")


def test_chunking_section_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(
        '[vault]\npath = "~/Vault"\n\n[chunking]\nmax_tokens = 128\noverlap = 32\nunit = "model_tokens"\n'
    )
    cfg = load_config(cfg_file)
    assert (cfg.chunking.max_tokens, cfg.chunking.overlap, cfg.chunking.unit) == (
        128,
        32,
        "model_tokens",
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"max_tokens": 0},
        {"max_tokens": 128, "overlap": 128},  # a window that never advances
        {"max_tokens": 128, "overlap": -1},
        {"unit": "characters"},
    ],
)
def test_chunking_rejects_impossible_settings(fields):
    from pydantic import ValidationError

    from ariostea.config.schema import ChunkingCfg

    with pytest.raises(ValidationError):
        ChunkingCfg(**fields)


def test_lowering_max_tokens_below_the_default_overlap_says_what_to_do():
    # The default overlap is 40, so a config that only sets max_tokens = 32
    # is invalid through a setting the user never wrote. Say so plainly.
    from pydantic import ValidationError

    from ariostea.config.schema import ChunkingCfg

    with pytest.raises(ValidationError, match="overlap .40. .*set overlap"):
        ChunkingCfg(max_tokens=32)
