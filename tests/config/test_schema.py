import pytest

from ariostea.config.schema import ConfigError, load_config


def test_minimal_config_applies_defaults(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.vault.path == "~/Vault"
    assert cfg.store.backend == "sqlite"  # default
    assert cfg.search.top_k == 10  # default
    assert cfg.search.k_sparse == 50  # default


def test_embedding_defaults_to_the_measured_model(tmp_path):
    # bge-m3 won the embedding sweep (docs/retrieval-tuning.md, "Embedding
    # model"): 157 of 167 hybrid hits at 256t+64 against 147 for the previous
    # default. A different default would ship numbers nobody measured.
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.embedding.local_model == "BAAI/bge-m3"


def test_embedding_normalize_defaults_off():
    # On would change every vault's vectors and fingerprint, forcing a reindex.
    from ariostea.config.schema import EmbeddingCfg

    assert EmbeddingCfg().normalize is False


def test_embedding_normalize_loads_from_toml(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n\n[embedding]\nnormalize = true\n')
    assert load_config(cfg_file).embedding.normalize is True


def test_full_config_parses(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(
        """
[vault]
path = "/notes"
ignore = [".obsidian/"]

[embedding]
local_model = "BAAI/bge-small-en-v1.5"
normalize = true

[store]
backend = "sqlite"
path = "/tmp/index.db"

[search]
k_dense = 40
top_k = 8
"""
    )
    cfg = load_config(cfg_file)
    assert cfg.embedding.local_model == "BAAI/bge-small-en-v1.5"
    assert cfg.embedding.normalize is True
    assert cfg.vault.ignore == [".obsidian/"]
    assert cfg.search.k_dense == 40 and cfg.search.top_k == 8


def test_rerank_defaults_match_the_sweep_pick(tmp_path):
    # mMiniLM-L12 at pool 30: the only configuration under 1 s at the quality
    # bar in the search-speed sweep (search-speed.22).
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.rerank.enabled is True
    assert cfg.rerank.model == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    assert cfg.rerank.pool == 30
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
    # 256 model tokens with 64 of overlap: measured for bge-m3, 157 of 167
    # hybrid hits against 155 at 160t+40 (bge-m3-default.06). The earlier
    # 160-token default was sized for the previous model's 128-token training.
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    cfg = load_config(cfg_file)
    assert cfg.chunking.max_tokens == 256
    assert cfg.chunking.overlap == 64
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
    # The default overlap is 64, so a config that only sets max_tokens = 32
    # is invalid through a setting the user never wrote. Say so plainly.
    from pydantic import ValidationError

    from ariostea.config.schema import ChunkingCfg

    with pytest.raises(ValidationError, match="overlap .64. .*set overlap"):
        ChunkingCfg(max_tokens=32)


def test_vault_ignore_defaults_to_nothing(tmp_path):
    # Dot-folders such as .obsidian are left out by the vault itself;
    # `ignore` only adds prefixes (vault-hardening.06).
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/Vault"\n')
    assert load_config(cfg_file).vault.ignore == []


# --- config errors are one line, never a traceback (AC-14)


@pytest.mark.parametrize("key", ["provider", "base_url", "api_key", "model"])
def test_removed_embedding_keys_are_rejected(tmp_path, key):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text(f'[vault]\npath = "~/Vault"\n\n[embedding]\n{key} = "x"\n')
    with pytest.raises(ConfigError) as err:
        load_config(cfg_file)
    message = str(err.value)
    assert f"embedding.{key} was removed" in message
    assert "embedding.local_model" in message
    assert "\n" not in message


def test_missing_config_file_is_a_config_error(tmp_path):
    missing = tmp_path / "nope.toml"
    with pytest.raises(ConfigError) as err:
        load_config(missing)
    assert str(err.value) == f"config file not found: {missing}"


def test_missing_vault_path_is_a_config_error(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text("[vault]\n")
    with pytest.raises(ConfigError) as err:
        load_config(cfg_file)
    message = str(err.value)
    assert "vault.path" in message
    assert str(cfg_file) in message
    assert "\n" not in message


def test_bad_toml_is_a_config_error(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text("[vault\npath = 1\n")
    with pytest.raises(ConfigError) as err:
        load_config(cfg_file)
    assert str(cfg_file) in str(err.value)
    assert "\n" not in str(err.value)


def test_invalid_value_is_a_config_error(tmp_path):
    cfg_file = tmp_path / "ariostea.toml"
    cfg_file.write_text('[vault]\npath = "~/V"\n\n[chunking]\nmax_tokens = 10\noverlap = 20\n')
    with pytest.raises(ConfigError) as err:
        load_config(cfg_file)
    assert "chunking.overlap (20)" in str(err.value)
    assert "\n" not in str(err.value)


def test_the_example_config_loads():
    from pathlib import Path

    example = Path(__file__).parents[2] / "ariostea.example.toml"
    assert load_config(example).vault.path
