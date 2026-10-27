from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, model_validator


class ConfigError(Exception):
    """The config file is missing or invalid. The message is one line, fit to
    print as it is."""


class VaultCfg(BaseModel):
    path: str
    ignore: list[str] = []


# Keys of the remote embedding provider, which was never wired in: a config
# that still sets one would otherwise be read and silently ignored.
_REMOVED_EMBEDDING_KEYS = ("provider", "base_url", "api_key", "model")


class EmbeddingCfg(BaseModel):
    # The embedding sweep's winner (docs/retrieval-tuning.md, "Embedding
    # model"): multilingual, 1024-dim, unit vectors of its own. The chunking
    # default was measured with it.
    local_model: str = "BAAI/bge-m3"
    # Return unit-length vectors, so the store's Euclidean ranking equals cosine
    # ranking. Off keeps existing vaults' vectors; switching it re-embeds.
    normalize: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_keys(cls, data: object) -> object:
        if isinstance(data, dict):
            for key in _REMOVED_EMBEDDING_KEYS:
                if key in data:
                    raise ValueError(
                        f"embedding.{key} was removed: Ariostea embeds locally with "
                        "fastembed; set embedding.local_model"
                    )
        return data


class StoreCfg(BaseModel):
    backend: str = "sqlite"
    path: str = "~/.ariostea/index.db"


class SearchCfg(BaseModel):
    k_dense: int = 50
    k_sparse: int = 50
    top_k: int = 10


class RerankCfg(BaseModel):
    enabled: bool = True
    model: str = "jinaai/jina-reranker-v2-base-multilingual"
    pool: int = 100  # candidates fused before reranking selects the final top_k
    # Score the context blurb with the chunk. Only matters when contextual
    # indexing is on; off until the wiki measurement says otherwise.
    use_context: bool = False


class ContextualCfg(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.1"
    timeout: float = 30.0
    max_tokens: int = 128
    # "note": one blurb per note, prepended to every chunk (today's behaviour).
    # "chunk": one LLM call per chunk, each seeing the whole document
    # (Anthropic's Contextual Retrieval; docs/design/2026-09-29-per-chunk-context.md).
    granularity: Literal["note", "chunk"] = "note"


class ChunkingCfg(BaseModel):
    """How notes are cut into chunks.

    The defaults were measured with the default embedding model, bge-m3: 256
    model tokens with 64 of overlap (docs/retrieval-tuning.md). `unit = "model_tokens"`
    counts with the embedding model's own tokenizer, so the cap means the same
    thing in every language. `"words"` counts whitespace-separated words, which
    overruns a model's input window on text that tokenizes to more than one
    token per word; the original policy, 512 words with no overlap, is still
    available that way.
    """

    max_tokens: int = 256
    overlap: int = 64  # units repeated from the end of the previous chunk
    unit: Literal["words", "model_tokens"] = "model_tokens"

    @model_validator(mode="after")
    def _window_advances(self) -> ChunkingCfg:
        if self.max_tokens <= 0:
            raise ValueError("chunking.max_tokens must be positive")
        if not 0 <= self.overlap < self.max_tokens:
            # An overlap as large as the window would never move past a word.
            # Name both values: with a default overlap of 40, lowering only
            # max_tokens trips this through a setting the user never wrote.
            raise ValueError(
                f"chunking.overlap ({self.overlap}) must be at least 0 and below "
                f"chunking.max_tokens ({self.max_tokens}); set overlap too when lowering max_tokens"
            )
        return self


class ServerCfg(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Config(BaseModel):
    vault: VaultCfg
    embedding: EmbeddingCfg = EmbeddingCfg()
    store: StoreCfg = StoreCfg()
    search: SearchCfg = SearchCfg()
    rerank: RerankCfg = RerankCfg()
    contextual: ContextualCfg = ContextualCfg()
    chunking: ChunkingCfg = ChunkingCfg()
    server: ServerCfg = ServerCfg()


def load_config(path: str | Path) -> Config:
    """Read and validate the config file. Raises ConfigError."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}") from None
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from None
    try:
        return Config(**data)
    except ValidationError as exc:
        raise ConfigError(f"{path}: {_first_error(exc)}") from None


def _first_error(exc: ValidationError) -> str:
    """The first validation error as one line: where, then what."""
    error = exc.errors()[0]
    where = ".".join(str(part) for part in error["loc"])
    # A validator's own ValueError arrives as "Value error, <its message>".
    what = error["msg"].removeprefix("Value error, ")
    message = f"{where}: {what}" if where and not what.startswith(where) else what
    return " ".join(message.split())
