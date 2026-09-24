from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator


class VaultCfg(BaseModel):
    path: str
    ignore: list[str] = [".obsidian/"]


class EmbeddingCfg(BaseModel):
    provider: str = "local"  # "local" | "openai_compat"
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    local_model: str = "BAAI/bge-small-en-v1.5"


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


class ContextualCfg(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:11434/v1"
    api_key: str = ""
    model: str = "llama3.1"
    timeout: float = 30.0
    max_tokens: int = 128


class ChunkingCfg(BaseModel):
    """How notes are cut into chunks. The defaults reproduce the original
    chunker exactly, so a config without this section behaves as before.

    `unit = "model_tokens"` counts with the embedding model's own tokenizer, so
    the cap means the same thing in every language; `"words"` counts
    whitespace-separated words, which overruns a model's input window on text
    that tokenizes to more than one token per word.
    """

    max_tokens: int = 512
    overlap: int = 0  # units repeated from the end of the previous chunk
    unit: Literal["words", "model_tokens"] = "words"

    @model_validator(mode="after")
    def _window_advances(self) -> ChunkingCfg:
        if self.max_tokens <= 0:
            raise ValueError("chunking.max_tokens must be positive")
        if not 0 <= self.overlap < self.max_tokens:
            # An overlap as large as the window would never move past a word.
            raise ValueError("chunking.overlap must be at least 0 and below max_tokens")
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
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return Config(**data)
