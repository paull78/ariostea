from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel


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


class Config(BaseModel):
    vault: VaultCfg
    embedding: EmbeddingCfg = EmbeddingCfg()
    store: StoreCfg = StoreCfg()
    search: SearchCfg = SearchCfg()


def load_config(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    return Config(**data)
