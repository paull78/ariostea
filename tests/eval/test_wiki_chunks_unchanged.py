"""AC-19: the default chunking of the wiki eval corpus is unchanged by the
vault-hardening fixes, so the retrieval numbers measured on it still hold
and no eval rerun is needed. The digest was computed on the chunker before
the fixes (step 06, vault-hardening)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.adapters.vault.filesystem import FilesystemVault
from ariostea.config.container import build_chunker
from ariostea.config.schema import ChunkingCfg, EmbeddingCfg

CORPUS = Path(__file__).resolve().parents[2] / "eval" / "wiki"
GOLDEN = "198542d4f548d11054755c0a905d8314fc959e9d37a7f8c3c208263d91f8d533"  # 2,858 chunks


def wiki_chunks_digest() -> tuple[str, int]:
    vault = FilesystemVault(CORPUS)
    chunker = build_chunker(ChunkingCfg(), FastEmbedEmbeddings(EmbeddingCfg().local_model))
    parser = ObsidianMarkdownParser()
    digest, count = hashlib.sha256(), 0
    for path in vault.notes():
        file = vault.read(path)
        note, body = parser.parse(path, file.text, file.mtime)
        for chunk in chunker.chunk(note, body):
            for part in (path, str(chunk.ordinal), "\x1f".join(chunk.heading_path), chunk.text):
                digest.update(part.encode("utf-8") + b"\x1e")
            count += 1
    return digest.hexdigest(), count


@pytest.mark.integration
def test_wiki_chunks_unchanged():
    digest, count = wiki_chunks_digest()
    assert digest == GOLDEN, f"{count} chunks, digest {digest}"


if __name__ == "__main__":
    print(*wiki_chunks_digest())
