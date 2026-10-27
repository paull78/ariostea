"""AC-07: a note holding a long whitespace-free string (an inline base64
image, a minified line) indexes within a bounded amount of memory. Before
vault-hardening the string became one chunk of thousands of tokens, padded
to the longest in the batch, and the embedding step took 20 to 25 GB."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import base64, json, random, resource, sqlite3, sys, tempfile
    from pathlib import Path

    from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
    from ariostea.config.container import build_container
    from ariostea.config.schema import Config, ContextualCfg, StoreCfg, VaultCfg
    from ariostea.domain.models import Query

    def peak():
        # Bytes on macOS, kilobytes on Linux.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss if sys.platform == "darwin" else rss * 1024

    root = Path(tempfile.mkdtemp())
    vault = root / "vault"
    vault.mkdir()
    blob = base64.b64encode(random.Random(0).randbytes(15_000)).decode()
    (vault / "clip.md").write_text(
        "# Web clip\\n\\nSaved from the web.\\n\\n![](data:image/png;base64," + blob + ")\\n\\nEnd.\\n"
    )
    config = Config(
        vault=VaultCfg(path=str(vault)),
        store=StoreCfg(path=str(root / "idx.db")),
        contextual=ContextualCfg(enabled=False),
    )
    container = build_container(config)
    container.searcher.search(Query(text="warm up", k=1))  # load the models first
    before = peak()
    container.indexer.index()
    after = peak()

    texts = [r[0] for r in sqlite3.connect(root / "idx.db").execute("SELECT text FROM chunks")]
    count = FastEmbedEmbeddings(config.embedding.local_model).count_tokens
    print(json.dumps({
        "blob": len(blob),
        "growth": after - before,
        "chunks": len(texts),
        "max_tokens": max(count(t) for t in texts),
        "budget": config.chunking.max_tokens,
    }))
    """
)


@pytest.mark.integration
def test_long_string_note_memory_bound():
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, check=True, timeout=600
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["blob"] >= 20_000
    assert result["growth"] < 2 * 1024**3, result
    assert result["chunks"] > 1, result
    # The budget counts the model's two special tokens; chunk text excludes them.
    assert result["max_tokens"] <= result["budget"] - 2, result
