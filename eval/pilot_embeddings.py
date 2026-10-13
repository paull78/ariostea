"""Check that each embedding candidate loads and behaves before the sweep.

Usage:
    uv run python -u eval/pilot_embeddings.py [model ...]

Per model, with unit-length vectors on (the setting the sweep measures):
load it, report the dimension, check that vectors have length 1, check that a
query and a passage of the same text embed differently where the model uses
prefixes, chunk the wiki corpus at 160 model tokens with overlap 40 using the
model's own tokenizer, and time embedding the first 100 chunks. A model that
fails is reported with its error and the pilot moves on.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.config.container import build_chunker
from ariostea.domain.models import Note
from ariostea.eval.chunk_sweep import parse_spec

# Importing wiki_index pins fastembed's cache under $HOME (see its comment);
# fastembed reads the path when a model is built, so import order is free.
from ariostea.eval.wiki_index import MULTILINGUAL_MODEL

WIKI = Path(__file__).resolve().parent / "wiki"
CANDIDATES = (
    MULTILINGUAL_MODEL,
    "intfloat/multilingual-e5-large",
    "BAAI/bge-m3",
    "jinaai/jina-embeddings-v3",
)
SAMPLE = "Il violino si accorda per quinte."
N_CHUNKS = 100


def _chunks(emb: FastEmbedEmbeddings) -> list[str]:
    parser = ObsidianMarkdownParser()
    chunker = build_chunker(parse_spec("160t+40"), emb)
    texts: list[str] = []
    for path in sorted(WIKI.glob("*/*.md")):
        rel = f"{path.parent.name}/{path.name}"
        _, body = parser.parse(rel, path.read_text(encoding="utf-8"), 0.0)
        note = Note(
            path=rel, title="", frontmatter={}, tags=(), wikilinks=(), content_hash="", mtime=0.0
        )
        texts.extend(c.text for c in chunker.chunk(note, body))
    return texts


def pilot(model: str) -> None:
    print(f"\n=== {model}", flush=True)
    started = time.perf_counter()
    emb = FastEmbedEmbeddings(model_name=model, normalize=True)
    print(f"  loaded in {time.perf_counter() - started:.1f} s; dimension {emb.dimension}")
    print(f"  fingerprint {emb.fingerprint}")
    query, passage = np.array(emb.embed_query(SAMPLE)), np.array(emb.embed_documents([SAMPLE])[0])
    print(f"  norms: query {np.linalg.norm(query):.6f}, passage {np.linalg.norm(passage):.6f}")
    print(f"  query vs passage of the same text, cosine {float(query @ passage):.4f}")
    texts = _chunks(emb)
    print(f"  chunks in the corpus at 160t+40 with this tokenizer: {len(texts)}")
    started = time.perf_counter()
    emb.embed_documents(texts[:N_CHUNKS])
    print(f"  embedded {N_CHUNKS} chunks in {time.perf_counter() - started:.1f} s", flush=True)


def main(argv: list[str]) -> int:
    failed = []
    for model in argv or CANDIDATES:
        try:
            pilot(model)
        except Exception:  # report and move on: one broken model must not hide the rest
            failed.append(model)
            print(f"  FAILED:\n{traceback.format_exc()}", flush=True)
    print(f"\nfailed: {', '.join(failed) or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
