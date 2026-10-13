# Overview

Ariostea is a local MCP server that indexes an Obsidian vault and answers retrieval queries over it: `status`, `reindex`, `search_knowledge`, `search_sources` and `get_note`. Everything runs on the user's machine.

## Main parts

The code follows ports and adapters under `src/ariostea/`.

- `ports/`: Protocols for embedding, store, rerank, fusion, chat and the indexing pipeline.
- `adapters/`: one implementation per port, each subclassing its port. The embedder is fastembed (`adapters/embedding/fastembed_local.py`), the store is sqlite with sqlite-vec for vectors and FTS5 for BM25 (`adapters/store/`), fusion is reciprocal rank fusion, and the reranker is the fastembed cross-encoder `jina-reranker-v2-base-multilingual`. Both fastembed adapters cache models under `~/.cache/fastembed` (`adapters/fastembed_cache.py`). The store migrates old indexes when it opens: it rebuilds a vector table of the wrong size and adds missing `chunks` columns (store.md).
- `indexing/` and `search/`: the use cases. Search runs dense and sparse retrieval, fuses them, and reranks the pool.
- `config/`: the pydantic schema (`schema.py`) and the container (`container.py`), which exposes config, use cases and port-typed admin only.
- `mcp/` and `cli.py`: the MCP handlers and the `ariostea` command.
- `eval/` (package) and `eval/` (scripts): the span-anchored wiki eval, with 167 gold cases across buried, cross_lingual, exact_term and paraphrase, the chunking sweep runner (`eval/run_chunk_sweep.py`, now with `--model` and `--normalize`) and the experiment log (`eval/results/runs.jsonl`, rendered to `experiment_log.html`).

## Defaults today

Embedding model `BAAI/bge-m3` (1024 dimensions, unit vectors of its own, so `embedding.normalize` stays off), chunks of 256 model tokens with overlap 64, reranker on, contextual indexing off. Hybrid span recall at k=5 on the wiki set is 0.940, 157 of 167, against 0.880 for the previous defaults (bge-m3-default.06). The eval pins the same model (bge-m3-default.03). After a change of model or chunking, run `ariostea reindex` once; until then a server on a new model answers from BM25 alone (bge-m3-default.07).

## Running and testing

```bash
uv run ariostea reindex                 # build the index for the vault in ariostea.toml
uv run ariostea serve                   # stdio MCP server
uv run pytest -m "not integration"      # fast suite, under a second
uv run pytest                           # everything, about twenty-three minutes, loads real models
```

## Key decisions

- embedding-sweep.02 and embedding-sweep.18: the decision rule for retrieval experiments, read as five cases out of 167.
- embedding-sweep.07 and embedding-sweep.08: the store ranks by Euclidean distance; `normalize` makes that cosine.
- embedding-sweep.19 and bge-m3-default.06: bge-m3 with 256-token chunks is the default.
- bge-m3-default.01 and bge-m3-default.09: how the store migrates old indexes; a new `chunks` column must join the migration list.
- bge-m3-default.13: the model cache lives under `~/.cache/fastembed`.

Full history of the retrieval experiments: `docs/retrieval-tuning.md`. Earlier cycles predate this workflow and are recorded in `recaps/`.
