# Overview

Ariostea is a local MCP server that indexes an Obsidian vault and answers retrieval queries over it: `status`, `reindex`, `search_knowledge`, `search_sources` and `get_note`. Everything runs on the user's machine.

## Main parts

The code follows ports and adapters under `src/ariostea/`. A test (`tests/test_architecture.py`) fails if `domain/`, `ports/`, `indexing/` or `search/` import an adapter, the config, the MCP layer or a vendor library (vault-hardening.17).

- `ports/`: Protocols for the vault, embedding, store, rerank, fusion, chat, the run lock and the indexing pipeline.
- `adapters/`: one implementation per port, each subclassing its port.
  - Vault: `FilesystemVault` and `InMemoryVault` share one note rule (`adapters/vault/rule.py`: `.md`, no dot segment, not Excalidraw, not under `vault.ignore`) and one contract suite. The filesystem adapter reads UTF-8 with the BOM dropped and wraps `watchfiles` for changes (vault.md).
  - The embedder is fastembed with bge-m3, safe to share between threads (embedder.md); the store is sqlite with sqlite-vec for vectors and FTS5 for BM25, on the write-ahead log, records each note's fingerprint and the last two runs, and migrates old indexes when it opens (store.md); fusion is reciprocal rank fusion; the reranker is the fastembed cross-encoder `jina-reranker-v2-base-multilingual`. Models are cached under `~/.cache/fastembed`.
  - The run lock: `FileRunLock`, an flock on `<index>.lock`, and `InMemoryRunLock` for tests, with one contract suite (runs.md).
  - The chunker (`adapters/chunk/heading_aware.py`) cuts a word costlier than the budget to fit and ignores headings inside code fences (indexing.md).
- `indexing/`: `IndexVault` (the two guards that protect the index, skipping unreadable notes, keeping a note whose hash and fingerprint match, the delete sweep, per-note progress values), `IndexRuns` (every run takes the run lock, records how it ended and releases the lock; no threads) and `WatchVault` (each pass is a run through `IndexRuns`, warning and carrying on when a guard fails).
- `search/`: `SearchKnowledge` and `SearchSources` (dense and sparse retrieval, fused, then reranked) and `GetNote` (index lookup, then the file as written).
- `config/`: the pydantic schema and the container, which exposes config, use cases (`runs`, `watcher`, `searcher`, `sources`, `notes`) and port-typed admin only (setup.01). It opens two store connections, a writer for the run holding the lock and a reader for everything else, and passes in the clock.
- `cli.py`, `progress.py` and `mcp/`: the `ariostea` command, its progress printer (which owns the clock) and the MCP handlers. Known CLI errors print one line and exit 1. CLI `reindex` and `watch` wait for a run in progress; the MCP `reindex` tool starts a run on a thread `mcp/background.py` owns and returns at once, or refuses while a run goes on; `status` reads the run records (runs.md).
- `eval/` (package and scripts): the span-anchored wiki eval, 167 gold cases, the sweep runner and the experiment log (`eval/results/runs.jsonl`).

## Defaults today

Embedding model `BAAI/bge-m3`, chunks of 256 model tokens with overlap 64, reranker on, contextual indexing off. Hybrid span recall at k=5 on the wiki set is 0.940, 157 of 167 (bge-m3-default.06). A running server holds about 5 GB and a search takes about 2.5 s, nearly all from the reranker (vault-hardening.29). After a change of model or chunking, run `ariostea reindex` once; if it stops partway, the next run picks up where it stopped. Chunker code fixes reach a note only when it next changes, since the fingerprint is unchanged; deleting the index with its `-wal` and `-shm` files (`rm index.db*`, nothing running) rebuilds everything (vault-hardening.08, server-robustness.08).

## Running and testing

```bash
uv run ariostea reindex                 # build the index for the vault in ariostea.toml
uv run ariostea watch                   # first pass, then follow changes
uv run ariostea serve                   # stdio MCP server
uv run pytest -m "not integration"      # fast suite
uv run pytest                           # everything, about twenty-five minutes, loads real models
uv run ruff check . && uv run ruff format --check .   # what CI runs; docs/ is excluded
```

`tests/test_awkward_vault.py` builds a 55-note awkward vault and drives the real CLI and stdio server over it (vault-hardening.14); `tests/test_server_robustness.py` runs several processes on one index over the same vault: a background `reindex`, refused and waiting writers, reads during a write, `watch`, and `status` across processes. A first index of that vault takes about 30 seconds.

## Key decisions

- embedding-sweep.02 and .18: the decision rule for retrieval experiments.
- embedding-sweep.19 and bge-m3-default.06: bge-m3 with 256-token chunks is the default.
- bge-m3-default.01 and .09: how the store migrates old indexes.
- vault-hardening.02, .04 and .06: what protects the index, how bad files are handled, what counts as a note.
- vault-hardening.13, .15 and .24: the vault is a driven actor behind a port, change notification included.
- vault-hardening.05 and .07: the chunk budget always holds; code fences are not structure.
- server-robustness.09, .02 and .03: every run goes through `IndexRuns` and the run lock; the MCP adapter owns the background thread; one run at a time, readers on the write-ahead log.
- server-robustness.10 and .05: per-note fingerprints make runs resumable; `status` reads run records from the index.

Full history of the retrieval experiments: `docs/retrieval-tuning.md`. Finished cycles: `docs/archive/`.
