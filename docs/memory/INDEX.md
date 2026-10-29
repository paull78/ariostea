# Memory: project

Durable project memory. Changed only when a cycle wraps.

## Pinned

## Areas
- overview.md · what the system is, its main parts, where things live
- retrieval.md · eval protocol and decision rule, contextual indexing measured and off, local LLM limits on this Mac, vector normalisation, embedding sweep results, bge-m3 and 256t+64 as defaults, pinned eval model, reranker cost and the speed bar, rerank sweep runner, mMiniLM at pool 30 as default, search_sources bound, measured server memory and search time
- embedder.md · fastembed adapters: prefixes, custom embedder and reranker registration, model cache under ~/.cache/fastembed, safe from two threads, the expired Hugging Face token on this machine
- store.md · sqlite store: vector table size and rebuild, column migration for old indexes, BM25-only until reindex, per-note fingerprints, deleting the index with its WAL files
- runs.md · index runs: IndexRuns and the run lock, background MCP reindex, one run at a time, WAL and two connections, run records and status
- vault.md · the Vault port and its adapters: the note rule, UTF-8 reading, guards that protect the index, watching changes
- indexing.md · chunk budget, code fences, fingerprint unchanged, get_note returns the file, progress, one-line CLI errors
- conventions.md · Container exposes ports, adapters subclass ports, experiment log and artifact, ask before changing a default, awkward-vault end-to-end test, dependency-rule test, ruff skips docs/, README and LEARNING.md roles
