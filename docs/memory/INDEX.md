# Memory: project

Durable project memory. Changed only when a cycle wraps.

## Pinned

## Areas
- overview.md · what the system is, its main parts, where things live
- retrieval.md · eval protocol and decision rule, contextual indexing measured and off, local LLM limits on this Mac, vector normalisation, embedding sweep results, bge-m3 and 256t+64 as defaults, pinned eval model
- embedder.md · fastembed adapter: prefixes, custom model registration, model cache under ~/.cache/fastembed
- store.md · sqlite store: vector table size and rebuild, column migration for old indexes, BM25-only until reindex
- conventions.md · Container exposes ports, adapters subclass ports, experiment log and artifact, ask before changing a default
