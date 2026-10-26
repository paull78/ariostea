# Conventions

Working rules for code and experiments in this repo, moved in from Claude Code's auto-memory at project-time setup (2026-10-26).

### setup.01 · decision · 2026-07-03 · architecture
The `Container` built in `config/container.py` exposes `config`, the use cases consumers call (`IndexVault`, `SearchKnowledge`, which stay concrete) and adapter roles only through their port type (for example `admin: IndexAdmin`). Concrete adapters such as `FastEmbedEmbeddings` or `SqliteStore` are wiring details and never appear on its public surface. Inside `build_container`, inject the store and providers into use cases as their narrow roles (`DocumentWriter`, `ChunkRetriever`), and drop any wiring object no consumer reads.
**Why:** exposing concrete adapters lets consumers couple to adapter-specific methods, which defeats dependency inversion. Use cases are exempt: an interface over a single interactor would be speculative.

### setup.02 · decision · 2026-07-03 · architecture
Every adapter explicitly subclasses the port `Protocol` it implements, listing all of them when it plays several roles: `class SqliteStore(DocumentWriter, ChunkRetriever, IndexAdmin)`, `class HeadingAwareChunker(Chunker)`. The convention covers adapters only; use cases stay concrete (setup.01).
**Why:** with structural typing alone the relationship is invisible in the adapter file and checked only where the adapter is injected. Subclassing shows the intent on the class and moves conformance errors to its definition; the import is a safe inward dependency.

### setup.03 · decision · 2026-09-24 · experiments
Every eval run (sweep, pilot or full wiki eval) is appended with `append_run` to `eval/results/runs.jsonl`, rendered with `uv run python eval/render_results.py`, committed, and `eval/results/experiment_log.html` is republished to the log artifact https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid (passed as `url`). The artifact belongs to another of the owner's organisations, so the session must be signed in to it (`/login`) before republishing. Raw eval output is saved inside the repo at once: scratch files under `/private/tmp` are purged between sessions and already lost one baseline log.
**Why:** the owner compares configurations on one shaded table (gains green, losses magenta, against each run's control) and wants the whole history in one place.

### setup.04 · decision · 2026-09-24 · experiments
Ask the owner before changing any default, above all one that forces every vault to reindex (embedding model, normalisation, chunking). Present the measured gain against the decision rule (retrieval.md, embedding-sweep.02) and let the owner decide.
**Why:** a default change costs every user a full reindex, and the owner decides whether the gain is worth it.
