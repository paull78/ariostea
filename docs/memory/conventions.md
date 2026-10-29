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

### vault-hardening.14 · decision · 2026-10-26 · spec
Testing: use cases against an in-memory `Vault` fake, the adapter on real temp files, a seeded random property test for the chunker (no new dependency), CLI through Typer's runner, integration tests for the memory bound and for unchanged wiki chunks at the defaults, and the hands-on test's synthetic-vault generator committed as an end-to-end fixture. The generator, its MCP client and queries were copied from the session scratchpad to `docs/current/assets/` so they survive the session.
**Why:** the awkward vault found every finding in this cycle, so it becomes a permanent regression test and the gates' hands-on run. Unchanged wiki chunks prove retrieval is untouched without a 15-minute eval. Rejected: throwaway hands-on scripts at each gate (nothing stops the same bugs coming back).

### vault-hardening.17 · decision · 2026-10-26 · challenge round
A fast test parses the imports of `domain/`, `ports/`, `indexing/` and `search/` with `ast` and fails on any import of adapters, config, the MCP layer, or the vendor libraries (`watchfiles`, `fastembed`, `sqlite3`, `sqlite_vec`, `mcp`, `httpx`). Checked on master: only `indexing/watch_vault.py` importing `watchfiles` breaks it.
**Why:** hexagonal lens, a boundary nothing checks erodes; that is how the filesystem scanner and `watchfiles` ended up inside `indexing/`. Standard library only, no new dependency. Rejected: relying on review, as before.

### vault-hardening.30 · issue · 2026-10-26 · step 09
`uv run ruff check .` fails on `docs/current/assets/q.py` (E401, I001), committed at gate 0 with the hands-on assets. `src`, `tests` and `eval` are clean. The assets move to `docs/archive/` at wrap and would keep failing there.
**Why:** found by the lint run in step 09. Proposed fix at wrap: exclude `docs/` from ruff in `pyproject.toml`, or leave the archived scripts as they ran. CI runs `ruff check .` and `ruff format --check .`, so both failed: 2 lint errors and 3 files to reformat, all in `docs/current/assets/`. Status: resolved at the final gate (2026-10-27): the owner chose `extend-exclude = ["docs"]` under `[tool.ruff]`, so the scripts stay as they ran.

### search-speed.25 · decision · 2026-10-29 · final gate
`README.md` is the product page: purpose, tools, example, quick start, running the index, development, licence. `LEARNING.md` holds the learning journey: how it is built, the eval set, the trials, the defaults today. `docs/retrieval-tuning.md` is the full account behind the trials, and `ARCHITECTURE.md` the ports and flows. A new trial goes in `LEARNING.md` and `docs/retrieval-tuning.md`; the README only quotes the figures a user needs.
**Why:** at 470 lines the README buried how to connect a client behind the trials. Rejected: one page with the trials trimmed.
