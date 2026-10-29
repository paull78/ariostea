# Step 06 · Defaults and README

**Milestone:** 3 · **Needs:** 05 · **Covers:** AC-05, AC-06

## Goal
The product matches the owner's decision at the Milestone 2 gate: `RerankCfg` defaults (if approved), README figures, the mMiniLM option, and the `search_sources` bound.

## Files touched
- `src/ariostea/config/schema.py`: `RerankCfg.model` and `RerankCfg.pool` defaults, only if the owner approved a change; `FastEmbedReranker`'s default `model_name` kept in step.
- `tests/config/test_schema.py` (or where `RerankCfg` defaults are tested): the defaults.
- `README.md`: reranker section (the two models, pool, measured trade-off), search time and memory figures, `search_sources` bounded by `rerank.pool`.
- `docs/memory` is not touched here (wrap).

## Tests to write first
- `test_rerank_defaults_match_the_sweep_pick`: `RerankCfg()` has the approved model and pool. Skipped as a no-op step if the owner keeps the defaults; then the existing default test stands.

## Implementation notes
- Memory and scratch-vault search time measured as in vault-hardening.29: stdio server on the 50-note scratch vault (generator in `docs/archive/2026-10-27-vault-hardening/assets/make_vault.py`), RSS after the first search and after seven, search time per call. Measure with the new defaults (or today's, if unchanged) and give the figures in the README, rounded as the README does today ("about").
- Before any hands-on run that writes data, pass a scratch `--config` file whose `store.path` is in the scratchpad, never `~/.ariostea/index.db` (general lesson on backing up the user's data).
- The README follows `~/.claude/VOICE.md` through human-write-time (neutral register).

## Verify
`uv run pytest -m "not integration" -q && uv run ruff check . && uv run ruff format --check .` → passes.

## Done when
- Defaults and README match the decision.
- manual: the owner reads the changed README sections.
