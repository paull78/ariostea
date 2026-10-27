# Step 09 · README, example config, ARCHITECTURE.md

**Milestone:** 4 · **Needs:** 03, 04, 05, 06, 07, 08 · **Covers:** AC-18

## Goal
A new user reading the README knows what the first run costs and does, and ARCHITECTURE.md shows the vault behind its port.

## Files touched
- `README.md`: quick start gains (vault-hardening.12): the first command downloads about 3.2 GB (bge-m3 2.1 GB, jina reranker 1.1 GB) into `~/.cache/fastembed`; run `uv run ariostea reindex` once from a terminal before an MCP client starts the server; the config is `ariostea.toml` in the current directory unless `--config` says otherwise (why `--directory` matters in the `claude mcp add` line); what the scanner skips (dot-folders, `.trash`, `*.excalidraw.md`, `vault.ignore`) and that unreadable files are named and skipped; a missing or empty vault leaves the index untouched; progress lines; measured cost (about 4.3 GB of memory for a running server, about 2.5 s per search on CPU, mostly the reranker). The "15 ms" sentence in "The defaults today" is labelled as query embedding time. The stale "160 tokens with overlap 40 is the default now" passage (around line 240) is reworded as history.
- `ariostea.example.toml`: `ignore` example shows `[]` with a comment on the built-in rule; no openai_compat lines (removed in step 03).
- `ARCHITECTURE.md`: the `Vault` port and its two adapters, `GetNote`, the dependency-rule test; the fingerprint paragraph names the chunker part; adapters subclass their ports (setup.02). Only those passages.
- Writing rules from `~/.claude/VOICE.md` via `/human-write-time` (neutral register, British spelling, no dashes as punctuation, sentence-case headings).

## Tests to write first
Exception: documentation. Checked by the verify command and by reading.

## Implementation notes
- Re-measure memory and per-search time on the synthetic vault before writing the numbers, if the gate run gives different figures than the hands-on test's 4.3 GB and 2.5 s.
- Keep the README's existing structure; add a short "First run" subsection under Quick start rather than spreading the facts around.

## Verify
`grep -n "3.2 GB\|~/.cache/fastembed\|excalidraw\|current directory" README.md` → each present; `uv run pytest tests/config -q` → the example config still loads.

## Done when
- manual: the owner reads the new Quick start and First run text.
