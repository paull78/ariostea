# Open issues

One entry per open issue found during the cycle, updated at every gate (general pin).

### vault-hardening.22 · issue · 2026-10-26 · gate M1
`docs/current/assets/make_vault.py` writes no note with a long whitespace-free string outside the Excalidraw drawing, which the vault now leaves out. The hands-on test's base64 web clip was added by hand at run time and is not in the generator, so step 10 (awkward-vault end-to-end test), ported from it, would never exercise the oversized-word path (AC-07, AC-08) end to end.
**Why:** found while checking the Milestone 1 gate run was safe to start: only `Boards/Sketch.excalidraw.md` had a word over 300 characters. Proposed fix: step 10's fixture adds a normal note with an inline base64 image of about 20,000 characters. Status: plan update approved at gate M1 (step 10 fixture gets the note).

### vault-hardening.23 · discovery · 2026-10-26 · gate M1
`test_control_vectors_match_plain_embed` built its reference `TextEmbedding` without `cache_dir`, so it used fastembed's temp-folder cache. macOS had purged that folder, the re-download got a Hugging Face 401, and the test failed on the full run while the adapter, reading `~/.cache/fastembed`, had the model all along. The test now passes `cache_dir=fastembed_cache_dir()`. Fixed at the gate, outside the plan; not a cycle regression.
**Why:** a full-suite failure at gate M1 traced to the test, not the code: 1 failed, 1172 passed. Rejected: re-running until the download worked (hides the same failure for the next run).

### vault-hardening.30 · issue · 2026-10-26 · step 09
`uv run ruff check .` fails on `docs/current/assets/q.py` (E401, I001), committed at gate 0 with the hands-on assets. `src`, `tests` and `eval` are clean. The assets move to `docs/archive/` at wrap and would keep failing there.
**Why:** found by the lint run in step 09. Proposed fix at wrap: exclude `docs/` from ruff in `pyproject.toml`, or leave the archived scripts as they ran. Status: open.
