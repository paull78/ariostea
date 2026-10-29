# Scope

### search-speed.01 · decision · 2026-10-28 · spec
The goal is latency: a search answers in under 1 s on the 50-note scratch vault, with no measurable loss in retrieval quality. Server memory (about 5 GB) is measured and reported, not a target.
**Why:** latency is felt on every call; nearly all of it is the reranker (vault-hardening.29). Rejected: both targets required (under 1 s and under 3 GB, wider scope, likely lazy loading or a reranker swap as well); memory first; a sweep first with targets set at a gate.

### search-speed.02 · decision · 2026-10-28 · spec
Quality bar for a faster configuration: at least 155 of 167 on the wiki HYBRID eval and no query type down more than 2 cases, against `2026-10-13-chunk-256t+64-bge-m3-norm-hybrid` (157 of 167). Ties go to span MRR.
**Why:** the bar bge-m3-default.02 used to rule out a loss; the set cannot resolve less than about 5 cases (embedding-sweep.18), so 2 cases is noise. Rejected: no loss at all (157, likely rules out most of the speed-up on noise); within 5 cases (152, could accept a real small loss).

### search-speed.04 · decision · 2026-10-28 · spec
Two knobs are in scope, both config only with no reindex: the rerank pool size and a quantised version of the same reranker (jina-reranker-v2, loaded from the ONNX files its repository ships through fastembed's custom-model registration).
**Why:** pool alone reaches 1 s only near 20 candidates (search-speed.03), which may fail the quality bar; quantisation keeps the model and its multilingual training and may buy room for a larger pool. Rejected: pool only (likely too few candidates); also cutting the text the reranker reads (it would no longer see the whole chunk); lazy loading (helps startup and idle memory, not search latency).
Superseded by search-speed.06.

### search-speed.06 · pivot · 2026-10-28 · spec
Supersedes search-speed.04. The knobs are the rerank pool size and the reranker model: jina-reranker-v2 as today, and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` loaded from its ONNX file through fastembed's custom-model registration. Both are config only, no reindex. Adopting the smaller model as the default is the owner's call at the end (setup.04).
**Why:** quantised jina is no faster on this machine (search-speed.05); mMiniLM-L12 is 3.4 times faster at the same pool and multilingual, so it gives a fallback if jina passes the bar only at too few candidates. Rejected: pool only (no fallback); cutting the text jina reads to 128 tokens (the reranker no longer sees the whole chunk).

### search-speed.07 · decision · 2026-10-28 · spec
Latency is measured inside the HYBRID wiki eval: every one of the 167 queries is timed through the production search path on the wiki index at the defaults, after one warm-up, on this M4 Pro. The target is the 95th percentile under 1 s; median and 95th percentile go into the run's `cost` record beside the existing `query_ms_median` (which times query embedding only).
**Why:** the wiki chunks are full length, so the pool fills as on a real vault, and one run gives recall and latency together. Rejected: median as the target (p95 is what a slow search feels like; with a full pool the two are close anyway); the scratch vault over stdio (short notes understate the cost, search-speed.03).

### search-speed.08 · decision · 2026-10-28 · spec
The sweep is 8 HYBRID wiki runs: pool 100, 50, 30 and 20 for jina-reranker-v2 and for mMiniLM-L12. Among configurations passing both bars (search-speed.02, search-speed.07), the one with the most hits is proposed; ties go to span MRR, then lower 95th percentile.
**Why:** quality first, speed only has to be good enough; jina at 100 is rerun to measure its latency the same way. Rejected: prefer jina whenever a jina pool passes (could give up hits for familiarity); fastest passing (speed beyond the target is worth less than a hit).

### search-speed.09 · decision · 2026-10-28 · spec
If no configuration passes both bars, the defaults stay and the recap shows the recall and latency table; the owner picks a trade-off as a config change or a later cycle.
**Why:** nothing unmeasured or below the bar ships by default. Rejected: proposing the fastest configuration that passes quality even above 1 s (a partial win the owner can still choose from the table); widening the knobs with shorter input (already rejected in search-speed.06).

### search-speed.10 · decision · 2026-10-28 · spec
server-robustness.14 (null status on an upgraded index until its first run) stays out of this cycle and stays open.
**Why:** unrelated to search speed, and one reindex clears it. Rejected: folding in a migration step that records a completed run from the old fingerprint.

### search-speed.11 · decision · 2026-10-28 · spec
Testing: fast unit tests for the latency helper (injected clock), the run record's latency fields and the config; one integration test loading mMiniLM-L12 through the real adapter into an empty cache directory; the 8 logged eval runs are the evidence for quality and latency. Test command `uv run pytest` plus ruff.
**Why:** the risky new code is fetching and registering a custom reranker (search-speed.05 pitfall); latency itself is a measurement, not a regression test. Rejected: a latency guard test (timing tests are flaky across machines and load); eval runs only (leaves the fetch path untested).

### search-speed.12 · decision · 2026-10-28 · spec
`search_sources` follows the pool: it groups at most `rerank.pool` reranked chunks by note, as today with 100. Each sweep run records for how many of the 167 queries it returns fewer than 10 notes, and the README states the bound.
**Why:** one setting, and the notes tool gets the same speed-up. Rejected: a fixed 50-candidate pool for `search_sources` (keeps it slow, 2.3 s with jina); reranking only the top of a 50-chunk fused list and appending the rest (two orderings mixed in one list).

### search-speed.13 · decision · 2026-10-28 · challenge round
mMiniLM-L12 runs from its fp32 `onnx/model.onnx`, not the `model_qint8_arm64` file.
**Why:** simple-made-easy lens: the arm64 file ties the model choice to the CPU type on a project that runs on macOS and Linux, and its int8 scores differ from the published model. fp32 takes 0.95 s at pool 50 against 0.67 s (search-speed.05), still under 1 s at the pools likely to pass. Rejected: the int8 arm64 file; measuring both (four more runs).

### search-speed.14 · decision · 2026-10-28 · challenge round
Each sweep run builds its container and runs its queries inside the existing `fail_on_warnings` guard (`src/ariostea/eval/blurb_eval.py`), so a reranker that falls back to fused order aborts the run before it is logged.
**Why:** hexagonal lens: the container degrades on purpose and only warns, so without a guard a failed mMiniLM load would be logged as mMiniLM with fused-order numbers (about 24 fewer cases). Reuses code that exists. Rejected: asserting the reranker's concrete type (reaches past the port, setup.01); no guard.

### search-speed.15 · decision · 2026-10-28 · plan
The sweep indexes the wiki corpus once and builds one container per (model, pool) over the same database, in a new runner `eval/run_rerank_sweep.py`, instead of reusing `eval/run_chunk_sweep.py`, which re-indexes per run. Plan approved at Gate 1 in sequential mode; the owner said "go go go" to the outline, read as the yes at Gate 1 and at the Milestone 1 gate. The Milestone 2 gate still waits for the owner, because it carries the default decision (setup.04).
**Why:** the reranker does not touch the index, so 8 runs take about 40 minutes instead of about 2 hours (334 s indexing each); the second-container trick already exists in `fused_channel`. Rejected: extending the chunk sweep with reranker flags (one index per run).

### search-speed.21 · decision · 2026-10-28 · step 05
Proposed default: reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `rerank.pool = 30`. It is the only configuration that passes both bars (search-speed.02, search-speed.07): 156 of 167, per type 39, 40, 40, 37 against 40, 41, 39, 37, p95 707 ms. The pick rule (search-speed.08) has a single candidate, so no tie-break applies. Cost the owner sees: span MRR 0.867 → 0.833, and `search_sources` returns fewer than 10 notes for 101 of 167 queries (54 today). Waits for the owner's yes or no (search-speed.15, setup.04).
**Why:** rule applied as written. Rejected: mMiniLM at pool 50 (p95 1059 ms, fails the latency bar), jina at pool 20 (151 hits and p95 1199 ms, fails both), mMiniLM at pool 20 (153 hits, fails the quality bar).

### search-speed.22 · decision · 2026-10-29 · gate M2
The owner approved search-speed.21: the defaults become `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` with `rerank.pool = 30`. `search_sources` keeps following the pool (search-speed.12), and the README states its real bound, min(50, `rerank.pool`) chunks (search-speed.20).
**Why:** p95 4623 ms → 707 ms for 1 hit of 167 and span MRR 0.867 → 0.833; the change needs no reindex and is two lines of config to undo. Rejected: keeping jina at pool 100 (misses the goal by 3.6 s); a separate pool of 50 for `search_sources` now (a spec amendment for a cost the owner accepts; kept as search-speed.23).

### search-speed.25 · decision · 2026-10-29 · final gate
The README splits in two. `README.md` is the product page (purpose, tools, example, quick start, running the index, development, licence); a new `LEARNING.md` next to `ARCHITECTURE.md` holds the architecture rationale, the eval instrument and Trials 1 to 5, linked from the README. The review's eight corrections go in with the move, and `docs/retrieval-tuning.md` gains a search-speed section so the learning page's "full account" link holds.
**Why:** the owner asked for it at the final gate's README check: at 470 lines the README buried "Connecting an MCP client" at line 216 and the learning journey was 44% of it. Rejected: trimming Trial 5 and reordering inside one file (keeps one page doing two jobs).
