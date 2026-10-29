# Recap: search-speed

Feature cycle on `feature/search-speed`, started on 2026-10-28 and finished on 2026-10-29. Source: a search took about 2.5 s on the 50-note scratch vault and the server held about 5 GB, nearly all of both in the reranker (vault-hardening.29).

## What was built

A search now answers in about 0.6 s instead of 2.5 s on the scratch vault, and its 95th percentile on the eval set fell from 4,623 ms to 707 ms. The defaults changed to the `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` reranker with `rerank.pool = 30`, approved by the owner at the Milestone 2 gate (search-speed.22). The change needs no reindex, and the old reranker is two lines of config away.

- `eval/run_rerank_sweep.py` measures a list of reranker configurations, written `MODEL:POOL`, over one index of the wiki corpus. For each it logs HYBRID span scores, the median and 95th percentile time of a search through the production path, and how many of the 167 queries get fewer than 10 notes from `search_sources` (search-speed.07, .15). A configuration whose reranker falls back to fused order aborts before it is logged (search-speed.14). The helpers live in `src/ariostea/eval/rerank_sweep.py`.
- The reranker adapter registers mMiniLM-L12 with fastembed from its fp32 ONNX file before loading it, and an integration test loads it into an empty cache directory and ranks an English and an Italian answer first (search-speed.13, AC-03).
- The 8 sweep runs are in `eval/results/runs.jsonl` and the log artifact.
- The README is now a product page, and a new `LEARNING.md` holds the architecture rationale, the eval set and Trials 1 to 5 (search-speed.25). `docs/retrieval-tuning.md` gained a search-speed section with the full table and the commands to reproduce it.

Deviations from the spec and plan:

- The spec's knobs changed before the plan: quantised jina was dropped for a second, smaller model (search-speed.06).
- `search_sources` groups at most 50 passages or `rerank.pool`, whichever is fewer, not `rerank.pool` as search-speed.12 assumed: `SearchSources` asks for 50 passages of its own (search-speed.20). The README states the real bound, which still satisfies AC-06.
- Step 07 (README split) was added at the final gate, when the owner's review of the README turned into a split.

## How it went

Three milestones. Milestone 1 (the sweep skeleton on today's reranker) passed its gate on 2026-10-28 with 1,344 passed and 1 skipped in 26 minutes. Milestone 2 (the full sweep and the pick) passed on 2026-10-29 with 1,347 passed and 1 skipped in 29 minutes, and carried the owner's yes on the default. The final gate passed on 2026-10-29 with 1,348 passed and 1 skipped in 16 minutes; the integration tests got faster with the smaller reranker.

The cycle started on the belief that a smaller pool, or a quantised jina, would reach 1 s. A cost measurement during the spec showed jina at about 46 ms per 256-token candidate, so a pool alone reaches 1 s only near 20 candidates (search-speed.03). A quick test then showed that jina's int8 files are no faster on this machine, its fp16 file does not load, and the CoreML provider is far slower, while mMiniLM-L12 is 2.4 to 3.4 times faster (search-speed.05). The second knob became the model (search-speed.06). The sweep bore this out: jina never got under 1 s at a passing pool, and only mMiniLM-L12 at pool 30 passed both bars, with 156 of 167 and a 95th percentile of 707 ms (search-speed.19, .21).

The control reproduced exactly through the new runner, 157 of 167 with the same per-type counts (search-speed.17), which settled the open risk at Gate 0. The step 04 integration test found that the expired Hugging Face token on this machine blocks the download of any model not cached yet (search-speed.18).

## Decisions that stand

- search-speed.02 · A faster configuration must find at least 155 of 167 with no query type down more than 2 cases · 2 cases is below what the set can resolve.
- search-speed.07 · Search latency is measured inside the HYBRID wiki eval, 95th percentile over the 167 queries after one warm-up · wiki chunks fill the pool as a real vault does, and one run gives recall and time together.
- search-speed.12 · `search_sources` follows the rerank pool · one setting, and the notes tool gets the same speed-up.
- search-speed.13 · mMiniLM-L12 runs from its fp32 ONNX file · the arm64 int8 file would tie the model to the CPU type.
- search-speed.14 · Sweep runs build their container inside the `fail_on_warnings` guard · a reranker that falls back to fused order would otherwise log fused numbers under its own name.
- search-speed.15 · The rerank sweep indexes once and builds one container per configuration · the reranker does not touch the index, so 8 runs take about an hour instead of several.
- search-speed.22 · Defaults: mMiniLM-L12 reranker, pool 30 · 95th percentile 4,623 ms to 707 ms for 1 hit and span MRR 0.867 to 0.833.
- search-speed.25 · README is the product page, `LEARNING.md` the learning journey · one page was doing two jobs.

## Discoveries

- The reranker's cost is linear in candidates and chunk length: jina-v2 takes about 46 ms per 256-token candidate on this M4 Pro (search-speed.03).
- Quantised jina buys nothing on this machine; mMiniLM-L12 is 2.4 to 3.4 times faster than jina at the same pool. fastembed downloads only the files of a model it registered first, so a custom model's extra files must be fetched explicitly or the cache cleared (search-speed.05).
- `search_sources` already came back short of 10 notes for 54 of 167 queries at pool 100, because the wiki notes are long and its 50 passages come from few notes (search-speed.17, .20).
- On the scratch vault the new default holds 2.75 GB after the first search and 3.13 GB after seven, and one query of seven ranks its note second where jina put it first (search-speed.24).

## Open issues and known limitations

- search-speed.23 · At pool 30, `search_sources` returns fewer than 10 notes for 101 of 167 queries. A separate pool of 50 for that tool would cost about 1 s per call with mMiniLM.
- search-speed.18 · The expired Hugging Face token on this machine makes any uncached model download fail with a 401. The owner can refresh it with `hf auth login` or remove it; the tests and sweep work around it with `HF_HUB_DISABLE_IMPLICIT_TOKEN`.
- server-robustness.14 stays open (search-speed.10): an upgraded index shows null status until its first run.
- Span MRR is 0.833 against 0.867: the right passage sits lower in the top 5 more often.

## Takeaways

- Measuring the cost model before the interview ended (milliseconds per candidate, then a ten-minute quantisation test) replaced a knob that would have failed and pointed at the one that worked, before any plan was written.
- Indexing once and building a container per configuration made an 8-run sweep a one-hour job, so the whole grid was measured instead of a guess at the likely pool.
- search-speed.12 assumed how `search_sources` chose its passages without reading `SearchSources`; reading the code during the spec would have stated the bound right from the start.
- The README had grown into two documents by the time it was reviewed. Checking its length and structure when a step adds a trial would have split it earlier.
