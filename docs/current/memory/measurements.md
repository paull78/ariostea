# Measurements

### search-speed.03 · discovery · 2026-10-28 · spec
The reranker's cost is linear in candidates and chunk length. On this M4 Pro, jina-reranker-v2 takes about 46 ms per 256-token candidate: 4.6 s for the default pool of 100, 2.3 s for 50, 1.1 s for 20; at 128 tokens 2.7 s, 1.2 s and 0.6 s (`docs/current/assets/rerank_cost.py`, README text cut to exact token lengths). So on a vault whose notes fill the pool with full chunks a search takes about 4.7 s, not the 2.5 s measured on the 50-note scratch vault, whose short notes give fewer and shorter candidates. Turning the reranker off is not an option on quality: at 160t+40 the fused order finds 131 of 167 against 155 with the reranker (`2026-10-12-chunk-160t+40-bge-m3-norm-*`).
**Why:** the scratch vault understates the cost the goal targets, and pool size alone reaches 1 s only near 20 candidates; this shapes the knobs and where latency is measured.

### search-speed.05 · discovery · 2026-10-28 · spec
Faster ways to run jina-reranker-v2 buy nothing on this M4 Pro (onnxruntime 1.26, 50 candidates of 256 tokens, 2.3 s in fp32): the repository's `model_int8` and `model_quantized` files take 2.3 s too (rank correlation 0.985 with fp32, top 5 overlap 4 of 5); `model_fp16` fails to load on the CPU provider; the CoreML provider takes 163 s (dynamic shapes) or fails. A smaller multilingual cross-encoder, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, ONNX in its repository), takes 0.95 s, and its `model_qint8_arm64` file 0.67 s, so 2.4 to 3.4 times faster; its quality is unmeasured. Scripts in `docs/current/assets/` (`quant.py`, `coreml.py`, `minilm.py`). Pitfall: fastembed downloads only the files of a model it registered first, so an extra ONNX file in an already-cached repository (and a custom model's `config.json` and tokenizer files) must be fetched explicitly.
**Why:** removes quantised jina as a knob (search-speed.04 assumed it would buy headroom) and puts a smaller model in its place as a candidate for the owner's call.

### search-speed.16 · discovery · 2026-10-28 · step 02
The runner's unlogged end-to-end check, jina-reranker-v2 at pool 20 (`docs/current/assets/nolog-jina-pool20.txt`): HYBRID 151 of 167 (span recall 0.904; buried 38, cross_lingual 41, exact_term 37, paraphrase 35), span MRR 0.844, search median 1,141 ms and p95 1,192 ms, so it fails both bars. `search_sources` returns fewer than 10 notes for 153 of 167 queries at pool 20: the 20 chunks cover too few notes, so following the pool (search-speed.12) costs that tool its breadth at small pools. Indexing took 341 s.
**Why:** the first evidence on the pool knob: jina alone does not reach 1 s with a passing pool, which makes the mMiniLM arm the likely path, and the short count is large enough that the owner should see it beside the pick at the Milestone 2 gate.

### search-speed.17 · discovery · 2026-10-28 · step 03
The control reproduces exactly through the new runner (`2026-10-28-rerank-jina-reranker-v2-base-multilingual-pool100`): HYBRID 157 of 167, per type 40, 41, 39, 37, span MRR 0.867, as in `2026-10-13-chunk-256t+64-bge-m3-norm-hybrid`, although the runner indexes with the production default `normalize = false` (bge-m3 returns unit vectors of its own). A search takes 4,009 ms median and 4,623 ms at the 95th percentile. `search_sources` already returns fewer than 10 notes for 54 of 167 queries at pool 100: the wiki notes are long, so its 50 chunks often come from few notes. Indexing 336 s. Log artifact republished (version 11).
**Why:** settles the open risk at Gate 0 (the runner measures what the chunk sweep did), gives the latency baseline for AC-02, and the short-count baseline for AC-06.

### search-speed.19 · discovery · 2026-10-28 · step 05
The full sweep, 8 HYBRID runs on the wiki gold set at 256t+64, one index for all, the control at the top. Types are buried, cross_lingual, exact_term, paraphrase; the control has 40, 41, 39, 37.

| Reranker, pool | Hits of 167 | Per type | Span MRR | Median ms | p95 ms | `search_sources` short |
|---|---|---|---|---|---|---|
| jina-v2, 100 (control) | 157 | 40, 41, 39, 37 | 0.867 | 4009 | 4623 | 54 |
| jina-v2, 50 | 156 | 40, 41, 39, 36 | 0.862 | 2340 | 2426 | 35 |
| jina-v2, 30 | 157 | 40, 42, 39, 36 | 0.863 | 1585 | 1634 | 101 |
| jina-v2, 20 | 151 | 38, 41, 37, 35 | 0.844 | 1146 | 1199 | 153 |
| mMiniLM-L12, 100 | 154 | 39, 38, 40, 37 | 0.828 | 1628 | 1942 | 36 |
| mMiniLM-L12, 50 | 154 | 39, 38, 40, 37 | 0.831 | 1003 | 1059 | 35 |
| mMiniLM-L12, 30 | 156 | 39, 40, 40, 37 | 0.833 | 576 | 707 | 101 |
| mMiniLM-L12, 20 | 153 | 37, 42, 38, 36 | 0.824 | 360 | 418 | 153 |

The 29 discriminated-out cases score 1.000 in every run. Only mMiniLM-L12 at pool 30 passes both bars. jina never gets under 1 s with a passing pool, and mMiniLM at pool 50 misses by 59 ms.
**Why:** the evidence for AC-01, AC-02 and AC-04, and the table the owner decides on at the Milestone 2 gate.

### search-speed.20 · discovery · 2026-10-28 · step 05
`search_sources` asks the searcher for 50 chunks (`SearchSources(pool=50)`, not wired to config), so it groups at most min(50, `rerank.pool`) chunks, not `rerank.pool` as search-speed.12 assumed. At pool 100 the reranker keeps the best 50 of 100, which sit on fewer notes, so it comes back short more often (54) than at pool 50 (35). At pool 30 and 20 the pool is the bound (101 and 153 short).
**Why:** the README sentence in AC-06 ("sees at most `rerank.pool` chunks") stays true but loose; step 06 (defaults and README) states the real bound, min(50, `rerank.pool`).
