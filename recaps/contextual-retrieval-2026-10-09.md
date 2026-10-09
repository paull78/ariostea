# Recap: contextual retrieval measured, both ways — 2026-10-09

> Audience: a future session resuming cold. Assumes you can read the code and run the repo, but were not in the conversation. Continues `retrieval-eval-and-chunking-2026-09-28.md`.

## Where things stand
- **Branch:** `master`. `feat/blurb-aware-rerank` merged locally as `9e673e1` (no PR), pushed to origin, branch deleted. No feature branch is open.
- **Working tree:** the default-embedding change (below) plus this recap, uncommitted at the time of writing; check `git status`.
- **Tests:** fast suite 1,044 passed at `9e673e1`. Full suite result: see the commit that lands the embedding default.
- **Next concrete step:** an embedding-model sweep aimed at cross-lingual (see "Open questions").

## Goal
Test whether contextual retrieval pays on the wiki gold set, now that chunking is settled. Two cycles: note-level blurbs with a blurb-aware reranker (`docs/design/2026-09-29-blurb-aware-reranking.md`), then Anthropic-style per-chunk context (`docs/design/2026-09-29-per-chunk-context.md`). Full account: `docs/retrieval-tuning.md`, sections "Contextual blurbs" through "Default embedding model".

## Results, hybrid span recall at 5 (what the server runs)
| configuration | overall | cross_lingual |
|---|---|---|
| no context (control) | 0.880 | 0.696 |
| note blurbs, reranker on raw text | 0.844 | 0.609 |
| note blurbs, reranker sees blurb | 0.814 | 0.609 |
| per-chunk context, reranker on raw text | 0.850 | 0.630 |
| per-chunk context, reranker sees context | 0.856 | 0.652 |

All four fail the decision rule (+0.03 overall, no type below −0.05). Per-chunk context does lift the first stage (dense 0.545 → 0.581, fused 0.629 → 0.665) and removes the within-article convergence that note blurbs caused, but the reranker already recovers what the context adds.

## Decisions made (and why)
- **Contextual indexing stays off, `rerank.use_context` stays off, `contextual.granularity = "chunk"` stays in the code off by default.** Nothing measured beats the plain hybrid baseline. Memory: `contextual-lift-measured`.
- **The default embedding model is now `paraphrase-multilingual-mpnet-base-v2`** (`EmbeddingCfg.local_model`, the adapter default, `ariostea.example.toml`). Every measurement used it and the chunking default was sized with its tokenizer; shipping `bge-small-en-v1.5` meant shipping unmeasured numbers. Owner approved on 2026-10-09. Existing vaults re-embed once (fingerprint change).
- **Per-chunk context was generated with `qwen2.5-7b-instruct` (MLX 4-bit), not the 14B.** The owner chose a smaller local model over the Haiku API after two kernel panics. The write-up says so and flags that a stronger model is untested.
- **Merged locally, no PR,** at the owner's request; earlier branches went in through PRs #2–#4. Still `--no-ff`, so the history reads the same.
- **The blurb's per-chunk duplication is accepted.** About 1.5 MB per 4,246 chunks plus an FTS copy, 10–15% of the DB; documented in the blurb design's "Storage and memory".

## Gotchas
- **The 14B model at a 32k context panics this 48 GB Mac.** Two overnight runs (2026-10-01, 2026-10-02) ended in `IOGPUGroupMemory::remove_memory_object()` kernel panics after Metal out-of-memory errors; panicking task `node` (LM Studio). The 7B 4-bit ran the same job in 3.5 h without trouble. Reports in `/Library/Logs/DiagnosticReports/panic-full-2026-10-0{2,4}-*.panic`.
- **LM Studio reuses the prompt prefix only with `lms load --parallel 1`.** With more slots every per-chunk call re-reads the whole article; the long article timed out. The launcher (`eval/run_chunk_context_overnight.sh`) loads the model that way and checks `lms ps`.
- **LM Studio's JIT load uses the wrong settings** (8k context, 4 slots). The runner's preflight reads `/api/v0/models/<model>` and aborts unless the model is already loaded with a 32k context.
- **Python buffers stdout when redirected.** The Oct 2 log lost every progress line to the crash; the launcher now runs `python -u`.
- **`CachingChat` stores successes only, keyed by label + system + user,** so a changed prompt or model regenerates everything. Contexts from one model cannot be mixed into a run with another; the 524 contexts cached from the 14B were useless for the 7B run.
- **The Claude Code permission classifier blocks `rm` of eval indexes and sometimes misreads a `grep` batch as destructive.** Delete leftover indexes by hand.
- **`lms get` from Hugging Face timed out at 94% and a retry reported "already in progress".** LM Studio's own service finishes the download; wait for the `.part` file to disappear.
- **`ls` is aliased to eza here;** use `/bin/ls` in scripts.

## Rejected options
- **Haiku 4.5 API for per-chunk context** (about $7–8, 1–2 h). Offered after the second panic; the owner chose a smaller local model. There is no Anthropic chat adapter, only `openai_compat.py`.
- **Moving the blurb column from `chunks` to `notes`.** The duplication is 10–15% of the DB and the query-time cost is nil.
- **Mixing the 14B contexts into the 7B run.** Different model, different cache label; the run must be one model throughout.
- **Regenerating the gold set.** It would make every past run incomparable, and no result needs it.

## Open questions
- **Cross-lingual is the weak spot:** hybrid 0.696 on 46 cases against 0.95+ elsewhere; sparse 0.022 there. Both context experiments hurt it most. Next experiment: an embedding-model sweep (`bge-m3`, `multilingual-e5-large`) with a runner like the chunking sweep, dense and hybrid, scored against the 160t+40 controls. About an hour per model for hybrid. Note that `MULTILINGUAL_MODEL` in `src/ariostea/eval/wiki_index.py` is pinned; the sweep needs a way to override it, and the chunker's token sizing follows the model's tokenizer.
- **Gold set drift:** unchanged from the previous recap. Gold regeneration would use the new chunking and embedding defaults.
- **Per-chunk context without the reranker** gains +0.036 (fused). Worth a look only if a reranker-free configuration ever matters.
- **Leftover indexes** in `eval/results/indexes/` (`blurbs-160t+40.db`, `chunk-context-160t+40.db`, `.prev.db`, `chunk-context-cache.jsonl`), about 60 MB, gitignored, safe to delete.

## Pointers
- `docs/retrieval-tuning.md`: results tables, finding/evidence/reason tables, reproduction commands.
- `eval/run_blurb_eval.py`: both experiments (`--granularity note|chunk`, `--preview`, `--reuse-index`, `--arms`).
- `eval/run_chunk_context_overnight.sh`: timed launcher; `ARIOSTEA_CTX_MODEL` picks the model.
- Contexts and blurbs: `eval/results/logs/2026-09-29-blurbs.json`, `eval/results/logs/2026-10-06-chunk-contexts.json`.
- Run ids: `2026-09-29-blurbs-{raw-rerank,fused,context-rerank}`, `2026-10-06-chunkctx-{raw-rerank,fused,context-rerank}`. Controls: `2026-09-24-chunk-160t+40-dense-hybrid-sparse`, `2026-09-28-chunk-160t+40-fused`.
- Log artifact: https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid (version 8). Republish after every run; from a new conversation, pass it as `url`.
- Memories: `contextual-lift-measured`, `per-chunk-context-pilot`, `experiment-log`, `chunking-policy-decisions`, `personal-folder-uses-paull78-gh`.

## How to resume
1. `git pull` on master; `lms ps` should show no loaded model before any eval.
2. Read memory `contextual-lift-measured` and the "Open questions" above.
3. For the embedding sweep: design → plan → implement as its own cycle, like chunking. Start from `eval/run_chunk_sweep.py` and `wiki_index.py`; add the model to the run config and the index fingerprint; log against the 160t+40 controls; republish the artifact.
