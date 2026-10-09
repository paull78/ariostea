# Recap: wiki gold set, experiment log, chunking default — 2026-09-28

> Audience: a future session resuming cold. Assumes you can read the code and run the repo, but were not in the conversation.

## Where things stand
- **Branch:** `master` at `92ccca2` (PR #4 merged). Feature branches `feat/eval-gold-generation` and `feat/chunking-policy` are deleted locally and on origin.
- **Working tree:** clean except this recap (uncommitted).
- **Tests:** full suite incl. integration passed at `d2fb2e7`: 978 passed, 1 skipped, about 12 min.
- **Next concrete step:** start the contextual-blurbs experiment. First make the reranker score the blurb, not only raw `chunk.text` (see memory `contextual-lift-measured`).

## Goal
Build an eval that can detect retrieval changes (span-anchored gold over a pinned Wikipedia corpus), then use it for the roadmap in `docs/design/2026-07-09-eval-corpus-expansion.md` ("What this unlocks"): chunking (done), contextual blurbs (next), BM25 tuning, graph retrieval. Full chunking account: `docs/retrieval-tuning.md`.

## Decisions made (and why)
- **Chunking default is now 160 model tokens, overlap 40** (`src/ariostea/config/schema.py`, `ChunkingCfg`). It was the only policy that passed the pre-set rule with the reranker (+0.048) and without it (+0.078). The owner approved it, framed as a fix for silent truncation rather than a tuning win.
- **Decision rule, pre-registered:** at least +0.03 hybrid span recall, no type below −0.05, at least 160/167 spans reachable. The owner accepted it. 0.03 × 167 ≈ 5 cases is the smallest resolvable change.
- **Default change was the owner's call, not automatic.** They said it "depends on how big the improvement is". Keep asking before changing defaults that force a reindex.
- **Default chunker's fingerprint is `""`**, and `IndexVault._fingerprint` skips empty parts. That way indexes built with the original 512-word policy keep their stored `emb|ctx` fingerprint instead of being forced to reindex.
- **FUSED channel is opt-in and kept out of `wiki_channels`.** Gold generation's discrimination filter drops a case only when every channel answers it at rank 1, so adding a channel there would change what the filter keeps.
- **Experiment log is the record of every eval run** (`eval/results/runs.jsonl`). The owner explicitly wants a graphic, green-shaded history of all tests. See memory `experiment-log`.
- **Gains are green, losses magenta, not red.** Green against red fails for the commonest colour blindness, and every cell also prints its signed delta.
- **Log page is pure ASCII.** The published artifact showed mojibake because the host decoded UTF-8 as Latin-1. Tests enforce ASCII for the template and the rendered output.
- **Spot review rejected 2 of 19 gold cases.** One was an exact_term case with an English query over a Spanish span; the other a chess capture question that `chess.md` answers as well. Both are moved to `gold_rejected.json` under stage `spot-review`.
- **Unreadable judge verdicts get stage `judge-unreadable`**, not `ambiguity`. A parse failure is the judge's fault, not a finding about the case.

## Gotchas
- **LM Studio plus the eval pipeline exhausts memory.** The 20 GB judge sits resident with a 183k context. Run `lms unload --all` before retrieval stages, and JIT reloads it on the first judge call. One gold run was killed at 26 GB swap before this.
- **The Claude Code memory watchdog kills background tasks.** Launch long runs with `nohup … & disown`. `setsid` doesn't exist on macOS.
- **`/private/tmp` scratch files get purged between sessions.** That lost the baseline eval log once; it was restored from conversation output and cross-checked. The sweep runner now writes raw output to `eval/results/logs/`.
- **Monitors: `pgrep -f <script>` matches the monitor's own shell.** It never exits. Watch a saved PID with `kill -0` instead.
- **`$!` from `nohup uv run …` is the `uv` wrapper,** which shows 0% CPU. The real Python worker is its child (`pgrep -P <pid>`).
- **Monitors expire after 30 min** even with `persistent: true`. Re-arm them, or pair them with a `ScheduleWakeup` fallback.
- **Competitor collection and HYBRID rerank 100 candidates per query on CPU,** about 1 hour per chunking config for hybrid. Dense and sparse take about 2–3 min.
- **Hybrid masks retrieval defects.** Fused without the reranker scores 0.551 on the old default versus 0.832 with it. Always check FUSED or DENSE too before concluding "no effect".
- **The gold set is biased toward what 512-word chunking found hard.** Discrimination and ambiguity gates ran against that index. The sweep also scores the 29 discriminated-out cases (`load_discriminated`) to catch policies that lose easy cases.
- **The default overlap of 40 makes `max_tokens ≤ 40` invalid unless overlap is set too.** The error message now says so.
- **The log page's `experiments` URL param splits on `|`,** because experiment names contain commas ("Fused, no reranker").
- **Figure regeneration:** serve `eval/results` with `python3 -m http.server`, since Playwright blocks `file:`. Open `experiment_log.html?view=table&channels=…&experiments=…&caption=…` and capture `.page` at deviceScaleFactor 2. Recipe in `docs/retrieval-tuning.md` → Reproducing.
- **Model-token sizing needs a local embedding adapter with `count_tokens`.** The container always builds `FastEmbedEmbeddings` today; an OpenAI-compatible provider would raise in `build_chunker`.
- **Production default embedding is `bge-small-en-v1.5`,** but all chunking numbers were measured with the owner's `paraphrase-multilingual-mpnet-base-v2`.

## Rejected options
- **Reusing the discrimination gate's hybrid searches for competitor collection.** The gate short-circuits on the first channel that misses, so the saving was near zero.
- **Adopting 160t+40 on the hybrid result alone.** The pass looked lucky: neighbours scored +0.012 and +0.006. The 192t+48 confirmation scored +0.018, so the FUSED measurement decided it.
- **192t+48 as default.** Best without the reranker (+0.090), but +0.018 with it fails the rule.
- **Keeping 512 words.** Briefly recommended, then withdrawn. It is a truncation defect, and the reranker-off fallback path pays for it.
- **Adding FUSED to `wiki_channels`.** It would silently change gold regeneration.

## Open questions
- **Contextual blurbs:** does fixing the reranker to see the blurb make blurbs worth their LLM cost on hybrid? Not started.
- **Gold set drift:** gold regeneration now uses the new chunking default via `wiki_config`. A regenerated set would differ from the committed one, which was filtered on 512-word chunks. There's no decision yet on when or whether to regenerate.

## Pointers
- `docs/retrieval-tuning.md`: full chunking account, figures, reproduction commands.
- `docs/design/2026-09-24-chunking-policy.md`, `docs/plans/2026-09-24-chunking-policy.md`: design and plan. The plan corrects the design: the `Chunker` port gained `fingerprint`.
- `eval/run_chunk_sweep.py`: specs like `160t+40`, plus `--channels`, `--experiment`, `--control`.
- `eval/results/runs.jsonl`, `eval/render_results.py`, `eval/results/template.html`: the experiment log.
- Log artifact: https://claude.ai/artifact/3yrQLivHKFcMRnpsUi4yid. Republish after every new run; from a new conversation, pass it as `url`.
- PRs: #3 (gold set, merged), #4 (chunking, merged).
- Baseline control run id: `2026-09-06-baseline`. Fused control: `2026-09-28-chunk-512w-fused`.
- Memories: `experiment-log`, `chunking-policy-decisions`, `contextual-lift-measured`, `personal-folder-uses-paull78-gh`.

## How to resume
1. `git pull` on master; confirm `lms ps` shows no loaded model before any eval.
2. Read memory `contextual-lift-measured` and `src/ariostea/adapters/rerank/fastembed_rerank.py` to see what text the reranker scores.
3. Design → plan → implement the blurb-aware reranking, as its own cycle like chunking.
4. Measure with `eval/run_chunk_sweep.py` and log it. Contextual indexing needs the `contextual` config, which the sweep runner doesn't expose yet. Republish the log artifact afterwards.
