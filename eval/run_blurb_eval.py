"""Measure note-level context blurbs, and blurb-aware reranking, on the wiki gold set.

Usage:
    uv run python eval/run_blurb_eval.py               # build the blurbed index, score, log
    uv run python eval/run_blurb_eval.py --reuse-index # score the index built last time

Indexes the corpus once at the production chunking default with contextual
indexing on, aborts unless every note got a blurb, then scores:
  arm 2  DENSE, SPARSE, HYBRID with the reranker on raw text, and FUSED
  arm 3  HYBRID with the reranker scoring blurb plus text
Arm 1 (no blurbs) is already logged; each entry names it as its control.
See docs/design/2026-09-29-blurb-aware-reranking.md.

The blurb LLM comes from:
    ARIOSTEA_CTX_BASE_URL  (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_CTX_MODEL     (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_CTX_API_KEY   (default empty)
    ARIOSTEA_CTX_TIMEOUT   (seconds, default 300: whole articles, local model)
Load the model with a context window of at least 32k tokens first.

Blurbing takes 20 to 40 minutes and each HYBRID pass about an hour, so the
blurbed index is kept under eval/results/indexes/ for --reuse-index.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

from ariostea.config.container import build_container
from ariostea.config.schema import ChunkingCfg, ContextualCfg
from ariostea.eval.blurb_eval import (
    ARMS,
    BlurbCoverageError,
    blurb_run_id,
    require_full_coverage,
)
from ariostea.eval.chunk_sweep import load_discriminated
from ariostea.eval.contextual import read_blurb_rows
from ariostea.eval.results_log import (
    append_run,
    code_commit,
    load_runs,
    make_run,
    render_html,
    report_to_dict,
)
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import load_wiki_gold
from ariostea.eval.wiki_index import (
    CHUNK_POOL,
    MULTILINGUAL_MODEL,
    context_rerank_channel,
    fused_channel,
    index_wiki_corpus,
    wiki_channels,
    wiki_config,
)

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
INDEX = RESULTS / "indexes" / "blurbs-160t+40.db"
K = 5


def _contextual() -> ContextualCfg:
    return ContextualCfg(
        enabled=True,
        base_url=os.environ.get("ARIOSTEA_CTX_BASE_URL", "http://localhost:1234/v1"),
        model=os.environ.get("ARIOSTEA_CTX_MODEL", "qwen2.5-14b-instruct-mlx"),
        api_key=os.environ.get("ARIOSTEA_CTX_API_KEY", ""),
        timeout=float(os.environ.get("ARIOSTEA_CTX_TIMEOUT", "300")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reuse-index", action="store_true", help="score the kept index")
    parser.add_argument("--experiment", default="Contextual blurbs", help="group name in the log")
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    args = parser.parse_args(argv)

    today = dt.date.today().isoformat()
    runs = {run["id"]: run for run in load_runs(RUNS)}
    if not args.no_log:
        for arm in ARMS:
            if arm.control not in runs:
                parser.error(f"control run {arm.control!r} is not in {RUNS}")
            if blurb_run_id(today, arm) in runs:
                parser.error(f"already logged today: {blurb_run_id(today, arm)}")

    ctx = _contextual()
    if args.reuse_index:
        if not INDEX.exists():
            parser.error(f"no kept index at {INDEX}; run without --reuse-index first")
        print(f"reusing {INDEX}", flush=True)
        container = build_container(wiki_config(WIKI, str(INDEX), contextual=ctx))
    else:
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        INDEX.unlink(missing_ok=True)
        print(f"indexing with blurbs from {ctx.model} ...", flush=True)
        container = index_wiki_corpus(WIKI, str(INDEX), contextual=ctx)

    try:
        blurbed = require_full_coverage(read_blurb_rows(str(INDEX)))
    except BlurbCoverageError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        return 1
    print(f"blurb coverage {blurbed}/{blurbed} notes", flush=True)

    cases = load_wiki_gold(WIKI / "gold.json")
    easy = load_discriminated(WIKI / "gold_rejected.json")
    channels = wiki_channels(str(INDEX), container)
    channels["FUSED"] = fused_channel(container)
    channels["HYBRID+CONTEXT"] = context_rerank_channel(container)

    commit = code_commit()
    for arm in ARMS:
        lines: list[str] = []

        def say(text: str) -> None:
            print(text, flush=True)
            lines.append(text)

        say(f"\n##### {arm.label}")
        scores: dict[str, dict] = {}
        dropped: dict[str, dict] = {}
        for name in arm.channels:
            fn = channels[arm.source[name]]
            say(f"  scoring {name} ({arm.source[name]}) ...")
            report = evaluate_spans(cases, fn, k=K, pool=CHUNK_POOL)
            scores[name] = report_to_dict(report)
            dropped[name] = report_to_dict(evaluate_spans(easy, fn, k=K, pool=CHUNK_POOL))[
                "overall"
            ]
            say(f"=== {name} ===\n{format_span_report(report)}")
            say(
                f"  discriminated-out cases ({len(easy)}): span recall "
                f"{dropped[name]['span_recall']:.3f}"
            )
        if args.no_log:
            continue

        run_id = blurb_run_id(today, arm)
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run = make_run(
            run_id=run_id,
            experiment=args.experiment,
            label=arm.label,
            date=today,
            commit=commit,
            control=arm.control,
            config={
                "k": K,
                "embedding": MULTILINGUAL_MODEL,
                "chunking": ChunkingCfg().model_dump(),
                "contextual": {
                    "model": ctx.model,
                    "timeout": ctx.timeout,
                    "blurbed_notes": blurbed,
                },
                "rerank_use_context": arm.key == "context-rerank",
            },
            cases=len(cases),
            # Blurbs do not change chunking, so the control's ceiling holds.
            reachable=runs[arm.control]["reachable"],
            channels=scores,
            notes=f"Channels measured: {', '.join(arm.channels)}.",
        )
        run["discriminated"] = dropped
        append_run(RUNS, run)
        print(f"  logged {run_id}", flush=True)

    if not args.no_log:
        page = RESULTS / "experiment_log.html"
        template = (RESULTS / "template.html").read_text(encoding="utf-8")
        page.write_text(render_html(load_runs(RUNS), template), encoding="utf-8")
        print(f"\nrendered {page}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
