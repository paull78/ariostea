"""Measure chunking policies on the wiki gold set and log every result.

Usage:
    uv run python eval/run_chunk_sweep.py 512w 128t 128t+32 --channels DENSE,SPARSE
    uv run python eval/run_chunk_sweep.py 128t+32 --experiment "Chunk sweep: full"

A spec is a size, a unit (w words, t model tokens) and an optional +overlap.
Per spec: index the corpus with that policy, count the gold spans it can reach
at all, score the requested channels on the gold set and on the cases the
discrimination gate dropped, and append the run to eval/results/runs.jsonl
with the committed baseline as its control. The shaded log page is re-rendered
at the end, and each run's printed tables are kept in eval/results/logs/.

HYBRID reranks every candidate on CPU and costs about an hour per spec; the
DENSE,SPARSE pass takes a few minutes and is the way to narrow a grid first.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import tempfile
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.config.container import build_chunker
from ariostea.config.schema import ChunkingCfg
from ariostea.eval.chunk_sweep import (
    describe,
    load_discriminated,
    parse_spec,
    reachable_spans,
    sweep_run_id,
)
from ariostea.eval.results_log import append_run, load_runs, make_run, render_html, report_to_dict
from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import WikiGoldCase, load_wiki_gold
from ariostea.eval.wiki_index import (
    CHUNK_POOL,
    MULTILINGUAL_MODEL,
    index_wiki_corpus,
    wiki_channels,
)

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
K = 5
CHANNELS = ("DENSE", "SPARSE", "HYBRID")
BASELINE = "2026-09-06-baseline"


def _commit() -> str:
    """Short HEAD, marked dirty when retrieval or eval code has uncommitted
    edits -- a logged number must be traceable to the code that produced it."""
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "eval/*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}+dirty" if dirty else head


def _bodies() -> dict[str, str]:
    """Note bodies exactly as indexing parses them, for the coverage count."""
    parser = ObsidianMarkdownParser()
    bodies = {}
    for path in sorted(WIKI.glob("*/*.md")):
        rel = f"{path.parent.name}/{path.name}"
        _, body = parser.parse(rel, path.read_text(encoding="utf-8"), 0.0)
        bodies[rel] = body
    return bodies


def _measure(
    spec: str,
    cfg: ChunkingCfg,
    channels: list[str],
    cases: list[WikiGoldCase],
    easy: list[WikiGoldCase],
    reachable: int,
) -> tuple[dict, dict, str]:
    """Index with `cfg`, score, and release the index on return (see
    generate_gold._retrieval_stages for why that has to be a function)."""
    lines: list[str] = []

    def say(text: str) -> None:
        print(text, flush=True)
        lines.append(text)

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        say(f"\n##### {spec}  ({describe(cfg)})  spans reachable {reachable}/{len(cases)}")
        say("  indexing ...")
        container = index_wiki_corpus(WIKI, db, chunking=cfg)
        available = wiki_channels(db, container)
        scores: dict[str, dict] = {}
        dropped: dict[str, dict] = {}
        for name in channels:
            say(f"  scoring {name} ...")
            report = evaluate_spans(cases, available[name], k=K, pool=CHUNK_POOL)
            scores[name] = report_to_dict(report)
            dropped[name] = report_to_dict(
                evaluate_spans(easy, available[name], k=K, pool=CHUNK_POOL)
            )["overall"]
            say(f"=== {name} ===\n{format_span_report(report)}")
            say(
                f"  discriminated-out cases ({len(easy)}): span recall "
                f"{dropped[name]['span_recall']:.3f}"
            )
    return scores, dropped, "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("specs", nargs="+", help="chunking specs, e.g. 512w 128t 128t+32")
    parser.add_argument("--channels", default=",".join(CHANNELS), help="comma-separated")
    parser.add_argument("--experiment", default="Chunk sweep", help="group name in the log")
    parser.add_argument("--control", default=BASELINE, help="run id to compare against")
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    args = parser.parse_args(argv)

    channels = [c.strip().upper() for c in args.channels.split(",") if c.strip()]
    unknown = sorted(set(channels) - set(CHANNELS))
    if unknown:
        parser.error(f"unknown channel(s): {', '.join(unknown)}")
    try:
        configs = [(spec, parse_spec(spec)) for spec in args.specs]  # fail before any indexing
    except ValueError as exc:
        parser.error(str(exc).splitlines()[0])

    today = dt.date.today().isoformat()
    commit = _commit()
    if not args.no_log:
        logged = {run["id"] for run in load_runs(RUNS)}
        if args.control not in logged:
            parser.error(f"control run {args.control!r} is not in {RUNS}")
        clashes = [s for s, _ in configs if sweep_run_id(today, s, channels) in logged]
        if clashes:
            parser.error(f"already logged today with these channels: {', '.join(clashes)}")

    cases = load_wiki_gold(WIKI / "gold.json")
    easy = load_discriminated(WIKI / "gold_rejected.json")
    bodies = _bodies()
    counter = None  # the model tokenizer, loaded once and only if a spec needs it

    for position, (spec, cfg) in enumerate(configs, start=1):
        print(f"\n[{position}/{len(configs)}] {spec}", flush=True)
        if cfg.unit == "model_tokens" and counter is None:
            counter = FastEmbedEmbeddings(model_name=MULTILINGUAL_MODEL)
        reachable = reachable_spans(cases, bodies, build_chunker(cfg, counter))
        scores, dropped, printed = _measure(spec, cfg, channels, cases, easy, reachable)
        if args.no_log:
            continue

        run_id = sweep_run_id(today, spec, channels)
        LOGS.mkdir(parents=True, exist_ok=True)
        (LOGS / f"{run_id}.txt").write_text(printed, encoding="utf-8")
        run = make_run(
            run_id=run_id,
            experiment=args.experiment,
            label=describe(cfg),
            date=today,
            commit=commit,
            control=args.control,
            config={"k": K, "embedding": MULTILINGUAL_MODEL, "chunking": cfg.model_dump()},
            cases=len(cases),
            reachable=reachable,
            channels=scores,
            notes=f"Channels measured: {', '.join(channels)}.",
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
