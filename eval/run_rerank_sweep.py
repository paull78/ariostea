"""Measure reranker configurations on the wiki gold set and log every result.

Usage:
    uv run python eval/run_rerank_sweep.py jinaai/jina-reranker-v2-base-multilingual:100
    uv run python eval/run_rerank_sweep.py MODEL:50 MODEL:20 --no-log

A configuration is a reranker model and the rerank pool, the number of fused
candidates it scores. The corpus is indexed once at the defaults; each
configuration gets a second container over the same database, since the
reranker does not touch the index. Per configuration: HYBRID span scores on
the gold set and on the discriminated-out cases, the median and 95th
percentile time of one search, and how many queries get fewer than 10 notes
from search_sources. Each run goes to eval/results/runs.jsonl with the
current default's HYBRID run as its control, and its printed tables to
eval/results/logs/.

A configuration whose container warns, such as a reranker falling back to
fused order, aborts the sweep before it is logged (search-speed.14).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
import time
from pathlib import Path

from ariostea.adapters.embedding.fastembed_local import FastEmbedEmbeddings
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.config.container import build_chunker, build_container
from ariostea.config.schema import ChunkingCfg, EmbeddingCfg
from ariostea.eval.chunk_sweep import load_discriminated, model_slug, reachable_spans
from ariostea.eval.rerank_sweep import (
    guarded,
    parse_entry,
    rerank_config,
    rerank_run_config,
    rerank_run_id,
    search_latency_ms,
    short_sources,
)
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
from ariostea.eval.wiki_index import CHUNK_POOL, index_wiki_corpus
from ariostea.mcp.handlers import search_payload, search_sources_payload

EVAL = Path(__file__).resolve().parent
WIKI = EVAL / "wiki"
RESULTS = EVAL / "results"
RUNS = RESULTS / "runs.jsonl"
LOGS = RESULTS / "logs"
K = 5
CONTROL = "2026-10-13-chunk-256t+64-bge-m3-norm-hybrid"


def _bodies() -> dict[str, str]:
    """Note bodies exactly as indexing parses them, for the coverage count."""
    parser = ObsidianMarkdownParser()
    bodies = {}
    for path in sorted(WIKI.glob("*/*.md")):
        rel = f"{path.parent.name}/{path.name}"
        _, body = parser.parse(rel, path.read_text(encoding="utf-8"), 0.0)
        bodies[rel] = body
    return bodies


def _measure(base, model, pool, cases, easy, say):
    """Score one configuration over the already-built index of `base`.

    Each gold query is searched once, timed, and the scoring reads those
    results back, so latency and recall come from the same searches."""
    container = build_container(rerank_config(base.config, model, pool))
    queries = [case.query for case in cases]
    results: dict[str, list[tuple[str, str]]] = {}

    def search(query: str) -> None:
        payload = search_payload(container, query=query, k=CHUNK_POOL)
        results[query] = [(r["note_path"], r["text"]) for r in payload["results"]]

    latency = search_latency_ms(search, queries)
    report = evaluate_spans(cases, lambda query, k: results[query][:k], k=K, pool=CHUNK_POOL)

    def live(query: str, k: int) -> list[tuple[str, str]]:
        search(query)
        return results[query][:k]

    dropped = report_to_dict(evaluate_spans(easy, live, k=K, pool=CHUNK_POOL))["overall"]
    short = short_sources(
        lambda query, k: search_sources_payload(container, query=query, k=k)["sources"], queries
    )
    say(f"=== HYBRID ===\n{format_span_report(report)}")
    say(f"  discriminated-out cases ({len(easy)}): span recall {dropped['span_recall']:.3f}")
    say(
        f"  search median {latency['search_ms_median']} ms, p95 {latency['search_ms_p95']} ms; "
        f"search_sources short of 10 notes for {short} of {len(queries)} queries"
    )
    return {"HYBRID": report_to_dict(report)}, {"HYBRID": dropped}, latency, short


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("configs", nargs="+", help="MODEL:POOL entries")
    parser.add_argument("--experiment", default="Search speed", help="group name in the log")
    parser.add_argument("--control", default=CONTROL, help="run id to compare against")
    parser.add_argument("--no-log", action="store_true", help="print only, record nothing")
    args = parser.parse_args(argv)

    try:
        configs = [parse_entry(entry) for entry in args.configs]  # fail before indexing
    except ValueError as exc:
        parser.error(str(exc))

    today = dt.date.today().isoformat()
    commit = code_commit()
    if not args.no_log:
        logged = {run["id"] for run in load_runs(RUNS)}
        if args.control not in logged:
            parser.error(f"control run {args.control!r} is not in {RUNS}")
        clashes = [f"{m}:{p}" for m, p in configs if rerank_run_id(today, m, p) in logged]
        if clashes:
            parser.error(f"already logged today: {', '.join(clashes)}")

    cases = load_wiki_gold(WIKI / "gold.json")
    easy = load_discriminated(WIKI / "gold_rejected.json")
    chunking = ChunkingCfg()
    counter = FastEmbedEmbeddings(model_name=EmbeddingCfg().local_model)
    reachable = reachable_spans(cases, _bodies(), build_chunker(chunking, counter))

    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "eval.db")
        print(f"indexing the wiki corpus at the defaults, spans reachable {reachable}/{len(cases)}")
        started = time.perf_counter()
        base = index_wiki_corpus(WIKI, db)
        index_seconds = round(time.perf_counter() - started, 1)
        print(f"  indexed in {index_seconds} s", flush=True)

        for position, (model, pool) in enumerate(configs, start=1):
            lines: list[str] = []

            def say(text: str) -> None:
                print(text, flush=True)
                lines.append(text)

            say(f"\n[{position}/{len(configs)}] {model}, pool {pool}")
            scores, dropped, latency, short = guarded(
                lambda: _measure(base, model, pool, cases, easy, say)
            )
            if args.no_log:
                continue

            run_id = rerank_run_id(today, model, pool)
            LOGS.mkdir(parents=True, exist_ok=True)
            (LOGS / f"{run_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            run = make_run(
                run_id=run_id,
                experiment=args.experiment,
                label=f"{model_slug(model)}, pool {pool}",
                date=today,
                commit=commit,
                control=args.control,
                config=rerank_run_config(K, model, pool),
                cases=len(cases),
                reachable=reachable,
                channels=scores,
                notes="Channels measured: HYBRID.",
            )
            run["discriminated"] = dropped
            run["cost"] = {"index_seconds": index_seconds, **latency}
            run["sources_short"] = short
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
