"""Render the experiment log to a static page.

Usage:  uv run python eval/render_results.py

Reads eval/results/runs.jsonl and writes eval/results/experiment_log.html:
one row per run, one column per channel and query type, each cell shaded by
its change against that run's control. The JSONL file is the record; the page
is derived and safe to regenerate at any time.
"""

from __future__ import annotations

from pathlib import Path

from ariostea.eval.results_log import load_runs, render_html

RESULTS = Path(__file__).resolve().parent / "results"
RUNS = RESULTS / "runs.jsonl"
TEMPLATE = RESULTS / "template.html"
PAGE = RESULTS / "experiment_log.html"


def main() -> int:
    runs = load_runs(RUNS)
    PAGE.write_text(render_html(runs, TEMPLATE.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"rendered {len(runs)} runs to {PAGE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
