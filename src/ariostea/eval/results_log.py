"""The experiment log: every evaluation run, kept as one JSON line per run.

`eval/results/runs.jsonl` is the record of what was measured, under which
configuration, at which commit, and against which control run. The rendered
page (`eval/render_results.py`) is derived from it and can be regenerated at
any time, so the JSONL file is what gets reviewed and never hand-edited.

A run names its `control` -- the run its numbers are compared against -- so
the page can shade each cell by its change rather than by its raw value. A
chunking change that lifts dense and sinks sparse should read as exactly that,
not as two columns of similar-looking numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ariostea.eval.spaneval import SpanEvalReport

PLACEHOLDER = "__RUNS_JSON__"

# Field order matches `format_span_report`'s columns, so a printed report and a
# recorded one describe the same numbers in the same order.
_METRICS = ("note_recall", "note_mrr", "note_ndcg", "span_recall", "span_mrr", "span_ndcg")
_ROW = re.compile(r"^(\w+)\s+(\d+)" + r"\s+(\d+\.\d+)" * len(_METRICS) + r"\s*$")


def report_to_dict(report: SpanEvalReport) -> dict[str, dict[str, float]]:
    """Type name -> metrics, query types first and `overall` last."""
    return {
        score.type: {
            "n": score.n,
            "note_recall": score.note_recall_at_k,
            "note_mrr": score.note_mrr,
            "note_ndcg": score.note_ndcg_at_k,
            "span_recall": score.span_recall_at_k,
            "span_mrr": score.span_mrr,
            "span_ndcg": score.span_ndcg_at_k,
        }
        for score in (*report.by_type, report.overall)
    }


def parse_span_report(text: str) -> dict[str, dict[str, float]]:
    """Read a table printed by `format_span_report` back into `report_to_dict` form.

    For backfilling runs that were only ever printed. Printed values carry three
    decimals, so a parsed run is rounded where a recorded one is exact.
    """
    table: dict[str, dict[str, float]] = {}
    for line in text.splitlines():
        match = _ROW.match(line.strip())
        if match is None:
            continue
        name, n, *values = match.groups()
        table[name] = {"n": int(n), **dict(zip(_METRICS, map(float, values), strict=True))}
    if not table:
        raise ValueError("no span report table found in the text")
    return table


def make_run(
    *,
    run_id: str,
    experiment: str,
    label: str,
    date: str,
    commit: str,
    control: str,
    config: dict[str, Any],
    cases: int,
    reachable: int,
    channels: dict[str, dict[str, dict[str, float]]],
    notes: str = "",
) -> dict[str, Any]:
    """One log entry. `reachable` counts gold spans some chunk contains whole --
    the ceiling on span recall for every channel under this chunking."""
    return {
        "id": run_id,
        "experiment": experiment,
        "label": label,
        "date": date,
        "commit": commit,
        "control": control,
        "config": config,
        "cases": cases,
        "reachable": reachable,
        "channels": channels,
        "notes": notes,
    }


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_run(path: Path, run: dict[str, Any]) -> None:
    """Append `run`, refusing a duplicate id or a control that is not logged.

    Both checks guard the page's deltas: a duplicate id makes "compared against
    X" ambiguous, and a missing control would render every cell as unchanged.
    """
    runs = load_runs(path)
    ids = {existing["id"] for existing in runs}
    if run["id"] in ids:
        raise ValueError(f"run {run['id']!r} is already logged")
    if run["control"] != run["id"] and run["control"] not in ids:
        raise ValueError(f"control run {run['control']!r} is not in the log")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(run, ensure_ascii=False) + "\n")


def render_html(runs: list[dict[str, Any]], template: str) -> str:
    """Embed `runs` into `template` at its placeholder.

    `</` is escaped so no string inside the data -- a run's free-text notes
    above all -- can close the script element the JSON sits in.
    """
    if PLACEHOLDER not in template:
        raise ValueError(f"template has no {PLACEHOLDER} slot to embed the runs in")
    payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    return template.replace(PLACEHOLDER, payload)
