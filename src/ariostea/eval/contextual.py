"""Pure helpers for the contextual-lift eval (run_contextual_eval.py).

Kept here, in the package, rather than in the runner script so they can be
unit-tested without a live model or database.
"""

from __future__ import annotations

from ariostea.eval.harness import EvalReport


def _pair(before: float, after: float) -> str:
    return f"{before:.3f} → {after:.3f} ({after - before:+.3f})"


def format_delta(off: EvalReport, on: EvalReport) -> str:
    """Render one OFF→ON row per scenario (plus overall) for a single channel.

    Both reports come from the same gold file, so their scenario sets match;
    rows are paired by scenario name.
    """
    before = {s.scenario: s for s in (*off.by_scenario, off.overall)}
    header = f"{'scenario':<12} {'n':>3}  {'recall@' + str(on.k):<24} mrr"
    lines = [header]
    for s in (*on.by_scenario, on.overall):
        b = before[s.scenario]
        recall = _pair(b.recall_at_k, s.recall_at_k)
        mrr = _pair(b.mrr, s.mrr)
        lines.append(f"{s.scenario:<12} {s.n:>3}  {recall:<24} {mrr}")
    return "\n".join(lines)


def find_uncontextualized_notes(rows: list[tuple[str, str | None]]) -> list[str]:
    """Given (note_path, context_blurb) rows (one per chunk), return the sorted,
    distinct note paths that have any null/empty blurb.

    A non-empty result means the ON index is only partially contextualized, so
    an OFF-vs-ON comparison would be confounded — the runner aborts on it.
    """
    missed = {path for path, blurb in rows if not blurb}
    return sorted(missed)
