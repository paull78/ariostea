from ariostea.eval.contextual import find_uncontextualized_notes, format_delta
from ariostea.eval.harness import EvalReport, ScenarioScore


def _report(buried, overall):
    return EvalReport(k=5, overall=overall, by_scenario=(buried,))


def test_format_delta_shows_before_after_and_signed_delta():
    off = _report(
        ScenarioScore("buried", 5, 0.200, 0.150),
        ScenarioScore("overall", 5, 0.200, 0.150),
    )
    on = _report(
        ScenarioScore("buried", 5, 0.800, 0.700),
        ScenarioScore("overall", 5, 0.800, 0.700),
    )
    out = format_delta(off, on)

    assert "buried" in out
    assert "0.200 → 0.800 (+0.600)" in out
    assert "0.150 → 0.700 (+0.550)" in out
    assert "overall" in out


def test_format_delta_renders_negative_and_zero_deltas():
    off = _report(
        ScenarioScore("direct", 2, 1.000, 1.000),
        ScenarioScore("overall", 2, 1.000, 1.000),
    )
    on = _report(
        ScenarioScore("direct", 2, 1.000, 0.500),
        ScenarioScore("overall", 2, 1.000, 0.500),
    )
    out = format_delta(off, on)

    assert "1.000 → 1.000 (+0.000)" in out
    assert "1.000 → 0.500 (-0.500)" in out


def test_all_blurbs_present_returns_empty():
    rows = [("a.md", "blurb"), ("a.md", "blurb"), ("b.md", "x")]
    assert find_uncontextualized_notes(rows) == []


def test_any_null_blurb_flags_its_note():
    rows = [("a.md", "blurb"), ("b.md", None), ("b.md", None)]
    assert find_uncontextualized_notes(rows) == ["b.md"]


def test_partial_and_empty_blurbs_flagged_and_sorted():
    # z.md has an empty-string blurb; a.md has one null among its chunks.
    rows = [("z.md", ""), ("a.md", None), ("a.md", "ok")]
    assert find_uncontextualized_notes(rows) == ["a.md", "z.md"]
