import pytest

from ariostea.eval.harness import GoldCase, dedupe, evaluate, format_report, load_gold


def test_load_gold_parses_cases(tmp_path):
    gold = tmp_path / "gold.json"
    # → is the "→" arrow; written escaped to keep the test ASCII-safe.
    gold.write_text(
        '[{"query": "dice game", "query_lang": "en", '
        '"expected": ["dadi_it.md"], "scenario": "en\\u2192it"}]',
        encoding="utf-8",
    )

    cases = load_gold(gold)

    assert cases == [
        GoldCase(
            query="dice game",
            query_lang="en",
            expected=("dadi_it.md",),
            scenario="en→it",
        )
    ]


def test_dedupe_keeps_first_occurrence_in_order():
    assert dedupe(["a.md", "b.md", "a.md", "c.md", "b.md"]) == ["a.md", "b.md", "c.md"]


def test_evaluate_aggregates_overall_and_by_scenario():
    cases = [
        GoldCase("q1", "en", ("it1.md",), "en→it"),
        GoldCase("q2", "it", ("en1.md",), "it→en"),
        GoldCase("q3", "en", ("en2.md",), "same"),
    ]
    # Fake ranker: q1 hits at rank 1, q2 misses entirely, q3 hits at rank 2.
    table = {
        "q1": ["it1.md", "x.md"],
        "q2": ["y.md", "z.md"],
        "q3": ["w.md", "en2.md"],
    }

    report = evaluate(cases, lambda query, k: table[query][:k], k=5)

    assert report.k == 5
    assert report.overall.n == 3
    assert report.overall.recall_at_k == pytest.approx(2 / 3)
    assert report.overall.mrr == pytest.approx((1.0 + 0.0 + 0.5) / 3)

    by = {s.scenario: s for s in report.by_scenario}
    assert by["en→it"].recall_at_k == 1.0 and by["en→it"].mrr == 1.0
    assert by["it→en"].recall_at_k == 0.0
    assert by["same"].mrr == 0.5


def test_format_report_contains_scenarios_and_overall():
    cases = [GoldCase("q1", "en", ("a.md",), "same")]
    report = evaluate(cases, lambda query, k: ["a.md"], k=3)

    text = format_report(report)

    assert "recall@3" in text
    assert "same" in text
    assert "overall" in text
