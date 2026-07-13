from ariostea.eval.spaneval import evaluate_spans, format_span_report
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase


def _case(query, type_, note, span_text):
    return WikiGoldCase(
        query=query,
        query_lang="en",
        type=type_,
        scenario=type_,
        expected_notes=(note,),
        answer_spans=(AnswerSpan(note=note, text=span_text),),
    )


def test_evaluate_spans_reports_note_and_span_metrics_per_type():
    cases = [
        _case("q_hit", "paraphrase", "violin.md", "perfect fifths"),
        _case("q_miss", "exact_term", "cello.md", "spruce top"),
    ]

    # q_hit: correct note + span text at rank 1. q_miss: right note but span text absent.
    responses = {
        "q_hit": [("violin.md", "the violin is tuned in perfect fifths")],
        "q_miss": [("cello.md", "the cello is large")],
    }

    def span_fn(query, k):
        return responses[query][:k]

    report = evaluate_spans(cases, span_fn, k=5)

    assert report.overall.n == 2
    assert report.overall.note_recall_at_k == 1.0  # both retrieved the right note
    assert report.overall.span_recall_at_k == 0.5  # only q_hit matched the span
    by_type = {s.group: s for s in report.by_type}
    assert by_type["paraphrase"].span_recall_at_k == 1.0
    assert by_type["exact_term"].span_recall_at_k == 0.0


def test_format_span_report_includes_overall_row():
    cases = [_case("q_hit", "paraphrase", "violin.md", "perfect fifths")]

    def span_fn(query, k):
        return [("violin.md", "tuned in perfect fifths")]

    text = format_span_report(evaluate_spans(cases, span_fn, k=5))
    assert "overall" in text
    assert "paraphrase" in text
