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
    by_type = {s.type: s for s in report.by_type}
    assert by_type["paraphrase"].span_recall_at_k == 1.0
    assert by_type["exact_term"].span_recall_at_k == 0.0


def test_note_recall_counts_distinct_notes_not_chunks():
    case = WikiGoldCase(
        query="q",
        query_lang="en",
        type="paraphrase",
        scenario="paraphrase",
        expected_notes=("cello.md",),
        answer_spans=(AnswerSpan(note="cello.md", text="four strings"),),
    )
    # Top chunks are all one note; the expected note appears 4th in the pool.
    pool_chunks = [
        ("violin.md", "a"),
        ("violin.md", "b"),
        ("violin.md", "c"),
        ("cello.md", "the cello has four strings"),
        ("guitar.md", "d"),
    ]

    def span_fn(query, k):
        return pool_chunks[:k]

    report = evaluate_spans([case], span_fn, k=3, pool=10)
    # cello is the 2nd DISTINCT note (violin, cello, guitar) → within top-3 notes.
    assert report.overall.note_recall_at_k == 1.0
    # but the cello CHUNK is at chunk-rank 4 → NOT within the top-3 chunks.
    assert report.overall.span_recall_at_k == 0.0


def test_format_span_report_includes_overall_row():
    cases = [_case("q_hit", "paraphrase", "violin.md", "perfect fifths")]

    def span_fn(query, k):
        return [("violin.md", "tuned in perfect fifths")]

    text = format_span_report(evaluate_spans(cases, span_fn, k=5))
    assert "overall" in text
    assert "paraphrase" in text
