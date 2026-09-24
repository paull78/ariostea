import json

import pytest

from ariostea.eval.results_log import (
    append_run,
    load_runs,
    make_run,
    parse_span_report,
    render_html,
    report_to_dict,
)
from ariostea.eval.spaneval import SpanEvalReport, SpanScore

REPORT = SpanEvalReport(
    k=5,
    overall=SpanScore("overall", 3, 0.9, 0.8, 0.85, 0.7, 0.6, 0.65),
    by_type=(
        SpanScore("buried", 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        SpanScore("paraphrase", 2, 0.85, 0.7, 0.775, 0.55, 0.4, 0.475),
    ),
)

PRINTED = """\
type             n   note_r@5  note_mrr note_ndcg   span_r@5  span_mrr span_ndcg
buried          40      0.800     0.507     0.563      0.325     0.197     0.200
paraphrase      40      0.775     0.561     0.599      0.300     0.228     0.214
overall        167      0.719     0.501     0.538      0.335     0.209     0.219
"""


def _run(run_id="r1", control="r1", **overrides):
    run = make_run(
        run_id=run_id,
        experiment="chunk size",
        label="512 words",
        date="2026-09-24",
        commit="abc1234",
        control=control,
        config={"chunking": {"max_tokens": 512}},
        cases=3,
        reachable=3,
        channels={"DENSE": report_to_dict(REPORT)},
    )
    run.update(overrides)
    return run


def test_report_to_dict_keys_every_type_and_overall():
    table = report_to_dict(REPORT)
    assert list(table) == ["buried", "paraphrase", "overall"]
    assert table["paraphrase"] == {
        "n": 2,
        "note_recall": 0.85,
        "note_mrr": 0.7,
        "note_ndcg": 0.775,
        "span_recall": 0.55,
        "span_mrr": 0.4,
        "span_ndcg": 0.475,
    }


def test_parse_span_report_reads_the_printed_table():
    # Backfills runs that were only ever printed, such as the chunk-size pilot.
    table = parse_span_report(PRINTED)
    assert list(table) == ["buried", "paraphrase", "overall"]
    assert table["overall"]["n"] == 167
    assert table["buried"]["span_recall"] == pytest.approx(0.325)


def test_parse_span_report_rejects_text_with_no_table():
    # An empty result would render as a run where every channel was skipped.
    with pytest.raises(ValueError, match="no span report"):
        parse_span_report("indexing 79 notes ...\n")


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run(path, _run())
    assert load_runs(path) == [_run()]


def test_append_refuses_a_duplicate_id(tmp_path):
    # Two rows with one id would make every delta against it ambiguous.
    path = tmp_path / "runs.jsonl"
    append_run(path, _run())
    with pytest.raises(ValueError, match="already logged"):
        append_run(path, _run())


def test_append_refuses_an_unknown_control(tmp_path):
    # A run compared against nothing would render every cell as neutral,
    # which reads as "no change" rather than "no baseline".
    path = tmp_path / "runs.jsonl"
    with pytest.raises(ValueError, match="control"):
        append_run(path, _run(run_id="r2", control="missing"))


def test_a_run_may_be_its_own_control(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_run(path, _run(run_id="base", control="base"))
    assert load_runs(path)[0]["control"] == "base"


def test_load_runs_on_a_missing_file_is_empty(tmp_path):
    assert load_runs(tmp_path / "absent.jsonl") == []


def test_render_embeds_the_runs_as_json():
    html = render_html([_run()], template="<script>__RUNS_JSON__</script>")
    payload = html.removeprefix("<script>").removesuffix("</script>")
    assert json.loads(payload) == [_run()]


def test_render_cannot_be_broken_out_of_by_a_note():
    # Notes are free text; one containing "</script>" must not end the block.
    html = render_html(
        [_run(notes="see </script><b>x</b>")], template="<script>__RUNS_JSON__</script>"
    )
    assert html.count("</script>") == 1


def test_render_needs_the_placeholder():
    with pytest.raises(ValueError, match="__RUNS_JSON__"):
        render_html([_run()], template="<p>no slot</p>")
