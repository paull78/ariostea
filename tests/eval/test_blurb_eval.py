import pytest

from ariostea.eval.blurb_eval import (
    ARMS,
    BlurbCoverageError,
    blurb_run_id,
    require_full_coverage,
)


def test_run_ids_are_distinct_per_arm_and_dated():
    ids = [blurb_run_id("2026-09-30", arm) for arm in ARMS]
    assert len(set(ids)) == len(ARMS)
    assert all(i.startswith("2026-09-30-blurbs-") for i in ids)


def test_each_arm_names_its_control_and_channels():
    by_key = {arm.key: arm for arm in ARMS}
    assert by_key["raw-rerank"].channels == ("DENSE", "SPARSE", "HYBRID")
    assert by_key["raw-rerank"].control == "2026-09-24-chunk-160t+40-dense-hybrid-sparse"
    assert by_key["fused"].channels == ("FUSED",)
    assert by_key["fused"].control == "2026-09-28-chunk-160t+40-fused"
    assert by_key["context-rerank"].channels == ("HYBRID",)
    assert by_key["context-rerank"].control == "2026-09-24-chunk-160t+40-dense-hybrid-sparse"


def test_full_coverage_passes_and_reports_the_count():
    rows = [("a.md", "blurb"), ("a.md", "blurb"), ("b.md", "other")]
    assert require_full_coverage(rows) == 2


def test_a_note_without_a_blurb_aborts_and_is_named():
    rows = [("a.md", "blurb"), ("b.md", None), ("c.md", "")]
    with pytest.raises(BlurbCoverageError, match="b.md.*c.md"):
        require_full_coverage(rows)


def test_an_empty_index_aborts():
    with pytest.raises(BlurbCoverageError):
        require_full_coverage([])
