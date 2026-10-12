import json

import pytest

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.config.schema import ChunkingCfg
from ariostea.eval.chunk_sweep import (
    describe,
    load_discriminated,
    model_slug,
    parse_spec,
    reachable_spans,
    run_config,
    sweep_run_id,
)
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase


def _case(note, span):
    return WikiGoldCase(
        query="q",
        query_lang="en",
        type="buried",
        scenario="buried",
        expected_notes=(note,),
        answer_spans=(AnswerSpan(note=note, text=span),),
    )


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("512w", ChunkingCfg(max_tokens=512, overlap=0, unit="words")),
        ("128t", ChunkingCfg(max_tokens=128, overlap=0, unit="model_tokens")),
        ("128t+32", ChunkingCfg(max_tokens=128, overlap=32, unit="model_tokens")),
        ("96w+24", ChunkingCfg(max_tokens=96, overlap=24, unit="words")),
    ],
)
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected


@pytest.mark.parametrize("spec", ["128", "t128", "128x", "128t+", "128t-32", ""])
def test_parse_spec_rejects_malformed_specs(spec):
    with pytest.raises(ValueError, match="spec"):
        parse_spec(spec)


def test_parse_spec_rejects_an_overlap_as_large_as_the_window():
    with pytest.raises(ValueError):
        parse_spec("64t+64")


def test_describe_reads_like_the_log_page():
    assert describe(parse_spec("128t+32")) == "128 tokens, overlap 32"
    assert describe(parse_spec("512w")) == "512 words"


def test_run_ids_are_distinct_per_spec_and_channel_set():
    # A cheap dense-and-sparse pass and a later full pass of the same spec are
    # separate log entries; the log refuses a duplicate id.
    cheap = sweep_run_id("2026-09-24", "128t+32", ["SPARSE", "DENSE"])
    full = sweep_run_id("2026-09-24", "128t+32", ["DENSE", "SPARSE", "HYBRID"])
    assert cheap == "2026-09-24-chunk-128t+32-dense-sparse"
    assert full != cheap
    assert sweep_run_id("2026-09-24", "128t", ["DENSE", "SPARSE"]) != cheap


def test_reachable_spans_counts_spans_some_chunk_holds_whole():
    notes = {"a.md": "# A\n" + " ".join(f"w{i}" for i in range(20))}
    straddling = _case("a.md", "w6 w7 w8 w9")  # cut between w7 and w8 at 10 words
    inside = _case("a.md", "w1 w2")
    cases = [straddling, inside]
    assert reachable_spans(cases, notes, HeadingAwareChunker(max_tokens=10)) == 1
    assert reachable_spans(cases, notes, HeadingAwareChunker(max_tokens=10, overlap=4)) == 2


def test_load_discriminated_reads_only_that_stage(tmp_path):
    path = tmp_path / "gold_rejected.json"
    path.write_text(
        json.dumps(
            [
                {
                    "stage": "discrimination",
                    "reason": "r",
                    "query": "q1",
                    "note": "a.md",
                    "span": "s1",
                    "type": "buried",
                },
                {
                    "stage": "ambiguity",
                    "reason": "r",
                    "query": "q2",
                    "note": "b.md",
                    "span": "s2",
                    "type": "paraphrase",
                },
            ]
        )
    )
    cases = load_discriminated(path)
    assert [c.query for c in cases] == ["q1"]
    assert cases[0].answer_spans == (AnswerSpan(note="a.md", text="s1"),)
    assert cases[0].expected_notes == ("a.md",)


def test_model_slug_is_the_lowercased_last_path_part():
    assert model_slug("intfloat/multilingual-e5-large") == "multilingual-e5-large"
    assert model_slug("BAAI/bge-m3") == "bge-m3"
    assert model_slug("plain-name") == "plain-name"


def test_run_id_without_a_model_is_unchanged():
    # Control reruns must keep the id shape every logged run already has.
    expected = "2026-10-12-chunk-160t+40-dense"
    assert sweep_run_id("2026-10-12", "160t+40", ["DENSE"]) == expected
    assert sweep_run_id("2026-10-12", "160t+40", ["DENSE"], model=None) == expected


def test_run_id_with_a_model_carries_its_slug():
    run_id = sweep_run_id("2026-10-12", "160t+40", ["DENSE"], model="BAAI/bge-m3")
    assert run_id == "2026-10-12-chunk-160t+40-bge-m3-dense"


def test_run_config_records_the_model():
    cfg = ChunkingCfg()
    assert run_config(5, cfg, "m") == {"k": 5, "embedding": "m", "chunking": cfg.model_dump()}
