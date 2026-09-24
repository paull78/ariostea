import pytest

from ariostea.eval.gold_passages import Passage
from ariostea.eval.gold_prompts import (
    GENERATION_SYSTEM,
    JUDGE_SYSTEM,
    generation_user,
    judge_user,
    parse_json_object,
)

PASSAGE = Passage(
    note="strings/violin.md",
    heading="Construction",
    text="The bridge transmits vibration to the body.",
    offset=100,
    note_chars=9000,
    rare_terms=("ponticello", "tasto"),
    rare_score=40.0,
)


def test_parse_json_object_reads_a_bare_object():
    assert parse_json_object('{"query": "a", "answer_span": "b"}') == {
        "query": "a",
        "answer_span": "b",
    }


def test_parse_json_object_unwraps_a_fenced_block():
    raw = 'Here you go:\n```json\n{"query": "a"}\n```\n'
    assert parse_json_object(raw) == {"query": "a"}


def test_parse_json_object_unwraps_an_unlabelled_fence():
    assert parse_json_object('```\n{"query": "a"}\n```') == {"query": "a"}


def test_parse_json_object_discards_a_reasoning_block():
    # qwen3-family models emit <think>...</think> before the answer. A brace
    # inside that block would otherwise be read as the start of the object.
    raw = '<think>I should answer {maybe} like this</think>\n{"query": "a"}'
    assert parse_json_object(raw) == {"query": "a"}


def test_parse_json_object_reads_an_object_surrounded_by_prose():
    assert parse_json_object('Sure. {"query": "a"} Hope that helps!') == {"query": "a"}


def test_parse_json_object_keeps_a_nested_object_intact():
    assert parse_json_object('{"a": {"b": 1}}') == {"a": {"b": 1}}


def test_parse_json_object_raises_when_there_is_no_object():
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_object("I cannot help with that.")


def test_parse_json_object_raises_on_malformed_json():
    with pytest.raises(ValueError):
        parse_json_object('{"query": }')


def test_parse_json_object_raises_when_the_value_is_not_a_mapping():
    # `["a"]` has no braces at all; `[{"a": 1}]` does, and would parse to a
    # list that the caller would then index by key.
    with pytest.raises(ValueError, match="no JSON object"):
        parse_json_object('["a", "b"]')


def test_generation_user_includes_the_passage_the_section_and_the_title():
    prompt = generation_user(PASSAGE, "paraphrase", title="Violin")
    assert PASSAGE.text in prompt
    assert "Construction" in prompt
    assert "Violin" in prompt


def test_generation_user_labels_a_headingless_passage_as_the_introduction():
    lead = Passage(note="a/x.md", heading="", text="Body.", offset=0, note_chars=100)
    assert "(introduction)" in generation_user(lead, "paraphrase", title="X")


def test_exact_term_prompt_names_the_rare_terms():
    prompt = generation_user(PASSAGE, "exact_term", title="Violin")
    assert "ponticello" in prompt and "tasto" in prompt


def test_cross_lingual_prompt_names_the_target_language():
    prompt = generation_user(PASSAGE, "cross_lingual", title="Violin", lang_name="Italian")
    assert "Italian" in prompt
    assert "Spanish" not in prompt


def test_generation_user_rejects_an_unknown_type():
    with pytest.raises(KeyError):
        generation_user(PASSAGE, "nonsense", title="Violin")


def test_judge_user_shows_the_query_and_span_but_not_the_passage():
    # The judge must decide whether the span ALONE answers the query. Showing
    # it the passage lets it answer from context the retrieval system will
    # never have, which is exactly the failure this gate exists to catch.
    prompt = judge_user(query="how is it tuned", span="tuned in fifths", title="Violin")
    assert "how is it tuned" in prompt
    assert "tuned in fifths" in prompt
    assert PASSAGE.text not in prompt


def test_system_prompts_demand_json_only():
    assert "JSON" in GENERATION_SYSTEM
    assert "JSON" in JUDGE_SYSTEM


def test_judge_system_names_all_four_verdict_keys():
    for key in ("answers", "unambiguous", "title_only", "reason"):
        assert key in JUDGE_SYSTEM
