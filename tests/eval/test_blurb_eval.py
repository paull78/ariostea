import logging

import pytest

from ariostea.eval.blurb_eval import (
    ARMS,
    HYBRID_CONTEXT,
    BlurbCoverageError,
    arm_label,
    blurb_run_id,
    blurbs_by_note,
    contexts_by_chunk,
    control_mismatch,
    experiment_name,
    fail_on_warnings,
    index_marker,
    lmstudio_load_problem,
    needed_channels,
    require_full_coverage,
    select_arms,
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


def test_only_the_context_arm_scores_hybrid_with_the_blurb_aware_channel():
    by_key = {arm.key: arm for arm in ARMS}
    assert [arm.key for arm in ARMS if arm.use_context] == ["context-rerank"]
    assert by_key["context-rerank"].source("HYBRID") == HYBRID_CONTEXT
    assert by_key["raw-rerank"].source("HYBRID") == "HYBRID"
    assert by_key["fused"].source("FUSED") == "FUSED"


def test_needed_channels_covers_only_the_selected_arms():
    assert needed_channels(select_arms("fused")) == {"FUSED"}
    assert needed_channels(ARMS) == {"DENSE", "SPARSE", "HYBRID", "FUSED", HYBRID_CONTEXT}


def test_select_arms_keeps_arms_order_and_ignores_blanks():
    assert [a.key for a in select_arms("context-rerank, fused,")] == ["fused", "context-rerank"]
    assert select_arms(",".join(a.key for a in ARMS)) == ARMS


def test_select_arms_names_unknown_keys():
    with pytest.raises(ValueError, match="bogus"):
        select_arms("fused,bogus")


def test_select_arms_rejects_an_empty_selection():
    with pytest.raises(ValueError):
        select_arms(" , ")


CHUNKING = {"max_tokens": 160, "overlap": 40, "unit": "model_tokens"}


def _control(chunking=CHUNKING, embedding="model"):
    return {"id": "ctl", "config": {"chunking": chunking, "embedding": embedding}}


def test_a_matching_control_passes():
    assert control_mismatch(_control(), CHUNKING, "model") is None


def test_a_control_with_other_chunking_is_rejected():
    other = {**CHUNKING, "overlap": 0}
    reason = control_mismatch(_control(chunking=other), CHUNKING, "model")
    assert reason is not None and "ctl" in reason and "chunking" in reason


def test_a_control_with_another_embedding_is_rejected():
    reason = control_mismatch(_control(embedding="other"), CHUNKING, "model")
    assert reason is not None and "embedding" in reason


def test_a_control_without_config_is_rejected():
    assert control_mismatch({"id": "ctl"}, CHUNKING, "model") is not None


def test_blurbs_by_note_keeps_one_blurb_per_note():
    rows = [("b.md", "B"), ("a.md", "A"), ("a.md", "A"), ("c.md", None)]
    assert blurbs_by_note(rows) == {"a.md": "A", "b.md": "B"}


def test_differing_blurbs_for_one_note_raise():
    with pytest.raises(ValueError, match="a.md"):
        blurbs_by_note([("a.md", "A"), ("a.md", "other")])


def test_fail_on_warnings_passes_a_quiet_block():
    with fail_on_warnings("test.blurb.quiet"):
        logging.getLogger("test.blurb.quiet").info("fine")


def test_fail_on_warnings_raises_naming_the_warning():
    logger = logging.getLogger("test.blurb.loud")
    with pytest.raises(RuntimeError, match="reranker unavailable"):
        with fail_on_warnings("test.blurb.loud"):
            logger.warning("reranker unavailable (%s)", "offline")
    assert not any(type(h).__name__ == "_Collect" for h in logger.handlers)


def test_fail_on_warnings_lets_the_blocks_own_error_through():
    with pytest.raises(KeyError):
        with fail_on_warnings("test.blurb.error"):
            logging.getLogger("test.blurb.error").warning("also warned")
            raise KeyError("boom")


def test_note_level_run_ids_keep_their_logged_form():
    # runs.jsonl already holds 2026-09-29-blurbs-* entries.
    arm = select_arms("fused")[0]
    assert blurb_run_id("2026-09-29", arm) == "2026-09-29-blurbs-fused"
    assert blurb_run_id("2026-09-29", arm, "note") == "2026-09-29-blurbs-fused"


def test_per_chunk_run_ids_cannot_collide_with_note_level_ones_on_the_same_day():
    note = {blurb_run_id("2026-09-30", arm, "note") for arm in ARMS}
    chunk = {blurb_run_id("2026-09-30", arm, "chunk") for arm in ARMS}
    assert len(chunk) == len(ARMS)
    assert not note & chunk
    assert blurb_run_id("2026-09-30", ARMS[0], "chunk") == "2026-09-30-chunkctx-raw-rerank"


def test_experiment_names_per_granularity():
    assert experiment_name("note") == "Contextual blurbs"
    assert experiment_name("chunk") == "Per-chunk context"


def test_unknown_granularity_is_rejected():
    with pytest.raises(ValueError, match="paragraph"):
        experiment_name("paragraph")
    with pytest.raises(ValueError):
        blurb_run_id("2026-09-30", ARMS[0], "paragraph")
    with pytest.raises(ValueError):
        index_marker("paragraph", "m")


def test_arm_labels_name_per_chunk_contexts_in_chunk_mode():
    arm = select_arms("raw-rerank")[0]
    assert arm_label(arm, "note") == arm.label
    assert arm_label(arm, "chunk") == "Per-chunk context, reranker scores raw text"


def test_index_marker_matches_the_contextualizer_fingerprints():
    from ariostea.adapters.contextualize.llm_chunk import LLMChunkContextualizer

    assert index_marker("note", "qwen") == "llm:qwen"
    assert index_marker("chunk", "qwen") == "llm-chunk:qwen"
    assert LLMChunkContextualizer(chat=None, model_name="qwen").fingerprint == "llm-chunk:qwen"


def test_contexts_by_chunk_groups_by_note_and_ordinal_skipping_empty():
    rows = [
        ("a.md", 0, "ctx a0"),
        ("a.md", 1, None),
        ("a.md", 2, "ctx a2"),
        ("b.md", 0, ""),
        ("c.md", 0, "ctx c0"),
    ]
    assert contexts_by_chunk(rows) == {"a.md": {0: "ctx a0", 2: "ctx a2"}, "c.md": {0: "ctx c0"}}


def test_a_model_lm_studio_has_not_loaded_is_refused():
    # LM Studio would load it on the first request with its own 8k default.
    problem = lmstudio_load_problem({"state": "not-loaded"}, "qwen", 32768)
    assert problem and "not loaded" in problem and "qwen" in problem


def test_a_model_loaded_with_too_small_a_context_is_refused():
    info = {"state": "loaded", "loaded_context_length": 8192}
    assert "8192" in lmstudio_load_problem(info, "qwen", 32768)


def test_a_model_loaded_with_enough_context_passes():
    assert (
        lmstudio_load_problem({"state": "loaded", "loaded_context_length": 32768}, "q", 32768)
        is None
    )
    assert lmstudio_load_problem({"state": "loaded"}, "q", 32768) is None
