import logging

import pytest

from ariostea.config.schema import ChunkingCfg, Config, EmbeddingCfg, StoreCfg, VaultCfg
from ariostea.eval.blurb_eval import WarnedError
from ariostea.eval.rerank_sweep import (
    guarded,
    parse_entry,
    rerank_config,
    rerank_run_config,
    rerank_run_id,
    search_latency_ms,
    short_sources,
)


class _Clock:
    """Returns the given instants in order, one per call."""

    def __init__(self, *instants):
        self._instants = list(instants)

    def __call__(self):
        return self._instants.pop(0)


def _durations_clock(*durations_ms):
    """A clock read twice per timed call, giving each call the next duration."""
    instants = []
    for i, ms in enumerate(durations_ms):
        instants += [float(i), i + ms / 1000]
    return _Clock(*instants)


def test_latency_warms_up_once_then_times_every_query():
    seen = []
    search_latency_ms(seen.append, ["a", "b", "c", "d"], _durations_clock(1, 1, 1, 1))
    assert seen == ["a", "a", "b", "c", "d"]


def test_latency_median_and_p95_from_injected_clock():
    # 20 searches taking 10, 20, ..., 200 ms. Nearest rank: the 19th smallest.
    durations = [10 * n for n in range(1, 21)]
    queries = [f"q{n}" for n in range(20)]
    latency = search_latency_ms(lambda q: None, queries, _durations_clock(*durations))
    assert latency == {"search_ms_median": 105.0, "search_ms_p95": 190.0}


def test_latency_rounds_to_one_decimal():
    latency = search_latency_ms(lambda q: None, ["a"], _durations_clock(12.3456))
    assert latency == {"search_ms_median": 12.3, "search_ms_p95": 12.3}


def test_latency_rejects_no_queries():
    with pytest.raises(ValueError):
        search_latency_ms(lambda q: None, [], _Clock())


def test_short_sources_counts_queries_below_k():
    counts = iter([10, 3, 10, 0])
    asked = []

    def sources(query, k):
        asked.append(k)
        return [object()] * next(counts)

    assert short_sources(sources, ["a", "b", "c", "d"]) == 2
    assert asked == [10, 10, 10, 10]


JINA = "jinaai/jina-reranker-v2-base-multilingual"


def test_run_id_names_model_and_pool():
    assert rerank_run_id("2026-10-28", JINA, 50) == (
        "2026-10-28-rerank-jina-reranker-v2-base-multilingual-pool50"
    )


def test_rerank_config_changes_only_the_reranker():
    base = Config(
        vault=VaultCfg(path="/v", ignore=[]),
        store=StoreCfg(backend="sqlite", path="/tmp/x.db"),
        chunking=ChunkingCfg(max_tokens=128, overlap=16),
    )
    changed = rerank_config(base, "cross-encoder/x", 20)
    assert changed.rerank.model == "cross-encoder/x"
    assert changed.rerank.pool == 20
    assert changed.rerank.enabled
    for field in ("vault", "store", "chunking", "embedding", "contextual", "search"):
        assert getattr(changed, field) == getattr(base, field)


def test_run_config_records_rerank_model_and_pool():
    config = rerank_run_config(5, JINA, 30)
    assert config["rerank"] == {"model": JINA, "pool": 30}
    assert config["k"] == 5
    assert config["embedding"] == EmbeddingCfg().local_model
    assert config["chunking"] == ChunkingCfg().model_dump()


@pytest.mark.parametrize("bad", ["jina:abc", "jina", ":50", "jina:0", "jina:-5"])
def test_parse_grid_rejects_bad_entries(bad):
    with pytest.raises(ValueError):
        parse_entry(bad)


def test_parse_grid_splits_on_the_last_colon():
    assert parse_entry("jinaai/x:20") == ("jinaai/x", 20)


def test_guarded_aborts_when_the_container_warns():
    def build_and_measure():
        logging.getLogger("ariostea.config.container").warning("reranker unavailable")
        return "scores"

    with pytest.raises(WarnedError):
        guarded(build_and_measure)


def test_guarded_returns_the_result_without_warnings():
    assert guarded(lambda: "scores") == "scores"
