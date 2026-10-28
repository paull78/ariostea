import pytest

from ariostea.eval.rerank_sweep import search_latency_ms, short_sources


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
