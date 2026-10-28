"""Pure helpers for the search-speed sweep (eval/run_rerank_sweep.py): what a
reranker configuration costs per search, and what it does to search_sources."""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Sequence


def search_latency_ms(
    search: Callable[[str], object],
    queries: Sequence[str],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, float]:
    """Median and 95th percentile milliseconds per search, after one untimed
    warm-up call, like `median_query_ms` in chunk_sweep.

    The 95th percentile is the nearest rank, so it is always a timing that
    was observed."""
    if not queries:
        raise ValueError("no queries to time")
    search(queries[0])
    timings = []
    for query in queries:
        start = clock()
        search(query)
        timings.append((clock() - start) * 1000)
    p95 = sorted(timings)[math.ceil(0.95 * len(timings)) - 1]
    return {
        "search_ms_median": round(statistics.median(timings), 1),
        "search_ms_p95": round(p95, 1),
    }


def short_sources(
    sources: Callable[[str, int], Sequence[object]], queries: Sequence[str], k: int = 10
) -> int:
    """How many queries get fewer than `k` notes back from `sources`: with a
    small rerank pool, search_sources groups too few chunks to fill its list."""
    return sum(1 for query in queries if len(sources(query, k)) < k)
