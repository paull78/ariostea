"""Pure helpers for the search-speed sweep (eval/run_rerank_sweep.py): what a
reranker configuration costs per search, and what it does to search_sources."""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Sequence, TypeVar

from ariostea.config.schema import ChunkingCfg, Config, EmbeddingCfg
from ariostea.eval.blurb_eval import fail_on_warnings
from ariostea.eval.chunk_sweep import model_slug

# The logger build_container warns on when the reranker falls back to fused
# order (search-speed.14).
CONTAINER_LOGGER = "ariostea.config.container"

T = TypeVar("T")


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


def parse_entry(text: str) -> tuple[str, int]:
    """`MODEL:POOL` -> (model, pool). Split on the last colon, so a model name
    may hold one; the pool must be a positive whole number."""
    model, sep, pool = text.rpartition(":")
    if not sep or not model or not pool.isdigit() or int(pool) < 1:
        raise ValueError(f"expected MODEL:POOL with a positive pool, got {text!r}")
    return model, int(pool)


def rerank_run_id(date: str, model: str, pool: int) -> str:
    return f"{date}-rerank-{model_slug(model)}-pool{pool}"


def rerank_config(config: Config, model: str, pool: int) -> Config:
    """`config` with the reranker's model and pool replaced, nothing else."""
    rerank = config.rerank.model_copy(update={"model": model, "pool": pool})
    return config.model_copy(update={"rerank": rerank})


def rerank_run_config(k: int, model: str, pool: int) -> dict:
    """The `config` a sweep run logs: the default embedding and chunking every
    run shares, and the reranker configuration it measures."""
    return {
        "k": k,
        "embedding": EmbeddingCfg().local_model,
        "chunking": ChunkingCfg().model_dump(),
        "rerank": {"model": model, "pool": pool},
    }


def guarded(run: Callable[[], T]) -> T:
    """`run()`, aborting with WarnedError if the container warned meanwhile: a
    reranker that fell back to fused order would log fused numbers under its
    own name."""
    with fail_on_warnings(CONTAINER_LOGGER):
        return run()
