"""Validation stage 2.5: drop the queries that have more than one right answer.

`answer_spans` is a fixed list, and `span_metrics` counts a retrieved chunk as
a hit only when it sits in the labelled note *and* contains the labelled span.
So a query that some other passage answers just as well is a trap: a retriever
that ranks that other passage first is behaving correctly and is scored as a
**miss**. The case does not measure retrieval quality, it measures whether the
retriever happened to guess which of several valid answers a model wrote down.

Stage 2 already asks its judge whether a query is `unambiguous`, but it asks
blind -- the judge sees only the query and the span, never the rest of the
corpus, so it can only catch queries that *read* ambiguously ("what is the
diameter?"), not queries that read fine and happen to be answered in three
other articles. That is a retrieval question, and this stage answers it with
retrieval: pull what the channels actually return for the query, strip the
chunks that would score as hits, and ask whether anything left over also
answers it.

Which is also why this cannot be a human step. Asking a reviewer to certify
that no other passage in 79 notes answers a query is asking them to hold the
corpus in their head; the plan originally did ask for that, in wording that
inverted the test as well.
"""

from __future__ import annotations

from collections.abc import Callable

from ariostea.adapters.chat.openai_compat import ChatError
from ariostea.eval.gold_prompts import parse_json_object
from ariostea.eval.harness import SpanSearchFn
from ariostea.eval.span_metrics import chunk_contains_span
from ariostea.eval.wiki_gold import WikiGoldCase
from ariostea.ports.chat import ChatProvider

# How many competing passages to put in front of the judge. Enough to catch a
# genuine second answer, few enough that the prompt stays readable and the
# reasoning model does not spend its token budget summarising distractors.
COMPETITOR_LIMIT = 6

AMBIGUITY_SYSTEM = (
    "You audit evaluation data for a retrieval benchmark. You are given a query "
    "and a numbered list of passages drawn from the same corpus. Decide whether "
    "any passage on its own is a complete and correct answer to the query.\n"
    "Judge each passage on its own merits; being on the list is not evidence "
    "either way. A passage that is merely on the same topic, or that answers a "
    "different question about it, does not count -- it must actually answer "
    "*this* query.\n"
    "You reply with a single JSON object and nothing else, with exactly these "
    "keys:\n"
    '  "competes": true only if at least one listed passage fully answers the '
    "query\n"
    '  "which": the 1-based number of that passage, or 0 if none\n'
    '  "reason": one short sentence explaining your judgement\n'
)


def ambiguity_user(query: str, passages: tuple[str, ...]) -> str:
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(passages, start=1))
    return f"Query:\n{query}\n\nPassages:\n{numbered}\n"


def competing_passages(
    case: WikiGoldCase, channels: dict[str, SpanSearchFn], limit: int
) -> tuple[str, ...]:
    """Retrieved chunks for `case.query` that would *not* score as hits.

    The complement of the hit rule, deliberately: a chunk counts as a
    competitor unless it is in an answer span's own note and contains that
    span. That sweeps in sibling chunks of the correct note, which is the
    subtle half -- a chunk of the right article that does not carry the span
    scores exactly like a chunk of an unrelated one, so if it answers the query
    the case is just as unfair. Filtering by note path alone would miss them.

    Deduplicated by text, preserving the order channels returned them in, and
    capped at `limit`. Raises on an empty `channels` map, which would otherwise
    report "no competitors" for every case and turn the stage into a no-op that
    looks like a clean bill of health.
    """
    if not channels:
        raise ValueError("competing_passages needs at least one channel to search with")

    seen: set[str] = set()
    competitors: list[str] = []
    for search_fn in channels.values():
        for note, text in search_fn(case.query, limit):
            is_hit = any(
                span.note == note and chunk_contains_span(text, span.text)
                for span in case.answer_spans
            )
            if is_hit or text in seen:
                continue
            seen.add(text)
            competitors.append(text)
    return tuple(competitors[:limit])


# Prefix marking a verdict the judge produced but nobody can parse -- empty
# output, truncated JSON, a missing key. Still a rejection (see
# `ambiguity_gate`), but a parse failure is not a finding about the case, and
# filing it as ambiguity would let a flaky judge pass for a quality signal.
# Three of the 75 "ambiguous" drops on the first full run were this.
UNREADABLE_PREFIX = "ambiguity verdict unreadable:"


def ambiguity_gate(
    judge: ChatProvider, case: WikiGoldCase, passages: tuple[str, ...]
) -> str | None:
    """Return why `case` is ambiguous against `passages`, or `None` to keep it.

    With no competing passages there is nothing to be ambiguous against, so the
    case is approved without a model call -- spending one would only create an
    opportunity to be wrong.

    The judge is shown the query and the competitors, never the labelled span
    or note. It has to decide whether a passage answers the query on that
    passage's own merits; showing it the official answer invites it to rank the
    competitors against gold rather than judge them, which is the same
    self-agreement failure `adversarial_gate` avoids by using a second model.

    Every failure to read a verdict is a rejection, matching `adversarial_gate`:
    an unparseable or truncated response is not evidence the case is sound.
    It is not evidence of ambiguity either, so those reasons carry
    `UNREADABLE_PREFIX` and the runner files them under their own stage.
    """
    if not passages:
        return None

    raw = judge.complete(system=AMBIGUITY_SYSTEM, user=ambiguity_user(case.query, passages))
    try:
        verdict = parse_json_object(raw)
    except ValueError as exc:
        return f"{UNREADABLE_PREFIX} {exc}"

    if "competes" not in verdict:
        return f"{UNREADABLE_PREFIX} missing the 'competes' key"

    reason = str(verdict.get("reason", "")).strip()
    if verdict.get("competes"):
        return f"another passage answers the query as well ({reason})"
    return None


def collect_competitors(
    cases: list[WikiGoldCase],
    channels: dict[str, SpanSearchFn],
    limit: int = COMPETITOR_LIMIT,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[tuple[WikiGoldCase, tuple[str, ...]]]:
    """Gather each case's competing passages. Needs the index; makes no model calls.

    `on_progress(done, total)` is called after each case. Same reasoning as
    in `ambiguity_filter`: the hybrid channel reranks its whole pool on CPU
    for every query, which took over two hours on the real set, and a stage
    that is silent for that long looks hung. One healthy run was killed on
    exactly that suspicion.

    Split from `ambiguity_filter` on purpose, and the reason is memory rather
    than tidiness. Retrieval holds an embedding model and a store open; judging
    holds a 35B reasoning model open. On the machine this was built for, those
    two together are what SIGKILLed two earlier runs -- and SIGKILL cannot be
    caught, so the run dies with nothing written. Collecting first and judging
    afterwards means the index is closed before the judge is ever called, so
    the peak is one of them rather than their sum.
    """
    collected: list[tuple[WikiGoldCase, tuple[str, ...]]] = []
    for index, case in enumerate(cases, start=1):
        collected.append((case, competing_passages(case, channels, limit)))
        if on_progress is not None:
            on_progress(index, len(cases))
    return collected


# Prefix marking a drop caused by the endpoint rather than by the case. The
# runner files these under a separate stage so an outage cannot be read as a
# quality signal; `unreachable_count` is what turns them into a non-zero exit.
UNREACHABLE_PREFIX = "judge unreachable:"


def ambiguity_filter(
    collected: list[tuple[WikiGoldCase, tuple[str, ...]]],
    judge: ChatProvider,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[list[WikiGoldCase], list[tuple[WikiGoldCase, str]]]:
    """Split pre-collected cases into `(kept, [(dropped, reason)])`, in order.

    Takes the output of `collect_competitors` rather than channels, so it needs
    no index of its own -- see that function for why the two are separated.

    `on_progress(done, total, kept)` is called after each verdict. Not
    decoration: judging the real set takes over half an hour of model calls,
    and a stage that prints nothing while it works is indistinguishable from a
    stage that has hung -- this pipeline has already lost days to that
    ambiguity once, and two runs were stopped mid-judging while they were in
    fact making steady progress.

    A `ChatError` drops the case with `UNREACHABLE_PREFIX` rather than aborting
    the run or being recorded as ambiguity. An endpoint that cannot load the
    model says nothing about the case, and this pipeline has already been
    bitten once by an outage being laundered into quality rejections -- 72
    candidates filed as judged when the judge had never started.
    """
    kept: list[WikiGoldCase] = []
    dropped: list[tuple[WikiGoldCase, str]] = []
    for index, (case, passages) in enumerate(collected, start=1):
        try:
            reason = ambiguity_gate(judge, case, passages)
        except ChatError as exc:
            reason = f"{UNREACHABLE_PREFIX} {exc}"
        if reason is None:
            kept.append(case)
        else:
            dropped.append((case, reason))
        if on_progress is not None:
            on_progress(index, len(collected), len(kept))
    return kept, dropped
