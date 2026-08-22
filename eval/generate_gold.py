"""Generate the span-anchored gold set over the pinned Wikipedia corpus.

Usage:  uv run python eval/generate_gold.py [--limit N] [--no-discrimination]

Pipeline, per selected passage:

    select -> generate -> stage 1 automatic -> stage 2 adversarial judge
                                                   |
                        stage 3 discrimination <---+
                                  |
                        eval/wiki/gold.json

Point it at a running OpenAI-compatible endpoint with:
    ARIOSTEA_GOLD_BASE_URL    (default http://localhost:1234/v1, LM Studio)
    ARIOSTEA_GOLD_MODEL       (default qwen2.5-14b-instruct-mlx)
    ARIOSTEA_GOLD_JUDGE_MODEL (default qwen/qwen3.6-35b-a3b)
    ARIOSTEA_GOLD_API_KEY     (default empty)

The judge model must differ from the generator. A model asked to audit its
own output agrees with itself, which turns stage 2 from a gate into a rubber
stamp -- the script refuses to run when the two names match.

The reasoning model judges and the plain instruct model generates, not the
other way round. Generation is mechanical -- copy a span, phrase a question --
and measured on this corpus the 14B instruct model got 6/6 spans verbatim at
7.5s a call, while the 35B reasoning model spent ~2800 thinking tokens and
~47s reaching the same kind of answer. Judging is the half that benefits from
deliberation, and it is also the smaller half, since only candidates that
survived stage 1 reach it.

Outputs, all under eval/wiki/:
    gold.json           the accepted cases, in the Plan 1 schema
    gold_rejected.json  every rejected candidate, with its stage and reason
    gold.meta.json      which models produced this gold, and the counts
    gold_review.md      a sample rendered for stage 4, human spot-review

Reproducibility differs from the corpus build on purpose.
`build_wiki_corpus.py` reproduces byte-identical output from pinned revision
ids; an LLM run cannot, even at temperature 0, across model builds. So the
*committed gold* is the artifact of record and `gold.meta.json` records what
produced it. Re-running this script writes a new gold set; it does not verify
the old one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ariostea.adapters.chat.openai_compat import ChatError, OpenAICompatChat
from ariostea.eval.chat_cache import CachingChat
from ariostea.eval.gold_discriminate import discrimination_filter
from ariostea.eval.gold_generate import Candidate, generate_case
from ariostea.eval.gold_passages import Passage, select_passages
from ariostea.eval.gold_validate import adversarial_gate, automatic_gate
from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase
from ariostea.eval.wiki_index import index_wiki_corpus, wiki_channels
from ariostea.eval.wiki_notes import load_corpus_notes, note_titles
from ariostea.ports.chat import ChatProvider

WIKI_DIR = Path(__file__).resolve().parent / "wiki"
GOLD = WIKI_DIR / "gold.json"
REJECTED = WIKI_DIR / "gold_rejected.json"
META = WIKI_DIR / "gold.meta.json"
REVIEW = WIKI_DIR / "gold_review.md"
# Raw model responses, keyed by prompt. Makes an interrupted run resumable and
# makes re-running with tuned gates free. Not committed -- it is a local
# scratch file, and `gold.json` is the artifact of record.
CACHE = WIKI_DIR / ".gold_cache.jsonl"

# ~150 queries, the design doc's Medium tier. cross_lingual is smaller because
# it is the most expensive to review by hand.
BUDGET = {"paraphrase": 40, "exact_term": 40, "buried": 40, "cross_lingual": 30}

# Cross-lingual queries alternate between the two languages the corpus holds
# parallel articles in, so neither track ends up a footnote.
LANGUAGES = (("it", "Italian"), ("es", "Spanish"))

BASE_URL = os.environ.get("ARIOSTEA_GOLD_BASE_URL", "http://localhost:1234/v1")
MODEL = os.environ.get("ARIOSTEA_GOLD_MODEL", "qwen2.5-14b-instruct-mlx")
JUDGE_MODEL = os.environ.get("ARIOSTEA_GOLD_JUDGE_MODEL", "qwen/qwen3.6-35b-a3b")
API_KEY = os.environ.get("ARIOSTEA_GOLD_API_KEY", "")
# Generous next to the 128-token default, but a plain instruct model needs no
# more than this for a query plus a span.
GEN_MAX_TOKENS = 512
# The judge is a reasoning model, and `max_tokens` covers its thinking as well
# as its answer. Measured on real judge prompts it spends 1200-1500 tokens
# reasoning; at 1024 *every* verdict came back empty. That failed safe rather
# than silently -- `adversarial_gate` treats an unreadable verdict as a
# rejection -- but it rejected everything, so the budget has to clear the
# thinking with room to spare.
JUDGE_MAX_TOKENS = 4096
TIMEOUT_S = 600.0
REVIEW_SAMPLE = 20


@dataclass(frozen=True)
class Rejection:
    stage: str  # "generate" | "automatic" | "adversarial" | "discrimination"
    reason: str
    query: str
    note: str
    span: str
    type: str


def _scenario(query_type: str, query_lang: str) -> str:
    """Scenario label in the existing gold sets' vocabulary: the query type
    for same-language cases, an arrow for cross-lingual ones."""
    return f"en→{query_lang}" if query_type == "cross_lingual" else query_type


def to_gold_case(candidate: Candidate) -> WikiGoldCase:
    """`Candidate` already carries the query language `generate_case` was told
    to use, so it is read from there rather than passed again -- two sources
    for one fact is two things that can disagree."""
    return WikiGoldCase(
        query=candidate.query,
        query_lang=candidate.query_lang,
        type=candidate.type,
        scenario=_scenario(candidate.type, candidate.query_lang),
        expected_notes=(candidate.note,),
        answer_spans=(AnswerSpan(note=candidate.note, text=candidate.span),),
    )


def _generate_all(
    chat: ChatProvider,
    selected: list[tuple[str, Passage]],
    notes: dict[str, str],
    titles: dict[str, str],
) -> tuple[list[Candidate], list[Rejection]]:
    """Stage 0 and stage 1 over every selected passage.

    A `ChatError` is recorded as a rejection rather than allowed to abort. One
    model call failing mid-run should cost one candidate, not the hundred
    already generated.
    """
    survivors: list[Candidate] = []
    rejections: list[Rejection] = []
    cross_lingual_seen = 0

    for index, (query_type, passage) in enumerate(selected, start=1):
        if query_type == "cross_lingual":
            query_lang, lang_name = LANGUAGES[cross_lingual_seen % len(LANGUAGES)]
            cross_lingual_seen += 1
        else:
            query_lang, lang_name = "en", "Italian"  # lang_name unused for en types

        title = titles[passage.note]
        try:
            candidate = generate_case(
                chat,
                passage,
                query_type,
                title=title,
                lang_name=lang_name,
                query_lang=query_lang,
            )
        except (ValueError, ChatError) as exc:
            rejections.append(Rejection("generate", str(exc), "", passage.note, "", query_type))
            continue

        reason = automatic_gate(candidate, notes, titles)
        if reason:
            rejections.append(
                Rejection(
                    "automatic",
                    reason,
                    candidate.query,
                    candidate.note,
                    candidate.span,
                    query_type,
                )
            )
            continue
        survivors.append(candidate)
        if index % 10 == 0:
            print(f"  generated {index}/{len(selected)}, {len(survivors)} past stage 1", flush=True)

    return survivors, rejections


def _judge_all(
    judge: ChatProvider, survivors: list[Candidate], titles: dict[str, str]
) -> tuple[list[WikiGoldCase], list[Rejection]]:
    """Stage 2 over the candidates stage 1 accepted."""
    cases: list[WikiGoldCase] = []
    rejections: list[Rejection] = []
    for index, candidate in enumerate(survivors, start=1):
        try:
            reason = adversarial_gate(judge, candidate, title=titles[candidate.note])
        except ChatError as exc:
            reason = f"judge unreachable: {exc}"
        if reason:
            rejections.append(
                Rejection(
                    "adversarial",
                    reason,
                    candidate.query,
                    candidate.note,
                    candidate.span,
                    candidate.type,
                )
            )
        else:
            cases.append(to_gold_case(candidate))
        if index % 10 == 0:
            print(f"  judged {index}/{len(survivors)}, {len(cases)} approved", flush=True)
    return cases, rejections


def generate_and_gate(
    chat: ChatProvider,
    judge: ChatProvider,
    selected: list[tuple[str, Passage]],
    notes: dict[str, str],
    titles: dict[str, str],
) -> tuple[list[WikiGoldCase], list[Rejection]]:
    """Run generation and validation stages 1 and 2 over every selected passage.

    The two stages run as separate passes rather than interleaved per
    candidate, and that is about hardware, not tidiness. The generator and the
    judge are different models totalling 36GB on a 48GB machine, so LM Studio
    cannot hold both; alternating them per candidate would swap models roughly
    three hundred times over a 150-passage run. Two passes cost one swap.

    Stage 2 still only ever sees candidates stage 1 accepted: the judge costs
    a model call, and there is nothing worth judging about a span that is not
    even in the note.
    """
    survivors, rejections = _generate_all(chat, selected, notes, titles)
    cases, judge_rejections = _judge_all(judge, survivors, titles)
    return cases, rejections + judge_rejections


def write_gold(path: Path, cases: list[WikiGoldCase]) -> None:
    """Write `cases` in the Plan 1 schema `load_wiki_gold` reads back."""
    rows = [
        {
            "query": case.query,
            "query_lang": case.query_lang,
            "type": case.type,
            "scenario": case.scenario,
            "expected_notes": list(case.expected_notes),
            "answer_spans": [{"note": s.note, "text": s.text} for s in case.answer_spans],
        }
        for case in cases
    ]
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def review_markdown(cases: list[WikiGoldCase], sample_size: int) -> str:
    """Render a sample for stage 4, human spot-review.

    Sampled by taking every nth case of each type rather than the first n:
    selection is round-robin by note, so the first cases of a type all come
    from the same handful of articles, and reviewing them would say more about
    those articles than about the gate.
    """
    lines = [
        "# Gold spot-review sample",
        "",
        "Tick a case only if **all three** hold: the query is answerable, the span "
        "answers it, and the span is not the only sentence in the corpus that "
        "plausibly could.",
        "",
    ]
    if not cases:
        lines.append("_No cases in the gold set to review._")
        return "\n".join(lines)

    by_type: dict[str, list[WikiGoldCase]] = {}
    for case in cases:
        by_type.setdefault(case.type, []).append(case)
    per_type = max(1, sample_size // len(by_type))

    for query_type in sorted(by_type):
        pool = by_type[query_type]
        step = max(1, len(pool) // per_type)
        sample = pool[::step][:per_type]
        lines += [f"## {query_type}  ({len(pool)} cases, showing {len(sample)})", ""]
        for case in sample:
            span = case.answer_spans[0]
            lines += [
                f"- [ ] **{case.query}**  `{case.query_lang}`",
                f"  - note: `{span.note}`",
                f"  - span: {span.text}",
                "",
            ]
    return "\n".join(lines)


def rejection_summary(rejections: list[Rejection]) -> str:
    """Counts by stage, then by reason within each stage.

    Printed after every run because the first real run is expected to reject a
    large fraction, and reading *why* is the only way to tell a model that is
    bad at the task from a threshold that is set wrong.
    """
    by_stage: dict[str, dict[str, int]] = {}
    for rejection in rejections:
        # Judge reasons carry the model's free text in parentheses; group on
        # the part before it, so the counts are about causes not phrasings.
        cause = rejection.reason.split("(")[0].strip()
        by_stage.setdefault(rejection.stage, {}).setdefault(cause, 0)
        by_stage[rejection.stage][cause] += 1

    lines: list[str] = []
    for stage in sorted(by_stage):
        lines.append(f"  {sum(by_stage[stage].values()):4d}  {stage}")
        for cause, count in sorted(by_stage[stage].items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"        {count:4d}  {cause}")
    return "\n".join(lines)


def report_shortfall(budget: dict[str, int], selected: list[tuple[str, Passage]]) -> None:
    """Print any query type the corpus could not fill.

    A silent shortfall reads as "the corpus supports this design" when it does
    not -- the same no-silent-caps rule the corpus build's dropped-template
    report follows.
    """
    for query_type, wanted in sorted(budget.items()):
        got = sum(1 for selected_type, _ in selected if selected_type == query_type)
        if got < wanted:
            print(
                f"  SHORTFALL {query_type}: only {got}/{wanted} eligible passages in the corpus",
                file=sys.stderr,
            )


def _write_outputs(
    cases: list[WikiGoldCase], rejections: list[Rejection], passages_selected: int
) -> None:
    """Write all four artifacts. Called from a `finally`, so it must cope with
    a partial run -- an empty `cases` list writes an empty gold file rather
    than raising, and `run_wiki_eval.py` refuses to score that."""
    write_gold(GOLD, cases)
    REJECTED.write_text(
        json.dumps([asdict(r) for r in rejections], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    META.write_text(
        json.dumps(
            {
                "generator_model": MODEL,
                "judge_model": JUDGE_MODEL,
                "base_url": BASE_URL,
                "passages_selected": passages_selected,
                "accepted": len(cases),
                "rejected": len(rejections),
                "by_type": {
                    query_type: sum(1 for case in cases if case.type == query_type)
                    for query_type in sorted(BUDGET)
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    REVIEW.write_text(review_markdown(cases, REVIEW_SAMPLE), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the wiki eval gold set.")
    parser.add_argument("--limit", type=int, help="cap the total passages (for a smoke run)")
    parser.add_argument(
        "--no-discrimination",
        action="store_true",
        help="skip stage 3, which has to index the corpus",
    )
    args = parser.parse_args(argv)

    if MODEL == JUDGE_MODEL:
        print(
            f"generator and judge are both {MODEL!r}; stage 2 needs an independent "
            f"model. Set ARIOSTEA_GOLD_JUDGE_MODEL to something else.",
            file=sys.stderr,
        )
        return 2

    notes = load_corpus_notes(WIKI_DIR)
    titles = note_titles(notes)
    budget = BUDGET
    if args.limit:
        share = max(1, args.limit // len(BUDGET))
        budget = {query_type: share for query_type in BUDGET}

    selected = select_passages(notes, budget)
    report_shortfall(budget, selected)
    print(f"{len(selected)} passages selected from {len({p.note for _, p in selected})} notes")

    chat = CachingChat(
        OpenAICompatChat(
            base_url=BASE_URL,
            model=MODEL,
            api_key=API_KEY,
            timeout=TIMEOUT_S,
            max_tokens=GEN_MAX_TOKENS,
        ),
        CACHE,
        label=MODEL,
    )
    judge = CachingChat(
        OpenAICompatChat(
            base_url=BASE_URL,
            model=JUDGE_MODEL,
            api_key=API_KEY,
            timeout=TIMEOUT_S,
            max_tokens=JUDGE_MAX_TOKENS,
        ),
        CACHE,
        label=JUDGE_MODEL,
    )
    print(f"generating with {MODEL}, judging with {JUDGE_MODEL} at {BASE_URL} ...", flush=True)

    cases: list[WikiGoldCase] = []
    rejections: list[Rejection] = []
    dropped: list[WikiGoldCase] = []
    try:
        cases, rejections = generate_and_gate(chat, judge, selected, notes, titles)
        print(f"{len(cases)} candidates survived stages 1 and 2", flush=True)

        if not args.no_discrimination and cases:
            with tempfile.TemporaryDirectory() as tmp:
                db = str(Path(tmp) / "eval.db")
                print("indexing the corpus for the discrimination filter ...", flush=True)
                container = index_wiki_corpus(WIKI_DIR, db)
                cases, dropped = discrimination_filter(cases, wiki_channels(db, container))
            rejections += [
                Rejection(
                    "discrimination",
                    "every channel answers at rank 1",
                    case.query,
                    case.expected_notes[0],
                    case.answer_spans[0].text,
                    case.type,
                )
                for case in dropped
            ]
            print(f"{len(dropped)} dropped as too easy; {len(cases)} remain", flush=True)
    finally:
        # In a `finally` for the same reason `build_wiki_corpus.py` writes its
        # manifest in one: the first full run was killed after every model
        # call had been paid for and before anything reached disk. The cache
        # makes those calls recoverable, but whatever this run did establish
        # should still be written down.
        _write_outputs(cases, rejections, len(selected))
        print(f"\ncache: {chat.hits + judge.hits} hits, {chat.misses + judge.misses} live calls")

    print("\nrejections by stage:")
    print("\nrejections by stage:")
    print(rejection_summary(rejections))
    print(f"\nwrote {len(cases)} cases to {GOLD}")
    print(f"spot-review sample: {REVIEW}")
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
