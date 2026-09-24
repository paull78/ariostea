"""The generation and judge prompts, and the reader that survives what local
instruct models actually return.

`parse_json_object` is deliberately forgiving. `OpenAICompatChat` has no
`response_format` support, the endpoints this runs against are local models
rather than a hosted API with a strict JSON mode, and every unparsed response
is a candidate silently lost from a ~150-query budget. Strictness would buy
nothing here: a malformed response is not more correct for being rejected on
a technicality, and the checks that matter -- is the span real, does it answer
the query -- all come later.
"""

from __future__ import annotations

import json
import re

from ariostea.eval.gold_passages import Passage

# qwen3-family models emit their reasoning in a <think> block before the
# answer. Stripped before anything else, because that block routinely contains
# braces and would otherwise be mistaken for the JSON object.
_THINK = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(raw: str) -> dict:
    """Extract the single JSON object from a chat response.

    Handles, in order: a reasoning block, a ``` fence, and surrounding prose
    (by taking the span between the first `{` and the last `}`). Raises
    `ValueError` -- which `json.JSONDecodeError` already subclasses -- so a
    caller has one exception type to catch for every unusable response.
    """
    text = _THINK.sub("", raw).strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"response is not a JSON object: {raw[:200]!r}")
    return parsed


GENERATION_SYSTEM = (
    "You write evaluation queries for a document retrieval system. You are given "
    "one passage from a Wikipedia article. You reply with a single JSON object and "
    "nothing else, with exactly these keys:\n"
    '  "query": the question a user would type into a search box\n'
    '  "answer_span": text copied VERBATIM and CONTIGUOUSLY from the passage, '
    "between 10 and 200 characters, that answers the query\n"
    "\n"
    "Rules that override every other instruction:\n"
    "1. answer_span must be an exact substring of the passage. Copy it character by "
    "character. Do not paraphrase it, do not correct its punctuation or spelling, "
    "and do not stitch together two pieces of text that are not adjacent.\n"
    "2. The query must be answerable only from the passage. Someone who knows just "
    "the article title must not be able to answer it.\n"
    "3. The query must not quote the article title verbatim.\n"
    "4. Ask about one fact, with one correct answer.\n"
)

_TYPE_INSTRUCTIONS = {
    "paraphrase": (
        "Write the query as a restatement that shares as few words with the passage "
        "as you can manage. Use synonyms throughout. A keyword search for your query "
        "should struggle to find this passage; only a meaning-based match should."
    ),
    "exact_term": (
        "Build the query around one of these rare technical terms, spelled exactly as "
        "given: {rare}. The query must hinge on that term -- someone who does not know "
        "the term cannot answer the query."
    ),
    "buried": (
        "This passage sits deep inside a long article, far from what its title "
        "suggests. Ask about the specific fact this passage states, not about the "
        "article's main subject, and do not mention the article title."
    ),
    "cross_lingual": (
        "Write the query in {lang_name}, not in English, even though the passage is in "
        "English. Use natural, fluent {lang_name} of the kind a native speaker would "
        "type. The answer_span stays in English, copied verbatim from the passage."
    ),
}


def generation_user(
    passage: Passage, query_type: str, title: str, lang_name: str = "Italian"
) -> str:
    """The per-passage user turn.

    Raises `KeyError` on an unknown query type. A typo must not quietly fall
    back to a generic prompt: the result would be a case labelled `exact_term`
    that stresses nothing in particular, which is worse than no case at all
    because the per-type breakdown would still report it.
    """
    instruction = _TYPE_INSTRUCTIONS[query_type].format(
        rare=", ".join(passage.rare_terms[:5]), lang_name=lang_name
    )
    return (
        f"Article title: {title}\n"
        f"Section: {passage.heading or '(introduction)'}\n"
        f"\n"
        f"Passage:\n{passage.text}\n"
        f"\n"
        f"Task: {instruction}\n"
        f"Reply with the JSON object only."
    )


JUDGE_SYSTEM = (
    "You audit evaluation data for a retrieval benchmark. You are strict: when in "
    "doubt, reject. You reply with a single JSON object and nothing else, with "
    "exactly these keys:\n"
    '  "answers": true only if the answer span, read on its own with no other '
    "context, answers the query\n"
    '  "unambiguous": true only if the query has one clear reading and one correct '
    "answer\n"
    '  "title_only": true if someone who knows only the article title, and has not '
    "read the span, could already answer the query\n"
    '  "reason": one short sentence explaining your judgement\n'
)


def judge_user(query: str, span: str, title: str) -> str:
    """The judge's user turn.

    It shows the query, the span and the article title -- and deliberately not
    the passage the span came from. The question this gate answers is whether
    the *span alone* answers the query, because a retrieved chunk is all the
    span-level metric will ever credit. A judge that could read the
    surrounding passage would approve spans that only make sense in context,
    which is the exact failure this stage exists to catch.
    """
    return (
        f"Article title: {title}\n"
        f"Query: {query}\n"
        f"Answer span: {span}\n"
        f"\n"
        f"Judge the query and span against your four keys. Reply with the JSON "
        f"object only."
    )
