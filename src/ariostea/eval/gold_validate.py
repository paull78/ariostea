"""The validation gates a generated candidate must survive.

Stage 1 (`automatic_gate`) is mechanical and free: everything decidable by
comparing strings. Stage 2 (`adversarial_gate`) costs a model call and only
ever sees candidates stage 1 has already accepted, because there is nothing
worth paying a judge to read about a span that is not even in the note.

Every check returns a *reason string* rather than raising or returning a
bool. The reasons are written to `eval/wiki/gold_rejected.json`, and a gate
whose rejections cannot be read is a gate nobody can tune: the first real run
is expected to reject a large fraction, and reading why is the only way to
tell "the model is bad at this" from "my threshold is wrong".
"""

from __future__ import annotations

import re

from ariostea.eval.gold_generate import Candidate
from ariostea.eval.normalize import normalize_ws

MIN_SPAN_CHARS = 10
MAX_SPAN_CHARS = 300
# A query sharing this fraction of its content words with the article title is
# a restatement of the title, and any chunk of the note answers it.
TITLE_OVERLAP = 0.8

_CONTENT_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def _content_tokens(text: str) -> set[str]:
    return {word.lower() for word in _CONTENT_WORD.findall(text)}


def automatic_gate(
    candidate: Candidate, notes: dict[str, str], titles: dict[str, str]
) -> str | None:
    """Return why `candidate` is unusable, or `None` if it survives stage 1.

    Checks run most-fundamental first, so the reported reason is the most
    informative one available: a span that is neither in the note nor long
    enough should be reported as fabricated, not as short.
    """
    if candidate.note not in notes:
        return f"note {candidate.note!r} not in corpus"

    span = normalize_ws(candidate.span)
    if span not in normalize_ws(candidate.passage):
        return "span is not verbatim in the passage the model was shown"
    if span not in normalize_ws(notes[candidate.note]):
        return "span is not verbatim in the cited note"
    if len(candidate.span) < MIN_SPAN_CHARS:
        return f"span is shorter than {MIN_SPAN_CHARS} characters"
    if len(candidate.span) > MAX_SPAN_CHARS:
        return f"span is longer than {MAX_SPAN_CHARS} characters"

    title_tokens = _content_tokens(titles[candidate.note])
    query_tokens = _content_tokens(candidate.query)
    if not query_tokens:
        return "query has no content words"
    if len(query_tokens & title_tokens) / len(query_tokens) >= TITLE_OVERLAP:
        return "query restates the article title"

    span_tokens = _content_tokens(candidate.span)
    if span_tokens and span_tokens <= title_tokens:
        return "span adds nothing beyond the article title"

    if candidate.type == "cross_lingual" and query_tokens <= _content_tokens(candidate.passage):
        # Every content word of the query already appears in the English
        # passage, so the query was written in English despite the
        # instruction. A real Italian or Spanish query always contributes
        # words the English text does not have.
        return "cross_lingual query is not in another language"

    return None
