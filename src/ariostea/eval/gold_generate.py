"""One generation call: a passage and a query type in, a `Candidate` out.

Deliberately thin. Everything it could get wrong -- is the span really in the
note, is the query answerable from the title alone, does the span actually
answer the query -- belongs to the validation gates, not here. This module's
only job is to turn a chat response into a typed object or a `ValueError` the
caller can record as a rejection, so no unusable response reaches a gate
dressed as data.
"""

from __future__ import annotations

from dataclasses import dataclass

from ariostea.eval.gold_passages import Passage
from ariostea.eval.gold_prompts import GENERATION_SYSTEM, generation_user, parse_json_object
from ariostea.ports.chat import ChatProvider


@dataclass(frozen=True)
class Candidate:
    """A generated query before any gate has looked at it.

    `passage` is carried alongside `note` because the first automatic check is
    that the span is verbatim in the passage *the model was shown*, which is a
    sharper signal than "verbatim somewhere in the note": a span that is in
    the note but not in the passage means the model reproduced text from
    memory rather than copying it, and the next one may be a plausible
    fabrication instead.
    """

    query: str
    query_lang: str
    type: str
    note: str
    passage: str
    span: str


def _text_field(data: dict, key: str) -> str:
    value = data.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {type(value).__name__}")
    return value.strip()


def generate_case(
    chat: ChatProvider,
    passage: Passage,
    query_type: str,
    title: str,
    lang_name: str = "Italian",
    query_lang: str = "en",
) -> Candidate:
    """Ask `chat` for one query anchored to `passage`.

    Raises `ValueError` for any response that cannot become a candidate. The
    caller catches it and records a rejection; a run of ~150 passages against
    a local model will always produce a handful.
    """
    raw = chat.complete(
        system=GENERATION_SYSTEM,
        user=generation_user(passage, query_type, title=title, lang_name=lang_name),
    )
    data = parse_json_object(raw)
    query = _text_field(data, "query")
    span = _text_field(data, "answer_span")
    if not query:
        raise ValueError("empty query")
    if not span:
        raise ValueError("empty answer_span")
    return Candidate(
        query=query,
        query_lang=query_lang,
        type=query_type,
        note=passage.note,
        passage=passage.text,
        span=span,
    )
