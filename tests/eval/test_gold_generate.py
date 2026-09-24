import pytest

from ariostea.eval.gold_generate import Candidate, generate_case
from ariostea.eval.gold_passages import Passage
from ariostea.eval.gold_prompts import GENERATION_SYSTEM

PASSAGE = Passage(
    note="strings/violin.md",
    heading="Tuning",
    text="The violin is tuned in perfect fifths: G, D, A, E.",
    offset=100,
    note_chars=9000,
    rare_terms=("ponticello",),
    rare_score=40.0,
)


class FakeChat:
    """A ChatProvider that returns canned responses and records its prompts."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def test_generate_case_builds_a_candidate_from_the_response():
    chat = FakeChat('{"query": "how is a violin tuned", "answer_span": "perfect fifths"}')
    candidate = generate_case(chat, PASSAGE, "paraphrase", title="Violin")
    assert candidate == Candidate(
        query="how is a violin tuned",
        query_lang="en",
        type="paraphrase",
        note="strings/violin.md",
        passage=PASSAGE.text,
        span="perfect fifths",
    )


def test_generate_case_sends_the_generation_system_prompt_and_the_passage():
    chat = FakeChat('{"query": "q about tuning", "answer_span": "perfect fifths"}')
    generate_case(chat, PASSAGE, "paraphrase", title="Violin")
    system, user = chat.calls[0]
    assert system == GENERATION_SYSTEM
    assert PASSAGE.text in user


def test_generate_case_marks_cross_lingual_cases_with_the_query_language():
    chat = FakeChat('{"query": "come si accorda", "answer_span": "perfect fifths"}')
    candidate = generate_case(
        chat, PASSAGE, "cross_lingual", title="Violin", lang_name="Italian", query_lang="it"
    )
    assert candidate.query_lang == "it"


def test_generate_case_strips_surrounding_whitespace():
    chat = FakeChat('{"query": "  q  ", "answer_span": "  perfect fifths  "}')
    assert generate_case(chat, PASSAGE, "paraphrase", title="Violin").span == "perfect fifths"


def test_generate_case_raises_on_an_empty_query():
    chat = FakeChat('{"query": "", "answer_span": "perfect fifths"}')
    with pytest.raises(ValueError, match="empty query"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_on_a_missing_span():
    chat = FakeChat('{"query": "how is a violin tuned"}')
    with pytest.raises(ValueError, match="empty answer_span"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_on_an_unparseable_response():
    chat = FakeChat("I am sorry, I cannot do that.")
    with pytest.raises(ValueError, match="no JSON object"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_when_the_span_is_a_list():
    # A model returning ["a", "b"] would otherwise stringify to "['a', 'b']"
    # and fail the verbatim check later with a thoroughly confusing reason.
    chat = FakeChat('{"query": "q", "answer_span": ["a", "b"]}')
    with pytest.raises(ValueError, match="answer_span must be a string"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")


def test_generate_case_raises_when_the_query_is_a_number():
    chat = FakeChat('{"query": 42, "answer_span": "perfect fifths"}')
    with pytest.raises(ValueError, match="query must be a string"):
        generate_case(chat, PASSAGE, "paraphrase", title="Violin")
