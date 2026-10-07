import logging

from ariostea.adapters.contextualize.llm_chunk import LLMChunkContextualizer
from ariostea.domain.models import Chunk, Note, contextual_text
from ariostea.ports.pipeline import Contextualizer


def _note():
    return Note(
        path="a.md", title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=1.0
    )


def _chunk(ordinal, text):
    return Chunk(
        note_path="a.md",
        ordinal=ordinal,
        heading_path=("A",),
        text=text,
        token_count=len(text.split()),
    )


class FakeChat:
    """Records (system, user) pairs and returns scripted answers in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_system_is_identical_and_holds_the_full_document():
    chat = FakeChat(["blurb one", "blurb two"])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    ctx.contextualize(_note(), "the full document", [_chunk(0, "alpha"), _chunk(1, "beta")])

    systems = [s for s, _ in chat.calls]
    assert systems[0] == systems[1]
    assert "<document>\nthe full document\n</document>" in systems[0]


def test_user_holds_the_chunk_and_language_instruction():
    chat = FakeChat(["blurb one"])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    ctx.contextualize(_note(), "doc", [_chunk(0, "alpha")])

    _, user = chat.calls[0]
    assert "<chunk>\nalpha\n</chunk>" in user
    assert "same language as the chunk" in user


def test_one_call_per_chunk_in_order():
    chat = FakeChat(["c0", "c1", "c2"])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    out = ctx.contextualize(
        _note(), "doc", [_chunk(0, "alpha"), _chunk(1, "beta"), _chunk(2, "gamma")]
    )

    assert len(chat.calls) == 3
    assert [c.chunk.ordinal for c in out] == [0, 1, 2]
    assert [c.context_blurb for c in out] == ["c0", "c1", "c2"]


def test_context_blurb_and_embedding_text():
    chat = FakeChat(["  a situating context  "])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha")])

    assert out[0].context_blurb == "a situating context"
    assert out[0].embedding_text == contextual_text("a situating context", "alpha")


def test_a_failed_call_degrades_only_its_own_chunk(caplog):
    chat = FakeChat(["good context", RuntimeError("boom"), "another good context"])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    with caplog.at_level(logging.WARNING):
        out = ctx.contextualize(
            _note(), "doc", [_chunk(0, "alpha"), _chunk(1, "beta"), _chunk(2, "gamma")]
        )

    assert out[0].context_blurb == "good context"
    assert out[1].context_blurb is None
    assert out[1].embedding_text == "beta"
    assert out[2].context_blurb == "another good context"
    assert "a.md" in caplog.text and "1" in caplog.text


def test_an_empty_answer_degrades_only_its_own_chunk(caplog):
    chat = FakeChat(["good context", "   "])
    ctx = LLMChunkContextualizer(chat, model_name="m")

    with caplog.at_level(logging.WARNING):
        out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha"), _chunk(1, "beta")])

    assert out[0].context_blurb == "good context"
    assert out[1].context_blurb is None
    assert out[1].embedding_text == "beta"
    assert "a.md" in caplog.text and "1" in caplog.text


def test_fingerprint_includes_model():
    assert LLMChunkContextualizer(FakeChat([]), model_name="gpt-4o-mini").fingerprint == (
        "llm-chunk:gpt-4o-mini"
    )


def test_is_a_contextualizer():
    assert isinstance(LLMChunkContextualizer(FakeChat([]), model_name="m"), Contextualizer)
