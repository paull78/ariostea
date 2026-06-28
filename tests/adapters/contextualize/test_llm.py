from ariostea.adapters.contextualize.llm import LLMContextualizer
from ariostea.domain.models import Chunk, Note


def _note():
    return Note(path="a.md", title="A", frontmatter={}, tags=(), wikilinks=(), content_hash="h", mtime=1.0)


def _chunk(ordinal, text):
    return Chunk(note_path="a.md", ordinal=ordinal, heading_path=("A",), text=text, token_count=len(text.split()))


class FakeChat:
    def __init__(self, reply="a situating blurb"):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


class BrokenChat:
    def complete(self, system, user):
        raise RuntimeError("provider down")


def test_blurb_is_prepended_to_every_chunk():
    chat = FakeChat("ACME Q2 report")
    ctx = LLMContextualizer(chat, model_name="m")

    out = ctx.contextualize(_note(), "the full document", [_chunk(0, "alpha"), _chunk(1, "beta")])

    assert [c.embedding_text for c in out] == ["ACME Q2 report\n\nalpha", "ACME Q2 report\n\nbeta"]
    assert all(c.context_blurb == "ACME Q2 report" for c in out)
    # the full document (not the chunk) is sent as the user content, once
    assert len(chat.calls) == 1
    assert chat.calls[0][1] == "the full document"


def test_empty_blurb_degrades_to_plain_text():
    ctx = LLMContextualizer(FakeChat("   "), model_name="m")
    out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha")])
    assert out[0].embedding_text == "alpha"
    assert out[0].context_blurb is None


def test_provider_failure_degrades_to_plain_text():
    ctx = LLMContextualizer(BrokenChat(), model_name="m")
    out = ctx.contextualize(_note(), "doc", [_chunk(0, "alpha"), _chunk(1, "beta")])
    assert [c.embedding_text for c in out] == ["alpha", "beta"]
    assert all(c.context_blurb is None for c in out)


def test_provider_failure_logs_a_warning(caplog):
    import logging

    ctx = LLMContextualizer(BrokenChat(), model_name="m")
    with caplog.at_level(logging.WARNING):
        ctx.contextualize(_note(), "doc", [_chunk(0, "alpha")])
    assert "a.md" in caplog.text and "indexing plain" in caplog.text


def test_fingerprint_includes_model():
    assert LLMContextualizer(FakeChat(), model_name="gpt-4o-mini").fingerprint == "llm:gpt-4o-mini"
