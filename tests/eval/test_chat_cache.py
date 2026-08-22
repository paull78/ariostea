import json

from ariostea.eval.chat_cache import CachingChat


class CountingChat:
    def __init__(self, response: str = "answer") -> None:
        self.response = response
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return f"{self.response}-{self.calls}"


def test_a_first_call_reaches_the_inner_provider(tmp_path):
    inner = CountingChat()
    chat = CachingChat(inner, tmp_path / "c.jsonl", label="gen")
    assert chat.complete("s", "u") == "answer-1"
    assert inner.calls == 1


def test_a_repeated_call_is_served_from_memory(tmp_path):
    inner = CountingChat()
    chat = CachingChat(inner, tmp_path / "c.jsonl", label="gen")
    assert chat.complete("s", "u") == chat.complete("s", "u")
    assert inner.calls == 1


def test_a_different_prompt_is_a_different_entry(tmp_path):
    inner = CountingChat()
    chat = CachingChat(inner, tmp_path / "c.jsonl", label="gen")
    chat.complete("s", "u1")
    chat.complete("s", "u2")
    assert inner.calls == 2


def test_a_different_system_prompt_is_a_different_entry(tmp_path):
    inner = CountingChat()
    chat = CachingChat(inner, tmp_path / "c.jsonl", label="gen")
    chat.complete("s1", "u")
    chat.complete("s2", "u")
    assert inner.calls == 2


def test_the_label_separates_generator_from_judge(tmp_path):
    # Both roles share one cache file; identical prompts to different models
    # must not collide.
    path = tmp_path / "c.jsonl"
    gen, judge = CountingChat("gen"), CountingChat("judge")
    assert CachingChat(gen, path, label="gen").complete("s", "u").startswith("gen")
    assert CachingChat(judge, path, label="judge").complete("s", "u").startswith("judge")


def test_a_response_survives_a_new_process(tmp_path):
    # The whole point: an interrupted run must not have to pay for its calls
    # a second time.
    path = tmp_path / "c.jsonl"
    first = CountingChat()
    CachingChat(first, path, label="gen").complete("s", "u")

    second = CountingChat()
    assert CachingChat(second, path, label="gen").complete("s", "u") == "answer-1"
    assert second.calls == 0


def test_an_error_from_the_inner_provider_is_not_cached(tmp_path):
    # Caching a failure would make a transient outage permanent for that
    # passage across every later run.
    class Failing:
        def __init__(self):
            self.calls = 0

        def complete(self, system, user):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("connection refused")
            return "recovered"

    inner = Failing()
    chat = CachingChat(inner, tmp_path / "c.jsonl", label="gen")
    try:
        chat.complete("s", "u")
    except RuntimeError:
        pass
    assert chat.complete("s", "u") == "recovered"


def test_a_corrupt_cache_line_is_skipped_not_fatal(tmp_path):
    # A run killed mid-write leaves a truncated final line; that must not
    # make every later run unstartable.
    path = tmp_path / "c.jsonl"
    path.write_text('{"key": "abc", "response": "ok"}\n{"key": "trunc', encoding="utf-8")
    inner = CountingChat()
    chat = CachingChat(inner, path, label="gen")
    assert chat.complete("s", "u") == "answer-1"


def test_entries_are_appended_as_one_json_line_each(tmp_path):
    path = tmp_path / "c.jsonl"
    chat = CachingChat(CountingChat(), path, label="gen")
    chat.complete("s", "u1")
    chat.complete("s", "u2")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {"key", "response"} <= set(lines[0])


def test_hits_and_misses_are_counted_for_reporting(tmp_path):
    chat = CachingChat(CountingChat(), tmp_path / "c.jsonl", label="gen")
    chat.complete("s", "u")
    chat.complete("s", "u")
    assert (chat.hits, chat.misses) == (1, 1)
