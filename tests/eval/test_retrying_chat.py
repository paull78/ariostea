import logging

import pytest

from ariostea.adapters.chat.openai_compat import ChatError
from ariostea.eval.retrying_chat import RetryingChat


class ScriptedChat:
    """Returns/raises each entry of `script` in order, one per call."""

    def __init__(self, script: list[str | Exception]) -> None:
        self._script = list(script)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_success_after_one_failure_is_returned_without_a_final_retry():
    inner = ScriptedChat([ChatError("down"), "ok"])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, attempts=3, backoff_s=30, sleep=sleep)
    assert chat.complete("s", "u") == "ok"
    assert inner.calls == 2
    assert sleep.calls == [30]


def test_a_blank_reply_is_retried_then_succeeds():
    inner = ScriptedChat(["   ", "ok"])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, attempts=3, backoff_s=30, sleep=sleep)
    assert chat.complete("s", "u") == "ok"
    assert inner.calls == 2


def test_all_attempts_failing_raises_chat_error():
    inner = ScriptedChat([ChatError("a"), ChatError("b"), ChatError("c")])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, attempts=3, backoff_s=30, sleep=sleep)
    with pytest.raises(ChatError):
        chat.complete("s", "u")
    assert inner.calls == 3


def test_no_sleep_happens_after_the_final_attempt():
    inner = ScriptedChat([ChatError("a"), ChatError("b"), ChatError("c")])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, attempts=3, backoff_s=30, sleep=sleep)
    with pytest.raises(ChatError):
        chat.complete("s", "u")
    assert sleep.calls == [30, 60]


def test_backoff_grows_with_the_attempt_number():
    inner = ScriptedChat([ChatError("a"), ChatError("b"), "ok"])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, attempts=3, backoff_s=10, sleep=sleep)
    assert chat.complete("s", "u") == "ok"
    assert sleep.calls == [10, 20]


def test_a_blank_reply_on_the_final_attempt_raises_chat_error_not_a_blank():
    inner = ScriptedChat(["", " ", "\n"])
    chat = RetryingChat(inner, attempts=3, backoff_s=0, sleep=RecordingSleep())
    with pytest.raises(ChatError):
        chat.complete("s", "u")


def test_defaults_are_three_attempts_and_thirty_second_backoff():
    inner = ScriptedChat([ChatError("a"), ChatError("b"), ChatError("c")])
    sleep = RecordingSleep()
    chat = RetryingChat(inner, sleep=sleep)
    with pytest.raises(ChatError):
        chat.complete("s", "u")
    assert inner.calls == 3
    assert sleep.calls == [30, 60]


def test_each_retry_logs_a_warning(caplog):
    inner = ScriptedChat([ChatError("down"), "ok"])
    chat = RetryingChat(inner, attempts=3, backoff_s=0, sleep=RecordingSleep())
    with caplog.at_level(logging.WARNING, logger="ariostea.eval.retrying_chat"):
        chat.complete("s", "u")
    assert any("retry" in record.message.lower() for record in caplog.records)
