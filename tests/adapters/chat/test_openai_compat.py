import json

import httpx
import pytest

from ariostea.adapters.chat.openai_compat import ChatError, OpenAICompatChat


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_builds_request_and_parses_response():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "a blurb"}}]})

    chat = OpenAICompatChat(base_url="http://x/v1", model="m", api_key="k", client=_client(handler))
    out = chat.complete(system="sys", user="usr")

    assert out == "a blurb"
    assert captured["url"] == "http://x/v1/chat/completions"
    assert captured["auth"] == "Bearer k"
    assert captured["body"]["model"] == "m"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_omits_auth_header_without_key():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "x"}}]})

    OpenAICompatChat(base_url="http://x/v1", model="m", client=_client(handler)).complete("s", "u")
    assert seen["auth"] is None


def test_raises_on_error_status():
    chat = OpenAICompatChat(
        base_url="http://x/v1",
        model="m",
        client=_client(lambda r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(ChatError):
        chat.complete(system="s", user="u")


def test_raises_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    chat = OpenAICompatChat(base_url="http://x/v1", model="m", client=_client(handler))
    with pytest.raises(ChatError):
        chat.complete(system="s", user="u")


def test_raises_on_malformed_response():
    chat = OpenAICompatChat(
        base_url="http://x/v1",
        model="m",
        client=_client(lambda r: httpx.Response(200, json={"choices": []})),
    )
    with pytest.raises(ChatError):
        chat.complete(system="s", user="u")
