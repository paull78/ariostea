from __future__ import annotations

import httpx

from ariostea.ports.chat import ChatProvider


class ChatError(RuntimeError):
    """A chat completion request failed (bad status or transport error)."""


class OpenAICompatChat(ChatProvider):
    """Chat via any OpenAI-compatible /chat/completions endpoint (OpenAI,
    Ollama, LM Studio, vLLM, llama.cpp, …). The client is injectable for tests
    (when injected, ``timeout`` is ignored — the client owns it). Every failure
    mode surfaces as ``ChatError`` so callers have a single exception to catch."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 30.0,
        max_tokens: int = 128,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._client = client or httpx.Client(timeout=timeout)

    def complete(self, system: str, user: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._max_tokens,
            "temperature": 0,
        }
        try:
            resp = self._client.post(
                f"{self._base_url}/chat/completions", json=payload, headers=headers
            )
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise ChatError(f"chat request failed: {exc}") from exc
        if resp.status_code >= 400:
            raise ChatError(f"chat completion failed: {resp.status_code} {resp.text}")
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:  # ValueError covers JSONDecodeError
            raise ChatError(f"unexpected chat response shape: {resp.text[:200]}") from exc
