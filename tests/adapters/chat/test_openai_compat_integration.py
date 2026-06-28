import os

import pytest

from ariostea.adapters.chat.openai_compat import OpenAICompatChat

BASE_URL = os.environ.get("ARIOSTEA_TEST_CHAT_BASE_URL")
MODEL = os.environ.get("ARIOSTEA_TEST_CHAT_MODEL", "llama3.1")


@pytest.mark.integration
@pytest.mark.skipif(not BASE_URL, reason="set ARIOSTEA_TEST_CHAT_BASE_URL to run")
def test_real_endpoint_returns_a_blurb():
    chat = OpenAICompatChat(
        base_url=BASE_URL, model=MODEL, api_key=os.environ.get("ARIOSTEA_TEST_CHAT_API_KEY", "")
    )
    out = chat.complete(system="Reply with one short sentence.", user="Say hello.")
    assert isinstance(out, str) and out.strip()
