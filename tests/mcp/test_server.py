from types import SimpleNamespace

import pytest

from ariostea.mcp.server import build_server, resolve_transport

# Enough of a container to build the server; no tool is called.
NO_CONTAINER = SimpleNamespace(runs=None)


def test_resolve_transport_maps_cli_names():
    assert resolve_transport("stdio") == "stdio"
    assert resolve_transport("http") == "streamable-http"


def test_resolve_transport_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_transport("grpc")


def test_build_server_applies_host_and_port():
    mcp = build_server(NO_CONTAINER, host="0.0.0.0", port=9001)
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 9001


def test_build_server_defaults_to_localhost_8000():
    mcp = build_server(NO_CONTAINER)
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8000


def test_http_app_mounts_mcp_route_and_serves_tools():
    import asyncio

    mcp = build_server(NO_CONTAINER)
    app = mcp.streamable_http_app()
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)

    tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "search_knowledge" in tool_names
    assert "get_note" in tool_names


def test_reindex_tool_returns_started():
    import asyncio
    import json
    import threading
    from types import SimpleNamespace

    from fixtures.runs import index_runs

    from ariostea.domain.models import IndexReport

    finished = threading.Event()

    class Indexer:
        fingerprint = "fp"

        def index(self, progress=None):
            finished.set()
            return IndexReport(0, 0, (), (), ())

    mcp = build_server(SimpleNamespace(runs=index_runs(Indexer())))
    content = asyncio.run(mcp.call_tool("reindex", {}))
    blocks = content[0] if isinstance(content, tuple) else content
    assert json.loads(blocks[0].text) == {"started": True}
    assert finished.wait(10)
