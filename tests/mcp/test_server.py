import pytest

from ariostea.mcp.server import build_server, resolve_transport


def test_resolve_transport_maps_cli_names():
    assert resolve_transport("stdio") == "stdio"
    assert resolve_transport("http") == "streamable-http"


def test_resolve_transport_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_transport("grpc")


def test_build_server_applies_host_and_port():
    mcp = build_server(object(), host="0.0.0.0", port=9001)
    assert mcp.settings.host == "0.0.0.0"
    assert mcp.settings.port == 9001


def test_build_server_defaults_to_localhost_8000():
    mcp = build_server(object())
    assert mcp.settings.host == "127.0.0.1"
    assert mcp.settings.port == 8000


def test_http_app_mounts_mcp_route_and_serves_tools():
    import asyncio

    mcp = build_server(object())
    app = mcp.streamable_http_app()
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)

    tool_names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "search_knowledge" in tool_names
    assert "get_note" in tool_names
