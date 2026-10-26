"""The dependency rule, checked: the application (domain, ports, use cases)
imports no adapter, no wiring, no transport and no vendor library. Those
depend on the application, never the other way round."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parents[1] / "src" / "ariostea"
INSIDE = ("domain", "ports", "indexing", "search")
FORBIDDEN = (
    "ariostea.adapters",
    "ariostea.config",
    "ariostea.mcp",
    "watchfiles",
    "fastembed",
    "sqlite3",
    "sqlite_vec",
    "mcp",
    "httpx",
)


def _imports(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def _violations(source: str) -> list[str]:
    return [
        name
        for name in _imports(source)
        if any(name == bad or name.startswith(bad + ".") for bad in FORBIDDEN)
    ]


@pytest.mark.parametrize(
    "path",
    sorted(p for package in INSIDE for p in (SRC / package).rglob("*.py")),
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_dependency_rule(path):
    assert _violations(path.read_text(encoding="utf-8")) == []


def test_the_rule_catches_what_it_should():
    assert _violations("import watchfiles") == ["watchfiles"]
    assert _violations("from ariostea.adapters.vault.filesystem import X") == [
        "ariostea.adapters.vault.filesystem"
    ]
    assert _violations("if TYPE_CHECKING:\n    from ariostea.config import c") == [
        "ariostea.config"
    ]
    assert _violations("from ariostea.ports.vault import Vault\nimport mcpx") == []
