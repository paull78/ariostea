"""Whitespace/case text normalization shared by span metrics and gold validation."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    """Lowercase and collapse all runs of whitespace to single spaces."""
    return _WS.sub(" ", text).strip().lower()
