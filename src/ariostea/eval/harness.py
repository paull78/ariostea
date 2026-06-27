from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GoldCase:
    query: str
    query_lang: str
    expected: tuple[str, ...]
    direction: str  # "en→it" | "it→en" | "same"


def load_gold(path: str | Path) -> list[GoldCase]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldCase(
            query=row["query"],
            query_lang=row["query_lang"],
            expected=tuple(row["expected"]),
            direction=row["direction"],
        )
        for row in rows
    ]
