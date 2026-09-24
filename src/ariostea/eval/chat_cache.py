"""An on-disk response cache in front of a `ChatProvider`.

Two problems, one fix.

The first is crash recovery. A full gold run is roughly an hour of model
calls, and the first attempt was killed during the indexing step that follows
generation -- after every expensive call had been paid for and before
anything was written. `build_wiki_corpus.py` learned the same lesson and
writes its manifest in a `finally`; this is the equivalent for work that costs
model time rather than network time.

The second is iteration. The validation gates are thresholds and prompts that
will need tuning, and without a cache every adjustment costs another hour of
inference. With one, re-running against already-generated responses is free,
so tuning a gate is a matter of seconds -- which is the difference between a
gate that gets tuned and one that gets left alone.

Only successes are cached. A failed call is a transient endpoint problem, and
caching it would make one outage permanent for that passage in every later run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ariostea.ports.chat import ChatProvider


class CachingChat(ChatProvider):
    """Wrap a `ChatProvider`, persisting each response to a JSONL file.

    `label` distinguishes roles that share one cache file: the generator and
    the judge are different models, and an identical prompt sent to both must
    not collide on one entry.
    """

    def __init__(self, inner: ChatProvider, path: Path, label: str) -> None:
        self._inner = inner
        self._path = Path(path)
        self._label = label
        self._entries = self._load()
        self.hits = 0
        self.misses = 0

    def _load(self) -> dict[str, str]:
        """Read the cache, skipping any line that will not parse.

        A run killed mid-write leaves a truncated final line. Refusing to
        start because of it would throw away the whole point of having a
        cache, so a bad line is dropped and its call simply repeats.
        """
        if not self._path.exists():
            return {}
        entries: dict[str, str] = {}
        for line in self._path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                entries[row["key"]] = row["response"]
            except (ValueError, KeyError, TypeError):
                continue
        return entries

    def _key(self, system: str, user: str) -> str:
        digest = hashlib.sha256(f"{self._label}\x00{system}\x00{user}".encode()).hexdigest()
        return digest

    def complete(self, system: str, user: str) -> str:
        key = self._key(system, user)
        if key in self._entries:
            self.hits += 1
            return self._entries[key]
        response = self._inner.complete(system, user)  # errors propagate uncached
        self.misses += 1
        self._entries[key] = response
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
        return response
