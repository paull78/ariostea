"""Where the fastembed adapters keep their downloaded models."""

from __future__ import annotations

import os
from pathlib import Path


def fastembed_cache_dir() -> str:
    """`FASTEMBED_CACHE_PATH` if set, else `~/.cache/fastembed`.

    fastembed's own default is `tempfile.gettempdir()/fastembed_cache`, which on
    macOS is a directory the OS purges. The purge deletes the downloaded blobs
    but leaves the snapshot links, so the next load either fails on a missing
    `model.onnx` or re-fetches under a stale lock and hangs. Read at call time,
    so a changed environment takes effect.
    """
    return os.environ.get("FASTEMBED_CACHE_PATH") or str(Path.home() / ".cache" / "fastembed")
