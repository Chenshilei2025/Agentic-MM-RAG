"""JSON helpers for large-but-local processed stores."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=32)
def load_json_cached(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json(path: Path, *, cached: bool = True) -> Any:
    if cached:
        return load_json_cached(str(path.resolve()))
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_count_pattern(path: Path, pattern: str) -> int:
    """Count text occurrences without loading the whole file as JSON."""

    needle = pattern.encode("utf-8")
    overlap = max(0, len(needle) - 1)
    count = 0
    tail = b""
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(needle)
            tail = data[-overlap:] if overlap else b""
    return count
