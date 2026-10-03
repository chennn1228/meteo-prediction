"""Canonical content hashes used as dependency keys."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not valid hash inputs")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Mapping):
        normalized_keys: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = key
            elif isinstance(key, int) and not isinstance(key, bool):
                normalized_key = str(key)
            else:
                raise TypeError("hash mapping keys must be strings or integers")
            if normalized_key in normalized_keys:
                raise TypeError(f"hash mapping keys collide after normalization: {normalized_key}")
            normalized_keys[normalized_key] = item
        return {key: _canonical_value(normalized_keys[key]) for key in sorted(normalized_keys)}
    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"unsupported canonical hash value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def content_hash(value: Any, *, length: int = 12) -> str:
    if not 8 <= length <= 64:
        raise ValueError("content hash length must be between 8 and 64")
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
