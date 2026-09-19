"""Machine-readable model registry derived from project_manifest.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())


def load_manifest():
    return yaml.safe_load((ROOT / "project_manifest.yaml").read_text(encoding="utf-8"))


def models():
    return load_manifest()["models"]


def deep_model_vnum():
    out = {}
    for record in models():
        version = record.get("internal_version")
        if version:
            out[record["id"]] = int(str(version).removeprefix("v"))
    return out


def validate_counts():
    manifest = load_manifest()
    records = manifest["models"]
    deep = [m for m in records if m.get("internal_version")]
    if manifest["model_count"] != len(records):
        raise ValueError("model_count does not match models[]")
    if manifest["deep_model_count"] != len(deep):
        raise ValueError("deep_model_count does not match internal-version models")
    ids = [m["id"] for m in records]
    if len(ids) != len(set(ids)):
        raise ValueError("model ids must be unique")
    return True
