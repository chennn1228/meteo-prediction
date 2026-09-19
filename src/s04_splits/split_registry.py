"""Read the single project protocol, without copying dates into model code."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from s01_core.config_loader import load_manifest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())


def manifest() -> dict:
    return load_manifest(ROOT / "project_manifest.yaml")


def utc(value: str, *, inclusive_date_end: bool = False) -> pd.Timestamp:
    """Turn a manifest date into a half-open UTC boundary."""
    time = pd.Timestamp(value)
    if time.tzinfo is None:
        time = time.tz_localize("UTC")
    else:
        time = time.tz_convert("UTC")
    if inclusive_date_end and len(str(value)) == 10:
        time += pd.Timedelta(days=1)
    return time


def protocol() -> dict:
    return manifest()["validation_protocol"]


def quantiles() -> tuple[float, ...]:
    return tuple(float(q) for q in manifest()["quantiles"])


def purge_days() -> int:
    p = protocol()
    hours = int(p["purge_hours"])
    if hours != int(p["maximum_sequence_lookback_hours"]) + int(p["maximum_lead_hours"]):
        raise ValueError("purge_hours disagrees with sequence lookback plus maximum lead")
    if hours % 24:
        raise ValueError("day-based rolling windows require an integral-day purge")
    return hours // 24
