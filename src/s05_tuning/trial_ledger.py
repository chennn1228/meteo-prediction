"""Auditable per-candidate/per-fold trial records."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Trial:
    model: str
    candidate: int
    parameters: dict
    outer: str
    inner: str
    seed: int
    fit_rows: int
    early_stop_rows: int
    scoring_rows: int
    mean_pinball: float | None
    wall_seconds: float
    device: str
    peak_memory_bytes: int | None
    epoch: int | None
    status: str
    error_reason: str | None


def as_frame(trials: list[Trial]) -> pd.DataFrame:
    records = []
    for trial in trials:
        record = asdict(trial)
        record["parameters"] = json.dumps(record["parameters"], sort_keys=True)
        records.append(record)
    return pd.DataFrame.from_records(records)


def write_ledger(trials: list[Trial], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    as_frame(trials).to_csv(destination, index=False)
