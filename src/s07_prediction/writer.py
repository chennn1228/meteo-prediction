"""Validated prediction output; no implicit official result creation."""

from pathlib import Path

import pandas as pd

from .schema import REQUIRED_COLUMNS, validate_predictions


def write_predictions(frame: pd.DataFrame, path: str | Path,
                      *, require_truth: bool = True) -> Path:
    path = Path(path)
    clean = validate_predictions(frame, require_truth=require_truth)
    clean = clean.loc[:, [*REQUIRED_COLUMNS, *(c for c in clean if c not in REQUIRED_COLUMNS)]]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        clean.to_csv(path, index=False)
    elif path.suffix.lower() == ".parquet":
        clean.to_parquet(path, index=False)
    else:
        raise ValueError("prediction file must be .csv or .parquet")
    return path
