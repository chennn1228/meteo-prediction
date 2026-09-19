"""Validated prediction input."""

from pathlib import Path

import pandas as pd

from .schema import validate_predictions


def read_predictions(path: str | Path, *, require_truth: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("prediction file must be .csv or .parquet")
    return validate_predictions(frame, require_truth=require_truth)
