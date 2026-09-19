"""Data-level gates invoked with explicit artifacts, never implicit old caches."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from s01_core.config_loader import load_data_config, load_manifest
from s02_data.clean_contract import audit_clean_frame, validate_cached_month
from s07_prediction.schema import validate_predictions


def validate_monthly_raw(path: Path, *, source: str, start: dt.date, end: dt.date,
                         requested_lat: float, requested_lon: float) -> dict:
    manifest = load_manifest()
    variables = load_data_config("02_variables.yaml")
    return validate_cached_month(path, source=source, cfg=variables, start=start, end=end,
                                 data_version=manifest["data_version"],
                                 requested_lat=requested_lat, requested_lon=requested_lon)


def validate_cleaned_site(frame: pd.DataFrame, *, start: dt.date, end: dt.date,
                          required_columns: tuple[str, ...]) -> dict:
    return audit_clean_frame(frame, start, end, required_columns)


def validate_result_table(frame: pd.DataFrame, *, official: bool = False) -> pd.DataFrame:
    valid = validate_predictions(frame)
    if official and not ((valid["implementation_level"] == "validated")
                         & (valid["execution_level"] == "official")
                         & (valid["result_status"] == "official")).all():
        raise ValueError("formal results must be validated + official + official")
    return valid
