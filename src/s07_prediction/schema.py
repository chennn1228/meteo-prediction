"""One prediction-row contract for models, calibration and evaluation.

Coordinates are JSON [latitude, longitude] values.  Identity and provenance
columns are metadata only; they must never be fed into a model feature matrix.
The seven quantile columns also exist on point-only rows, but must be null.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from s01_core.config_loader import load_manifest

_PROTOCOL = load_manifest()
QUANTILES = tuple(float(value) for value in _PROTOCOL["quantiles"])
QUANTILE_COLUMNS = tuple(f"q{tau:.2f}" for tau in QUANTILES)
COORDINATE_COLUMNS = (
    "requested_coordinates", "gfs_service_coordinates", "truth_service_coordinates"
)
REQUIRED_COLUMNS = (
    "model_id", "implementation_level", "execution_level", "prediction_type",
    "location_id", *COORDINATE_COLUMNS,
    "target_time_utc", "forecast_issue_time_utc", "lead_time",
    "outer_fold", "inner_fold", "seed", "y", "point_prediction",
    *QUANTILE_COLUMNS, "data_version", "feature_version",
    "protocol_revision", "experiment_id", "result_status",
)
IMPLEMENTATION_LEVELS = frozenset(_PROTOCOL["implementation_levels"])
EXECUTION_LEVELS = frozenset(_PROTOCOL["execution_levels"])
PREDICTION_TYPES = frozenset({"point", "quantile"})
RESULT_STATUSES = frozenset({"diagnostic", "provisional", "official"})


def _coordinates(value: object, column: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{column} must be JSON [latitude, longitude]") from exc
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{column} must be [latitude, longitude]")
    try:
        latitude, longitude = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} has nonnumeric coordinates") from exc
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"{column} coordinates outside geographic bounds")
    return json.dumps([latitude, longitude], separators=(",", ":"))


def _utc(series: pd.Series, column: str) -> pd.Series:
    # Explicit offset is required; silently localising naive timestamps to UTC
    # would conceal issue-time/target-time errors.
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        invalid = series.astype(str).map(
            lambda value: re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value) is None
        )
        if invalid.any():
            raise ValueError(f"{column} requires explicit UTC/offset timestamps")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{column} contains invalid timestamps")
    return parsed


def validate_predictions(frame: pd.DataFrame, *, require_truth: bool = True) -> pd.DataFrame:
    """Validate and return a normalised copy; never reorder quantiles silently.

    Quantile crossing is deliberately accepted and later measured before any
    calibration repair.  No missing/nonfinite targets or predictions are hidden.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"prediction contract missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction table is empty")
    out = frame.copy()
    for column, choices in (
        ("implementation_level", IMPLEMENTATION_LEVELS),
        ("execution_level", EXECUTION_LEVELS),
        ("prediction_type", PREDICTION_TYPES),
        ("result_status", RESULT_STATUSES),
    ):
        invalid = ~out[column].isin(choices)
        if invalid.any():
            raise ValueError(f"invalid {column}: {out.loc[invalid, column].unique().tolist()}")
    official = (out.execution_level == "official") | (out.result_status == "official")
    if (official & ((out.implementation_level != "validated") |
                   (out.execution_level != "official"))).any():
        raise ValueError("official results require validated implementation and official execution")
    if ((out.model_id == "raw_gfs") & (out.prediction_type != "point")).any():
        raise ValueError("raw_gfs is a point forecast, not a probability baseline")
    if (official & (out.model_id == "pinn")).any():
        raise ValueError("PINN remains experimental pending physical-unit constraint validation")
    if (official & out.model_id.astype(str).str.fullmatch(r"v\d+", case=False)).any():
        raise ValueError("official model_id must be semantic; vN is provenance only")
    for column in ("model_id", "location_id", "outer_fold", "data_version",
                   "feature_version", "protocol_revision", "experiment_id"):
        if out[column].isna().any() or (out[column].astype(str).str.strip() == "").any():
            raise ValueError(f"{column} contains blanks")
    for column in COORDINATE_COLUMNS:
        out[column] = out[column].map(lambda value: _coordinates(value, column))
    for column in ("target_time_utc", "forecast_issue_time_utc"):
        out[column] = _utc(out[column], column)
    if (out.forecast_issue_time_utc >= out.target_time_utc).any():
        raise ValueError("forecast issue time must precede target time")
    out["lead_time"] = pd.to_numeric(out.lead_time, errors="coerce")
    if (~np.isfinite(out.lead_time) | (out.lead_time <= 0)).any():
        raise ValueError("lead_time must be positive finite hours")
    out["seed"] = pd.to_numeric(out.seed, errors="coerce")
    if (~np.isfinite(out.seed) | (out.seed < 0) | (out.seed % 1 != 0)).any():
        raise ValueError("seed must be a nonnegative integer")
    out["y"] = pd.to_numeric(out.y, errors="coerce")
    if require_truth and (~np.isfinite(out.y)).any():
        raise ValueError("observed y must be finite")
    quantile = out.prediction_type == "quantile"
    point = ~quantile
    out["point_prediction"] = pd.to_numeric(out.point_prediction, errors="coerce")
    if (point & ~np.isfinite(out.point_prediction)).any():
        raise ValueError("point forecasts require finite point_prediction")
    if (quantile & out.point_prediction.notna() &
        ~np.isclose(out.point_prediction, pd.to_numeric(out["q0.50"], errors="coerce"),
                    equal_nan=False)).any():
        raise ValueError("quantile point_prediction, if set, must equal q0.50")
    for column in QUANTILE_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if (quantile & ~np.isfinite(out[column])).any():
            raise ValueError(f"quantile forecast missing finite {column}")
        if (point & out[column].notna()).any():
            raise ValueError(f"point-only rows must not fabricate {column}")
    return out
