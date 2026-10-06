"""Config-driven prediction schema, generation, validation, and persistence."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


COORDINATE_COLUMNS = (
    "requested_coordinates", "gfs_service_coordinates",
    "truth_service_coordinates")
PREDICTION_TYPES = frozenset({"point", "quantile"})
RESULT_STATUSES = frozenset({"diagnostic", "provisional", "official"})


def quantiles(protocol_config: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(value)
                 for value in protocol_config["probability"]["quantiles"])


def quantile_columns(protocol_config: Mapping[str, Any]) -> tuple[str, ...]:
    columns = tuple(protocol_config["evaluation"]["prediction_contract"]
                    ["quantile_columns"])
    expected = tuple(f"q{value:.2f}" for value in quantiles(protocol_config))
    if columns != expected:
        raise ValueError("prediction columns disagree with registered quantiles")
    return columns


def required_columns(protocol_config: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        "model_id", "implementation_level", "execution_level",
        "prediction_type", "location_id", *COORDINATE_COLUMNS,
        "target_time_utc", "forecast_issue_time_utc", "lead_time",
        "outer_fold", "inner_fold", "seed", "y", "point_prediction",
        *quantile_columns(protocol_config), "data_version", "feature_version",
        "protocol_version", "experiment_id", "result_status")


def _coordinates(value: object, column: str) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{column} must be JSON [latitude, longitude]") from exc
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{column} must be [latitude, longitude]")
    try:
        latitude, longitude = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} has nonnumeric coordinates") from exc
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"{column} coordinates outside geographic bounds")
    return json.dumps([latitude, longitude], separators=(",", ":"))


def _utc(series: pd.Series, column: str) -> pd.Series:
    if not isinstance(series.dtype, pd.DatetimeTZDtype):
        invalid = series.astype(str).map(
            lambda value: re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", value) is None)
        if invalid.any():
            raise ValueError(f"{column} requires explicit UTC/offset timestamps")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{column} contains invalid timestamps")
    return parsed


def validate_predictions(frame: pd.DataFrame,
                         protocol_config: Mapping[str, Any], *,
                         require_truth: bool = True) -> pd.DataFrame:
    """Validate the canonical row contract without repairing crossings."""
    required = required_columns(protocol_config)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"prediction contract missing columns: {missing}")
    if frame.empty:
        raise ValueError("prediction table is empty")
    output = frame.copy()
    gate = protocol_config["execution_gate"]
    for column, choices in (
        ("implementation_level", frozenset(gate["implementation_levels"])),
        ("execution_level", frozenset(gate["execution_levels"])),
        ("prediction_type", PREDICTION_TYPES),
        ("result_status", RESULT_STATUSES)):
        invalid = ~output[column].isin(choices)
        if invalid.any():
            raise ValueError(
                f"invalid {column}: {output.loc[invalid, column].unique().tolist()}")
    official = ((output.execution_level == "official")
                | (output.result_status == "official"))
    if (official & ((output.implementation_level != "validated")
                    | (output.execution_level != "official"))).any():
        raise ValueError(
            "official results require validated implementation and official execution")
    contract = protocol_config["evaluation"]["prediction_contract"]
    if ((output.model_id == contract["point_only_model"])
            & (output.prediction_type != "point")).any():
        raise ValueError(
            f"{contract['point_only_model']} is a point forecast, not a probability baseline")
    if (official & (output.model_id == "pinn")).any():
        raise ValueError(
            "PINN remains experimental pending physical-unit constraint validation")
    if (official & output.model_id.astype(str).str.fullmatch(
            r"v\d+", case=False)).any():
        raise ValueError("official model_id must be semantic; vN is provenance only")
    for column in (
            "model_id", "location_id", "outer_fold", "data_version",
            "feature_version", "protocol_version", "experiment_id"):
        if (output[column].isna().any()
                or (output[column].astype(str).str.strip() == "").any()):
            raise ValueError(f"{column} contains blanks")
    for column in COORDINATE_COLUMNS:
        output[column] = output[column].map(
            lambda value: _coordinates(value, column))
    for column in ("target_time_utc", "forecast_issue_time_utc"):
        output[column] = _utc(output[column], column)
    if (output.forecast_issue_time_utc >= output.target_time_utc).any():
        raise ValueError("forecast issue time must precede target time")
    output["lead_time"] = pd.to_numeric(output.lead_time, errors="coerce")
    if (~np.isfinite(output.lead_time) | (output.lead_time <= 0)).any():
        raise ValueError("lead_time must be positive finite hours")
    output["seed"] = pd.to_numeric(output.seed, errors="coerce")
    if (~np.isfinite(output.seed) | (output.seed < 0)
            | (output.seed % 1 != 0)).any():
        raise ValueError("seed must be a nonnegative integer")
    output["y"] = pd.to_numeric(output.y, errors="coerce")
    if require_truth and (~np.isfinite(output.y)).any():
        raise ValueError("observed y must be finite")
    is_quantile = output.prediction_type == "quantile"
    is_point = ~is_quantile
    output["point_prediction"] = pd.to_numeric(
        output.point_prediction, errors="coerce")
    if (is_point & ~np.isfinite(output.point_prediction)).any():
        raise ValueError("point forecasts require finite point_prediction")
    columns = quantile_columns(protocol_config)
    median_column = columns[quantiles(protocol_config).index(0.5)]
    if (is_quantile & output.point_prediction.notna()
            & ~np.isclose(output.point_prediction,
                          pd.to_numeric(output[median_column], errors="coerce"),
                          equal_nan=False)).any():
        raise ValueError(
            f"quantile point_prediction, if set, must equal {median_column}")
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
        if (is_quantile & ~np.isfinite(output[column])).any():
            raise ValueError(f"quantile forecast missing finite {column}")
        if (is_point & output[column].notna()).any():
            raise ValueError(f"point-only rows must not fabricate {column}")
    return output


def predict_frame(model: Any, features: pd.DataFrame, *, model_id: str,
                  protocol_config: Mapping[str, Any]) -> pd.DataFrame:
    contract = protocol_config["evaluation"]["prediction_contract"]
    levels = quantiles(protocol_config)
    point = np.asarray(model.predict(features), dtype=float)
    if point.shape != (len(features),):
        raise ValueError("prediction length does not match feature rows")
    result = features.loc[:, [column for column in contract["temporal_fields"]
                              if column in features]].copy()
    result["model_id"] = model_id
    result["point_prediction"] = point
    if model_id != contract["point_only_model"]:
        values = np.asarray(model.predict_quantiles(features, levels), dtype=float)
        if values.shape != (len(features), len(levels)):
            raise ValueError(
                "quantile prediction shape does not match registered quantiles")
        for index, name in enumerate(quantile_columns(protocol_config)):
            result[name] = values[:, index]
    return result


def write_predictions(frame: pd.DataFrame, path: str | Path,
                      protocol_config: Mapping[str, Any], *,
                      require_truth: bool = True) -> Path:
    destination = Path(path)
    clean = validate_predictions(
        frame, protocol_config, require_truth=require_truth)
    required = required_columns(protocol_config)
    clean = clean.loc[:, [*required,
                          *(column for column in clean if column not in required)]]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".csv":
        clean.to_csv(destination, index=False)
    elif destination.suffix.lower() == ".parquet":
        clean.to_parquet(destination, index=False)
    else:
        raise ValueError("prediction file must be .csv or .parquet")
    return destination


def read_predictions(path: str | Path, protocol_config: Mapping[str, Any], *,
                     require_truth: bool = True) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    elif source.suffix.lower() == ".parquet":
        frame = pd.read_parquet(source)
    else:
        raise ValueError("prediction file must be .csv or .parquet")
    return validate_predictions(
        frame, protocol_config, require_truth=require_truth)
