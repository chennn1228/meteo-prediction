"""Tiny deterministic forecast rows for tests; never train or publish models."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import build_forecast_features  # noqa: E402
from s03_features.preprocessing import FoldPreprocessor  # noqa: E402
from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402


def synthetic_feature_and_prediction_rows():
    """Return features, frozen preprocessor, calibration and later test rows."""
    targets = pd.to_datetime([
        "2025-01-02T00:00:00Z", "2025-01-03T00:00:00Z",
        "2025-01-11T00:00:00Z", "2025-01-12T00:00:00Z",
        "2025-01-13T00:00:00Z", "2025-01-14T00:00:00Z",
        "2025-01-21T00:00:00Z", "2025-01-22T00:00:00Z",
    ])
    input_frame = pd.DataFrame({
        "location_id": ["om-gfs-land-32.100000-119.100000"] * len(targets),
        "target_time_utc": targets,
        "forecast_issue_time_utc": targets - pd.Timedelta(hours=24),
        "lead_time": [24] * len(targets),
        "ghi_fcst": [100., 120., 130., 135., 140., 145., 150., 160.],
        "dhi_fcst": [30.] * len(targets),
        "dni_fcst": [200.] * len(targets),
        "ghi_clear_sky": [300.] * len(targets),
        "dni_clear_sky": [450.] * len(targets),
        "wind_dir_fcst": [10., 20., 30., 40., 50., 60., 70., 80.],
        "solar_azimuth": [180.] * len(targets),
        "solar_elevation": [30.] * len(targets),
        "cloud_cover_fcst": [20., 30., 40., 50., 60., 70., 80., 90.],
    })
    features = build_forecast_features(input_frame)
    columns = ("ghi_fcst", "kt_model", "wind_dir_sin")
    processor = FoldPreprocessor(columns).fit(features.iloc[:2],
                                              fit_end_utc="2025-01-02T00:00:00Z")
    manifest = load_manifest()
    rows = []
    for position in range(2, len(features)):
        source = features.iloc[position]
        center = float(source.ghi_fcst)
        record = {
            "model_id": "synthetic_quantile", "implementation_level": "prototype",
            "execution_level": "smoke", "prediction_type": "quantile",
            "location_id": source.location_id,
            "requested_coordinates": [32.0, 119.0],
            "gfs_service_coordinates": [32.1, 119.1],
            "truth_service_coordinates": [32.2, 119.2],
            "target_time_utc": source.target_time_utc,
            "forecast_issue_time_utc": source.forecast_issue_time_utc,
            "lead_time": 24, "outer_fold": "outer_synthetic",
            "inner_fold": "inner_synthetic", "seed": 0,
            "y": center + 5., "point_prediction": center,
            "data_version": manifest["data_version"],
            "feature_version": manifest["feature_version"],
            "protocol_revision": manifest["protocol_version"],
            "experiment_id": "synthetic-smoke-only", "result_status": "diagnostic",
        }
        for column, offset in zip(QUANTILE_COLUMNS, (-30., -20., -10., 0., 10., 20., 30.)):
            record[column] = center + offset
        rows.append(record)
    predictions = pd.DataFrame(rows)
    return features, processor, predictions.iloc[:4].copy(), predictions.iloc[4:].copy()


def raw_point_companion(quantile_rows: pd.DataFrame) -> pd.DataFrame:
    raw = quantile_rows.copy()
    raw["model_id"] = "raw_gfs"
    raw["prediction_type"] = "point"
    for column in QUANTILE_COLUMNS:
        raw[column] = np.nan
    return raw
