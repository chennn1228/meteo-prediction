"""Issue-time-safe forecast features without station identity or future truth."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s01_core.config_loader import ProtocolError, load_manifest
from s01_core.schemas import assert_model_features


FORECAST_ONLY_INPUTS = frozenset({
    "ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "terrestrial_fcst",
    "cloud_cover_fcst", "temp_fcst", "rh_fcst", "dewpoint_fcst",
    "wind_speed_fcst", "wind_dir_fcst", "pressure_fcst", "precip_fcst",
    "sunshine_fcst", "ghi_clear_sky", "dni_clear_sky", "solar_elevation",
    "solar_azimuth", "source_grid_latitude", "source_grid_longitude",
    "source_grid_elevation", "lead_time", "forecast_issue_time_utc",
    "target_time_utc",
})


def formal_daylight_mask(frame: pd.DataFrame) -> pd.Series:
    """Single formal GHI daytime convention; alternatives are sensitivity only."""
    if load_manifest()["daylight_definition"]["formal"] != "solar_elevation_gt_0":
        raise ProtocolError("unregistered formal daylight definition")
    if "solar_elevation" not in frame:
        raise ProtocolError("solar elevation is required for formal daytime")
    return pd.to_numeric(frame["solar_elevation"], errors="coerce") > 0


def assert_forecast_issue_semantics(frame: pd.DataFrame) -> None:
    required = {"forecast_issue_time_utc", "target_time_utc", "lead_time"}
    missing = required - set(frame)
    if missing:
        raise ProtocolError(f"missing time semantics: {sorted(missing)}")
    issue = pd.to_datetime(frame["forecast_issue_time_utc"], utc=True)
    target = pd.to_datetime(frame["target_time_utc"], utc=True)
    lead = pd.to_numeric(frame["lead_time"], errors="coerce")
    registered = set(load_manifest()["data_sources"]["forecast"]["leads_hours"])
    if not set(lead.dropna().unique()).issubset(registered) or lead.isna().any():
        raise ProtocolError("forecast lead is outside registered D+1/D+2/D+3 hours")
    if issue.isna().any() or target.isna().any() or not (
        issue + pd.to_timedelta(lead, unit="h") == target
    ).all():
        raise ProtocolError("forecast issue + lead must equal target time")


def _ratio(numerator: pd.Series, denominator: pd.Series, threshold: float) -> np.ndarray:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > threshold)


def build_forecast_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive only known-at-issue predictors; observations are not consulted.

    The supplied clear-sky/solar columns must be computed at the *returned GFS
    service coordinates*, not at the requested probe location. Identity fields
    remain in the returned frame solely for joining and provenance.
    """
    assert_forecast_issue_semantics(frame)
    required = {"location_id", "ghi_fcst", "dhi_fcst", "dni_fcst", "ghi_clear_sky",
                "dni_clear_sky", "wind_dir_fcst", "solar_azimuth", "cloud_cover_fcst"}
    missing = required - set(frame)
    if missing:
        raise ProtocolError(f"missing forecast feature sources: {sorted(missing)}")
    data = frame.copy()
    config = load_manifest()
    threshold = float(config["feature_policy"]["clear_sky_denominator_min_wm2"])
    diffuse_threshold = float(config["feature_policy"]["diffuse_denominator_min_wm2"])
    data["kt_fcst"] = np.clip(_ratio(data["ghi_fcst"], data["ghi_clear_sky"], threshold), 0, 1.5)
    data["kni"] = np.clip(_ratio(data["dni_fcst"], data["dni_clear_sky"], threshold), 0, 1.5)
    data["diffuse_fraction"] = np.clip(
        _ratio(data["dhi_fcst"], data["ghi_fcst"], diffuse_threshold), 0, 1
    )
    data["ghi_fcst_minus_clear"] = data["ghi_fcst"] - data["ghi_clear_sky"]
    for source, prefix in (("wind_dir_fcst", "wind_dir"), ("solar_azimuth", "solar_azimuth")):
        radians = np.deg2rad(pd.to_numeric(data[source], errors="coerce"))
        data[f"{prefix}_sin"] = np.sin(radians)
        data[f"{prefix}_cos"] = np.cos(radians)
    target = pd.to_datetime(data["target_time_utc"], utc=True)
    local = target.dt.tz_convert("Asia/Shanghai")
    hour = local.dt.hour
    month = local.dt.month
    doy = local.dt.dayofyear
    for label, values, period in (("hour_local", hour, 24), ("month", month - 1, 12),
                                  ("doy", doy - 1, 365.25)):
        data[f"{label}_sin"] = np.sin(2 * np.pi * values / period)
        data[f"{label}_cos"] = np.cos(2 * np.pi * values / period)
    data = data.sort_values(["location_id", "lead_time", "forecast_issue_time_utc"]).copy()
    grouped = data.groupby(["location_id", "lead_time"], sort=False)
    for lag in (1, 2):
        data[f"ghi_fcst_lag{lag}"] = grouped["ghi_fcst"].shift(lag)
    data["cloud_cover_change"] = grouped["cloud_cover_fcst"].diff()
    for window in (2, 4):
        suffix = "roll1" if window == 2 else "roll3"
        data[f"cloud_cover_{suffix}"] = grouped["cloud_cover_fcst"].transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
    return data.sort_index()


def formal_feature_columns(frame: pd.DataFrame) -> list[str]:
    config = load_manifest()
    names: list[str] = []
    for group, values in config["feature_groups"].items():
        if group in {"selection_evidence", "shap_role"}:
            continue
        names.extend(values)
    names = list(dict.fromkeys(names))
    assert_model_features(names)
    missing = set(names) - set(frame)
    if missing:
        raise ProtocolError(f"formal features missing from frame: {sorted(missing)}")
    return names
