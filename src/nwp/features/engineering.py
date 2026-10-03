"""Issue-time-safe forecast feature engineering."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from nwp.core.schema import ContractError, assert_model_features


def formal_daylight_mask(frame: pd.DataFrame, protocol_config: Mapping[str, Any]) -> pd.Series:
    """Apply the one registered formal GHI daylight definition."""
    if protocol_config["daylight_definition"]["formal"] != "solar_elevation_gt_0":
        raise ContractError("unregistered formal daylight definition")
    if "solar_elevation" not in frame:
        raise ContractError("solar elevation is required for formal daytime")
    return pd.to_numeric(frame["solar_elevation"], errors="coerce") > 0


def assert_forecast_issue_semantics(
    frame: pd.DataFrame,
    data_config: Mapping[str, Any],
) -> None:
    required = {"forecast_issue_time_utc", "target_time_utc", "lead_time"}
    missing = required - set(frame)
    if missing:
        raise ContractError(f"missing time semantics: {sorted(missing)}")
    issue = pd.to_datetime(frame["forecast_issue_time_utc"], utc=True)
    target = pd.to_datetime(frame["target_time_utc"], utc=True)
    lead = pd.to_numeric(frame["lead_time"], errors="coerce")
    registered = {int(value) for value in data_config["forecast"]["leads"]}
    if lead.isna().any() or not set(int(value) for value in lead.unique()) <= registered:
        raise ContractError("forecast lead is outside the registered lead hours")
    if issue.isna().any() or target.isna().any() or not (
        issue + pd.to_timedelta(lead, unit="h") == target
    ).all():
        raise ContractError("forecast issue + lead must equal target time")


def _ratio(numerator: pd.Series, denominator: pd.Series, threshold: float) -> np.ndarray:
    num = pd.to_numeric(numerator, errors="coerce").to_numpy(dtype=float)
    den = pd.to_numeric(denominator, errors="coerce").to_numpy(dtype=float)
    return np.divide(num, den, out=np.full(len(num), np.nan), where=den > threshold)


def build_forecast_features(
    frame: pd.DataFrame,
    *,
    data_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Derive only predictors available at the forecast issue time."""
    assert_forecast_issue_semantics(frame, data_config)
    required = {
        "location_id", "ghi_fcst", "dhi_fcst", "dni_fcst", "ghi_clear_sky",
        "dni_clear_sky", "wind_dir_fcst", "solar_azimuth", "cloud_cover_fcst",
    }
    missing = required - set(frame)
    if missing:
        raise ContractError(f"missing forecast feature sources: {sorted(missing)}")
    data = frame.copy()
    policy = feature_config["policy"]
    derived = feature_config["derived_features"]
    clear_threshold = float(policy["clear_sky_denominator"]["minimum_wm2"])
    diffuse_threshold = float(policy["diffuse_denominator"]["minimum_wm2"])

    if derived.get("kt"):
        data["kt_raw"] = _ratio(data["ghi_fcst"], data["ghi_clear_sky"], clear_threshold)
        data["kt_model"] = data["kt_raw"]
    if derived.get("kni"):
        data["kni_raw"] = _ratio(data["dni_fcst"], data["dni_clear_sky"], clear_threshold)
        data["kni_model"] = data["kni_raw"]
    if derived.get("diffuse_fraction"):
        data["diffuse_fraction_raw"] = _ratio(
            data["dhi_fcst"], data["ghi_fcst"], diffuse_threshold
        )
        data["diffuse_fraction_model"] = data["diffuse_fraction_raw"]
    data["ghi_fcst_minus_clear"] = data["ghi_fcst"] - data["ghi_clear_sky"]

    for source, prefix in (
        ("wind_dir_fcst", "wind_dir"),
        ("solar_azimuth", "solar_azimuth"),
    ):
        radians = np.deg2rad(pd.to_numeric(data[source], errors="coerce"))
        data[f"{prefix}_sin"] = np.sin(radians)
        data[f"{prefix}_cos"] = np.cos(radians)
    target = pd.to_datetime(data["target_time_utc"], utc=True)
    local = target.dt.tz_convert("Asia/Shanghai")
    for label, values, period in (
        ("hour_local", local.dt.hour, 24),
        ("month", local.dt.month - 1, 12),
        ("doy", local.dt.dayofyear - 1, 365.25),
    ):
        data[f"{label}_sin"] = np.sin(2 * np.pi * values / period)
        data[f"{label}_cos"] = np.cos(2 * np.pi * values / period)

    ordered = data.sort_values(
        ["location_id", "lead_time", "forecast_issue_time_utc"]
    ).copy()
    grouped = ordered.groupby(["location_id", "lead_time"], sort=False)
    if derived.get("lag_features"):
        for lag in (1, 2):
            ordered[f"ghi_fcst_lag{lag}"] = grouped["ghi_fcst"].shift(lag)
    if derived.get("cloud_change"):
        ordered["cloud_cover_change"] = grouped["cloud_cover_fcst"].diff()
        for window, suffix in ((2, "roll1"), (4, "roll3")):
            ordered[f"cloud_cover_{suffix}"] = grouped["cloud_cover_fcst"].transform(
                lambda values: values.rolling(window, min_periods=1).mean()
            )
    return ordered.sort_index()


def formal_feature_columns(
    frame: pd.DataFrame,
    feature_config: Mapping[str, Any],
) -> list[str]:
    names = list(
        dict.fromkeys(
            name
            for values in feature_config["feature_groups"].values()
            for name in values
        )
    )
    assert_model_features(names, feature_config["policy"])
    missing = set(names) - set(frame)
    if missing:
        raise ContractError(f"formal features missing from frame: {sorted(missing)}")
    return names
