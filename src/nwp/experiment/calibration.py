"""Chronological quantile calibration and UTC calendar-day bootstrap."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .prediction import quantile_columns, quantiles, validate_predictions


def _timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} requires an explicit timezone")
    return timestamp.tz_convert("UTC")


def additive_residual_offsets(y: object, q: object,
                              levels: object) -> np.ndarray:
    truth = np.asarray(y, dtype=float)
    forecast = np.asarray(q, dtype=float)
    taus = np.asarray(levels, dtype=float)
    if (truth.ndim != 1 or not len(truth)
            or forecast.shape != (len(truth), len(taus))
            or not np.isfinite(truth).all()
            or not np.isfinite(forecast).all()):
        raise ValueError("finite y and [sample, quantile] predictions required")
    if not np.all((0 < taus) & (taus < 1)) or not np.all(np.diff(taus) > 0):
        raise ValueError("quantile levels must increase inside (0,1)")
    residual = truth[:, None] - forecast
    return np.array([np.quantile(residual[:, index], tau)
                     for index, tau in enumerate(taus)], dtype=float)


def apply_additive_offsets(q: object, offsets: object, *,
                           lower_bound: float | None = None,
                           upper_bound: float | None = None) -> np.ndarray:
    forecast = np.asarray(q, dtype=float)
    adjustment = np.asarray(offsets, dtype=float)
    if (forecast.ndim != 2 or forecast.shape[1] != len(adjustment)
            or not np.isfinite(forecast).all()
            or not np.isfinite(adjustment).all()):
        raise ValueError(
            "finite quantile matrix and matching finite offsets required")
    result = np.sort(forecast + adjustment, axis=1)
    if lower_bound is not None or upper_bound is not None:
        result = np.clip(
            result, -np.inf if lower_bound is None else lower_bound,
            np.inf if upper_bound is None else upper_bound)
    return result


@dataclass
class AdditiveQuantileCalibrator:
    protocol_config: Mapping[str, Any]
    lower_bound: float | None = 0.0
    deltas_: dict[str, float] = field(default_factory=dict, init=False)
    fit_end_: pd.Timestamp | None = field(default=None, init=False)
    early_stop_end_: pd.Timestamp | None = field(default=None, init=False)
    calibration_end_: pd.Timestamp | None = field(default=None, init=False)
    experiment_id_: str | None = field(default=None, init=False)

    def fit(self, calibration: pd.DataFrame, *, fit_end: object,
            early_stop_end: object) -> "AdditiveQuantileCalibrator":
        clean = validate_predictions(calibration, self.protocol_config)
        if (clean.prediction_type != "quantile").any():
            raise ValueError("calibration requires seven-quantile predictions")
        if clean.experiment_id.nunique() != 1:
            raise ValueError("calibration rows must belong to one experiment")
        fit_boundary = _timestamp(fit_end, "fit_end")
        stop_boundary = _timestamp(early_stop_end, "early_stop_end")
        if fit_boundary >= stop_boundary:
            raise ValueError("fit must end before early stopping ends")
        if stop_boundary >= clean.target_time_utc.min():
            raise ValueError("calibration truth must follow early stopping")
        if stop_boundary >= clean.forecast_issue_time_utc.min():
            raise ValueError("calibration forecast issues must follow early stopping")
        columns = quantile_columns(self.protocol_config)
        offsets = additive_residual_offsets(
            clean.y.to_numpy(), clean.loc[:, columns].to_numpy(),
            quantiles(self.protocol_config))
        self.deltas_ = dict(zip(columns, offsets.tolist()))
        self.fit_end_ = fit_boundary
        self.early_stop_end_ = stop_boundary
        self.calibration_end_ = clean.target_time_utc.max()
        self.experiment_id_ = str(clean.experiment_id.iloc[0])
        return self

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not self.deltas_ or self.calibration_end_ is None:
            raise RuntimeError("fit the calibrator before apply")
        clean = validate_predictions(
            predictions, self.protocol_config, require_truth=False)
        if (clean.prediction_type != "quantile").any():
            raise ValueError(
                "calibration cannot create a distribution from point forecasts")
        if (clean.experiment_id != self.experiment_id_).any():
            raise ValueError(
                "calibration and predictions must have the same experiment_id")
        if (clean.forecast_issue_time_utc <= self.calibration_end_).any():
            raise ValueError(
                "calibration truth overlaps or follows a forecast issue time")
        columns = quantile_columns(self.protocol_config)
        repaired = apply_additive_offsets(
            clean.loc[:, columns].to_numpy(dtype=float),
            [self.deltas_[column] for column in columns],
            lower_bound=self.lower_bound)
        clean.loc[:, columns] = repaired
        clean.loc[:, "point_prediction"] = repaired[:, quantiles(
            self.protocol_config).index(0.5)]
        clean["calibration_method"] = (
            "additive_signed_residual_quantile_rearranged")
        clean["calibration_end_utc"] = self.calibration_end_.isoformat()
        clean["calibration_fit_end_utc"] = self.fit_end_.isoformat()
        clean["calibration_early_stop_end_utc"] = (
            self.early_stop_end_.isoformat())
        return validate_predictions(
            clean, self.protocol_config, require_truth=False)


@dataclass
class CausalIssueQuantileCalibrator:
    protocol_config: Mapping[str, Any]
    lower_bound: float | None = 0.0
    calibration_: pd.DataFrame | None = field(default=None, init=False)
    fit_end_: pd.Timestamp | None = field(default=None, init=False)
    early_stop_end_: pd.Timestamp | None = field(default=None, init=False)
    identity_: dict[str, object] = field(default_factory=dict, init=False)

    def fit(self, calibration: pd.DataFrame, *, fit_end: object,
            early_stop_end: object) -> "CausalIssueQuantileCalibrator":
        clean = validate_predictions(calibration, self.protocol_config)
        if (clean.prediction_type != "quantile").any():
            raise ValueError("calibration requires seven-quantile predictions")
        fit_boundary = _timestamp(fit_end, "fit_end")
        stop_boundary = _timestamp(early_stop_end, "early_stop_end")
        if fit_boundary >= stop_boundary:
            raise ValueError("fit must end before early stopping ends")
        if stop_boundary >= clean.target_time_utc.min():
            raise ValueError("calibration truth must follow early stopping")
        if stop_boundary >= clean.forecast_issue_time_utc.min():
            raise ValueError("calibration forecast issues must follow early stopping")
        identity_columns = (
            "experiment_id", "model_id", "seed", "data_version",
            "feature_version", "protocol_version")
        for column in identity_columns:
            if clean[column].nunique(dropna=False) != 1:
                raise ValueError(f"calibration mixes {column}")
        self.identity_ = {column: clean[column].iloc[0]
                          for column in identity_columns}
        self.calibration_ = clean.sort_values(
            "target_time_utc").reset_index(drop=True)
        self.fit_end_ = fit_boundary
        self.early_stop_end_ = stop_boundary
        return self

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if self.calibration_ is None:
            raise RuntimeError("fit the calibrator before apply")
        clean = validate_predictions(
            predictions, self.protocol_config, require_truth=False)
        if (clean.prediction_type != "quantile").any():
            raise ValueError(
                "calibration cannot create a distribution from point forecasts")
        for column, value in self.identity_.items():
            if (clean[column] != value).any():
                raise ValueError(
                    f"calibration and predictions disagree on {column}")
        if (clean.target_time_utc
                <= self.calibration_.target_time_utc.max()).any():
            raise ValueError("prediction target overlaps the calibration block")
        result = clean.copy()
        result["calibration_available_rows"] = 0
        result["calibration_latest_truth_utc"] = ""
        columns = quantile_columns(self.protocol_config)
        levels = quantiles(self.protocol_config)
        median_index = levels.index(0.5)
        for issue, positions in result.groupby(
                "forecast_issue_time_utc", sort=True).indices.items():
            available = self.calibration_.loc[
                self.calibration_.target_time_utc < issue]
            if available.empty:
                raise ValueError(
                    f"no calibration truth known before forecast issue {issue}")
            offsets = additive_residual_offsets(
                available.y.to_numpy(dtype=float),
                available.loc[:, columns].to_numpy(dtype=float), levels)
            original = result.iloc[positions].loc[
                :, columns].to_numpy(dtype=float)
            calibrated = apply_additive_offsets(
                original, offsets, lower_bound=self.lower_bound)
            result.iloc[positions,
                        result.columns.get_indexer(columns)] = calibrated
            result.iloc[positions,
                        result.columns.get_loc("point_prediction")] = (
                            calibrated[:, median_index])
            result.iloc[positions,
                        result.columns.get_loc(
                            "calibration_available_rows")] = len(available)
            result.iloc[positions,
                        result.columns.get_loc(
                            "calibration_latest_truth_utc")] = (
                                available.target_time_utc.max().isoformat())
        result["calibration_method"] = (
            "causal_issue_additive_signed_residual_quantile_rearranged")
        result["calibration_fit_end_utc"] = self.fit_end_.isoformat()
        result["calibration_early_stop_end_utc"] = (
            self.early_stop_end_.isoformat())
        return validate_predictions(
            result, self.protocol_config, require_truth=False)


def block_bootstrap_indices(times: object, *, block_days: int = 7,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    if isinstance(block_days, bool) or not isinstance(block_days, int) or block_days < 1:
        raise ValueError("block_days must be a positive integer")
    series = pd.Series(times)
    if series.empty:
        raise ValueError("times must not be empty")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError("times contain invalid timestamps")
    days = parsed.dt.normalize().to_numpy(dtype="datetime64[ns]")
    unique_days = np.unique(days)
    random = rng if rng is not None else np.random.default_rng()
    chosen: list[np.ndarray] = []
    sampled_days = 0
    while sampled_days < len(unique_days):
        start = unique_days[int(random.integers(len(unique_days)))]
        end = start + np.timedelta64(block_days, "D")
        within = np.flatnonzero((days >= start) & (days < end))
        sampled_days += len(np.unique(days[within]))
        chosen.append(within)
    return np.concatenate(chosen)


def bootstrap_metric(frame: pd.DataFrame,
                     statistic: Callable[[pd.DataFrame], float], *,
                     time_col: str = "target_time_utc", block_days: int = 7,
                     replicates: int = 1000, seed: int = 0,
                     confidence: float = 0.95) -> dict[str, object]:
    if frame.empty or time_col not in frame:
        raise ValueError("nonempty frame with time_col required")
    if replicates < 2 or not (0 < confidence < 1):
        raise ValueError("replicates >= 2 and 0 < confidence < 1 required")
    random = np.random.default_rng(seed)
    estimates = np.array([
        statistic(frame.iloc[block_bootstrap_indices(
            frame[time_col], block_days=block_days, rng=random)])
        for _ in range(replicates)], dtype=float)
    if not np.isfinite(estimates).all():
        raise ValueError("bootstrap statistic returned nonfinite estimates")
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(statistic(frame)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1 - alpha)),
        "confidence": confidence, "block_days": block_days,
        "replicates": replicates, "seed": seed,
        "time_basis": "UTC calendar days",
    }
