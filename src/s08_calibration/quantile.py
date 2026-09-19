"""Signed-residual additive calibration fitted only on a later calibration fold."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from s07_prediction.schema import QUANTILES, QUANTILE_COLUMNS, validate_predictions


def _timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} requires an explicit timezone")
    return timestamp.tz_convert("UTC")


def additive_residual_offsets(y: object, q: object,
                              taus: object = QUANTILES) -> np.ndarray:
    """One shared implementation of delta_tau = quantile(y - q_tau, tau)."""
    truth = np.asarray(y, dtype=float)
    forecast = np.asarray(q, dtype=float)
    levels = np.asarray(taus, dtype=float)
    if (truth.ndim != 1 or not len(truth) or forecast.shape != (len(truth), len(levels))
            or not np.isfinite(truth).all() or not np.isfinite(forecast).all()):
        raise ValueError("finite y and [sample, quantile] predictions required")
    if not np.all((0 < levels) & (levels < 1)) or not np.all(np.diff(levels) > 0):
        raise ValueError("quantile levels must increase inside (0,1)")
    residual = truth[:, None] - forecast
    return np.array([np.quantile(residual[:, i], tau)
                     for i, tau in enumerate(levels)], dtype=float)


def apply_additive_offsets(q: object, offsets: object,
                           *, lower_bound: float | None = None,
                           upper_bound: float | None = None) -> np.ndarray:
    """Apply signed offsets, rearrange crossings and optionally clip in W/m²."""
    forecast = np.asarray(q, dtype=float)
    adjustment = np.asarray(offsets, dtype=float)
    if (forecast.ndim != 2 or forecast.shape[1] != len(adjustment)
            or not np.isfinite(forecast).all() or not np.isfinite(adjustment).all()):
        raise ValueError("finite quantile matrix and matching finite offsets required")
    result = np.sort(forecast + adjustment, axis=1)
    if lower_bound is not None or upper_bound is not None:
        result = np.clip(result,
                         -np.inf if lower_bound is None else lower_bound,
                         np.inf if upper_bound is None else upper_bound)
    return result


@dataclass
class AdditiveQuantileCalibrator:
    """Add quantile_tau(y - qhat_tau) and repair crossing by rearrangement.

    `fit_end` and `early_stop_end` are truth-time boundaries, not row indices.
    Every application forecast issue must be after the latest calibration truth:
    otherwise that sample would consume future observations.
    """

    lower_bound: float | None = 0.0
    deltas_: dict[str, float] = field(default_factory=dict, init=False)
    fit_end_: pd.Timestamp | None = field(default=None, init=False)
    early_stop_end_: pd.Timestamp | None = field(default=None, init=False)
    calibration_end_: pd.Timestamp | None = field(default=None, init=False)
    experiment_id_: str | None = field(default=None, init=False)

    def fit(self, calibration: pd.DataFrame, *, fit_end: object,
            early_stop_end: object) -> "AdditiveQuantileCalibrator":
        clean = validate_predictions(calibration)
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
        offsets = additive_residual_offsets(clean.y.to_numpy(),
                                            clean.loc[:, QUANTILE_COLUMNS].to_numpy())
        self.deltas_ = dict(zip(QUANTILE_COLUMNS, offsets.tolist()))
        self.fit_end_ = fit_boundary
        self.early_stop_end_ = stop_boundary
        self.calibration_end_ = clean.target_time_utc.max()
        self.experiment_id_ = str(clean.experiment_id.iloc[0])
        return self

    def apply(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if not self.deltas_ or self.calibration_end_ is None:
            raise RuntimeError("fit the calibrator before apply")
        clean = validate_predictions(predictions, require_truth=False)
        if (clean.prediction_type != "quantile").any():
            raise ValueError("calibration cannot create a distribution from point forecasts")
        if (clean.experiment_id != self.experiment_id_).any():
            raise ValueError("calibration and predictions must have the same experiment_id")
        if (clean.forecast_issue_time_utc <= self.calibration_end_).any():
            raise ValueError("calibration truth overlaps or follows a forecast issue time")
        raw = clean.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
        # Rearrangement is a documented calibration operation, not an eval-time
        # suppression of crossing. Crossing must be scored on raw predictions.
        repaired = apply_additive_offsets(raw,
                                          [self.deltas_[column] for column in QUANTILE_COLUMNS],
                                          lower_bound=self.lower_bound)
        clean.loc[:, QUANTILE_COLUMNS] = repaired
        clean.loc[:, "point_prediction"] = repaired[:, 3]
        clean["calibration_method"] = "additive_signed_residual_quantile_rearranged"
        clean["calibration_end_utc"] = self.calibration_end_.isoformat()
        clean["calibration_fit_end_utc"] = self.fit_end_.isoformat()
        clean["calibration_early_stop_end_utc"] = self.early_stop_end_.isoformat()
        return validate_predictions(clean, require_truth=False)
