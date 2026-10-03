"""Deterministic, seven-quantile, reliability, and central-PIT metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _pair(y: object, prediction: object) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(y, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    if observed.ndim != 1 or forecast.shape != observed.shape or not len(observed):
        raise ValueError("y and prediction must be nonempty equal-length vectors")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("point metrics require finite y and prediction")
    return observed, forecast


def deterministic(y: object, prediction: object, *,
                  reference_prediction: object | None = None) -> dict[str, float]:
    observed, forecast = _pair(y, prediction)
    error = forecast - observed
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }
    variance = float(np.sum((observed - observed.mean()) ** 2))
    result["r2"] = (float(1 - np.sum(error ** 2) / variance)
                    if variance > 0 else float("nan"))
    if reference_prediction is not None:
        _, reference = _pair(observed, reference_prediction)
        reference_rmse = float(np.sqrt(np.mean((reference - observed) ** 2)))
        if reference_rmse <= 0:
            raise ValueError("reference RMSE must be positive for skill")
        result["rmse_skill"] = float(
            1 - result["rmse"] / reference_rmse)
    return result


point_metrics = deterministic


def _validated(y: object, q: object, levels: object
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(y, dtype=float)
    forecast = np.asarray(q, dtype=float)
    quantiles = np.asarray(levels, dtype=float)
    if (observed.ndim != 1 or not len(observed)
            or forecast.shape != (len(observed), len(quantiles))
            or quantiles.ndim != 1 or len(quantiles) < 2):
        raise ValueError(
            "expected nonempty y and [sample, quantile] forecast matrix")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError(
            "probability metrics require finite observations and quantiles")
    if (not np.all((0 < quantiles) & (quantiles < 1))
            or not np.all(np.diff(quantiles) > 0)):
        raise ValueError(
            "quantile levels must be increasing and strictly inside (0,1)")
    return observed, forecast, quantiles


def quantile_losses(y: object, q: object, levels: object) -> np.ndarray:
    observed, forecast, quantiles = _validated(y, q, levels)
    residual = observed[:, None] - forecast
    return np.maximum(
        quantiles[None, :] * residual,
        (quantiles[None, :] - 1) * residual)


def mean_pinball(observed: np.ndarray, predicted: np.ndarray,
                 quantiles: tuple[float, ...]) -> float:
    return float(quantile_losses(observed, predicted, quantiles).mean())


def truncated_quantile_crps(y: object, q: object, levels: object) -> float:
    """2 x trapezoidal integral only over the registered central grid."""
    _, _, quantiles = _validated(y, q, levels)
    loss = quantile_losses(y, q, quantiles).mean(axis=0)
    return float(np.sum(
        (loss[:-1] + loss[1:]) * np.diff(quantiles)))


def probability_metrics(y: object, q: object, *, levels: object,
                        reference_q50: object | None = None) -> dict[str, float]:
    observed, forecast, quantiles = _validated(y, q, levels)
    if len(quantiles) != 7 or 0.5 not in quantiles:
        raise ValueError("formal probability metrics require seven quantiles including 0.50")
    losses = quantile_losses(observed, forecast, quantiles)
    result = {f"pinball_q{level:.2f}": float(losses[:, index].mean())
              for index, level in enumerate(quantiles)}
    result["mean_pinball"] = float(losses.mean())
    result["crps_q7_trunc"] = truncated_quantile_crps(
        observed, forecast, quantiles)
    result["crossing_rate"] = float(np.mean(
        np.any(np.diff(forecast, axis=1) < 0, axis=1)))
    for nominal, low, high in ((50, 2, 4), (80, 1, 5), (90, 0, 6)):
        lower, upper = forecast[:, low], forecast[:, high]
        result[f"coverage_{nominal}"] = float(np.mean(
            (observed >= lower) & (observed <= upper)))
        result[f"interval_width_{nominal}"] = float(np.mean(upper - lower))
        result[f"sharpness_{nominal}"] = result[f"interval_width_{nominal}"]
    result.update(deterministic(
        observed, forecast[:, list(quantiles).index(0.5)],
        reference_prediction=reference_q50))
    return result


def quantile_reliability(y: object, q: object, *,
                         levels: object) -> pd.DataFrame:
    observed, forecast, quantiles = _validated(y, q, levels)
    empirical = np.mean(observed[:, None] <= forecast, axis=0)
    return pd.DataFrame({
        "quantile": quantiles, "empirical_cdf": empirical,
        "calibration_error": empirical - quantiles, "n": len(observed)})


def pit_central(y: object, q: object, *, levels: object) -> np.ndarray:
    observed, forecast, quantiles = _validated(y, q, levels)
    if np.any(np.diff(forecast, axis=1) < 0):
        raise ValueError("PIT requires noncrossing quantiles; report crossing first")
    values = np.full(len(observed), np.nan)
    for index, (truth, row) in enumerate(zip(observed, forecast)):
        if truth < row[0] or truth > row[-1]:
            continue
        ties = np.flatnonzero(row == truth)
        values[index] = (float((quantiles[ties[0]] + quantiles[ties[-1]]) / 2)
                         if len(ties) else float(np.interp(truth, row, quantiles)))
    return values
