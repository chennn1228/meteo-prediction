"""Auxiliary deterministic metrics for q0.50 or a point-only baseline."""

from __future__ import annotations

import numpy as np


def _pair(y: object, prediction: object) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(y, dtype=float)
    forecast = np.asarray(prediction, dtype=float)
    if observed.ndim != 1 or forecast.shape != observed.shape or not len(observed):
        raise ValueError("y and prediction must be nonempty equal-length vectors")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("point metrics require finite y and prediction")
    return observed, forecast


def point_metrics(y: object, prediction: object,
                  *, reference_prediction: object | None = None) -> dict[str, float]:
    """Bias is forecast minus truth; RMSE skill uses 1 - RMSE/RMSE_ref."""
    observed, forecast = _pair(y, prediction)
    error = forecast - observed
    result = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }
    if reference_prediction is not None:
        result["rmse_skill"] = rmse_skill(observed, forecast, reference_prediction)
    return result


def rmse_skill(y: object, prediction: object, reference_prediction: object) -> float:
    observed, forecast = _pair(y, prediction)
    _, reference = _pair(observed, reference_prediction)
    rmse_reference = float(np.sqrt(np.mean((reference - observed) ** 2)))
    if rmse_reference <= 0:
        raise ValueError("reference RMSE must be positive for skill")
    return float(1 - np.sqrt(np.mean((forecast - observed) ** 2)) / rmse_reference)
