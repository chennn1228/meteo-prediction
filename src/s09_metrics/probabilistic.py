"""Seven-quantile scores. Truncated quadrature is never called full CRPS."""

from __future__ import annotations

import numpy as np

from s07_prediction.schema import QUANTILES
from .deterministic import point_metrics


def _validated(y: object, q: object, taus: object = QUANTILES
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.asarray(y, dtype=float)
    forecast = np.asarray(q, dtype=float)
    levels = np.asarray(taus, dtype=float)
    if (observed.ndim != 1 or not len(observed) or forecast.shape != (len(observed), len(levels))
            or levels.ndim != 1 or len(levels) < 2):
        raise ValueError("expected nonempty y and [sample, quantile] forecast matrix")
    if not np.isfinite(observed).all() or not np.isfinite(forecast).all():
        raise ValueError("probability metrics require finite observations and quantiles")
    if not np.all((0 < levels) & (levels < 1)) or not np.all(np.diff(levels) > 0):
        raise ValueError("quantile levels must be increasing and strictly inside (0,1)")
    return observed, forecast, levels


def quantile_losses(y: object, q: object, taus: object = QUANTILES) -> np.ndarray:
    """Per-sample, per-quantile pinball losses; crossing is not repaired here."""
    observed, forecast, levels = _validated(y, q, taus)
    residual = observed[:, None] - forecast
    return np.maximum(levels[None, :] * residual, (levels[None, :] - 1) * residual)


def truncated_quantile_crps(y: object, q: object, taus: object = QUANTILES) -> float:
    """2 × trapezoidal ∫ pinball dτ only over [min(τ), max(τ)].

    It omits both probability tails, so is NOT full CRPS and must not be
    reported under an unqualified ``crps`` name.
    """
    _, _, levels = _validated(y, q, taus)
    loss_by_level = quantile_losses(y, q, levels).mean(axis=0)
    # Explicit trapezoid keeps compatibility with NumPy 1.x and 2.x.
    return float(np.sum((loss_by_level[:-1] + loss_by_level[1:]) * np.diff(levels)))


def probability_metrics(y: object, q: object, *, taus: object = QUANTILES,
                        reference_q50: object | None = None) -> dict[str, float]:
    observed, forecast, levels = _validated(y, q, taus)
    if not np.array_equal(levels, np.asarray(QUANTILES)):
        raise ValueError("formal probability metrics require the canonical seven quantiles")
    losses = quantile_losses(observed, forecast, levels)
    result = {f"pinball_q{level:.2f}": float(losses[:, i].mean())
              for i, level in enumerate(levels)}
    result["mean_pinball"] = float(losses.mean())
    result["crps_q7_trunc"] = truncated_quantile_crps(observed, forecast, levels)
    result["crossing_rate"] = float(np.mean(np.any(np.diff(forecast, axis=1) < 0, axis=1)))
    for nominal, low, high in ((50, 2, 4), (80, 1, 5), (90, 0, 6)):
        lower, upper = forecast[:, low], forecast[:, high]
        result[f"coverage_{nominal}"] = float(np.mean((observed >= lower) & (observed <= upper)))
        result[f"interval_width_{nominal}"] = float(np.mean(upper - lower))
        result[f"sharpness_{nominal}"] = result[f"interval_width_{nominal}"]
    result.update(point_metrics(observed, forecast[:, 3],
                                reference_prediction=reference_q50))
    return result
