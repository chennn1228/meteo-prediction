"""Quantile reliability, interior PIT and truly group-local metric summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from s07_prediction.schema import QUANTILES, QUANTILE_COLUMNS, validate_predictions
from .probabilistic import _validated, probability_metrics


def quantile_reliability(y: object, q: object, *, taus: object = QUANTILES) -> pd.DataFrame:
    observed, forecast, levels = _validated(y, q, taus)
    empirical = np.mean(observed[:, None] <= forecast, axis=0)
    return pd.DataFrame({"quantile": levels, "empirical_cdf": empirical,
                         "calibration_error": empirical - levels,
                         "n": len(observed)})


def pit_central(y: object, q: object, *, taus: object = QUANTILES) -> np.ndarray:
    """Interpolate PIT only inside the known quantile range.

    Outside [q0.05, q0.95] the tails are unknown and PIT is NaN. For tied
    quantiles at the observation, use the midpoint of their probability span.
    This is an approximate, interval-censored diagnostic, not a full PIT.
    """
    observed, forecast, levels = _validated(y, q, taus)
    if np.any(np.diff(forecast, axis=1) < 0):
        raise ValueError("PIT requires noncrossing quantiles; report crossing first")
    values = np.full(len(observed), np.nan)
    for i, (truth, quantiles) in enumerate(zip(observed, forecast)):
        if truth < quantiles[0] or truth > quantiles[-1]:
            continue
        ties = np.flatnonzero(quantiles == truth)
        if len(ties):
            values[i] = float((levels[ties[0]] + levels[ties[-1]]) / 2)
        else:
            values[i] = float(np.interp(truth, quantiles, levels))
    return values


def grouped_metrics(frame: pd.DataFrame, by: str | list[str] | tuple[str, ...]
                    ) -> pd.DataFrame:
    """Recompute all metrics, including crossing, independently per group."""
    clean = validate_predictions(frame)
    if (clean.prediction_type != "quantile").any():
        raise ValueError("grouped probability metrics cannot consume point-only rows")
    names = [by] if isinstance(by, str) else list(by)
    if not names or any(name not in clean.columns for name in names):
        raise ValueError("by must name existing grouping columns")
    rows: list[dict[str, object]] = []
    for key, subset in clean.groupby(names, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        result = dict(zip(names, key_tuple))
        result["n"] = len(subset)
        result.update(probability_metrics(subset.y.to_numpy(),
                                          subset.loc[:, QUANTILE_COLUMNS].to_numpy()))
        rows.append(result)
    return pd.DataFrame(rows)
