"""Moving-block resampling over actual UTC calendar days, not row counts."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def block_bootstrap_indices(times: object, *, block_days: int = 7,
                            rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample contiguous calendar-day windows, retaining every row per day.

    A 7-day block means [start midnight UTC, start+7 calendar days), regardless
    of hourly coverage, location count, or row ordering. Repeated blocks are
    represented by repeated row indices. Missing days remain missing.
    """
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
        # The sampled start itself always has records, so this cannot be empty.
        sampled_days += len(np.unique(days[within]))
        chosen.append(within)
    return np.concatenate(chosen)


def bootstrap_metric(frame: pd.DataFrame, statistic: Callable[[pd.DataFrame], float],
                     *, time_col: str = "target_time_utc", block_days: int = 7,
                     replicates: int = 1000, seed: int = 0,
                     confidence: float = 0.95) -> dict[str, object]:
    """Calendar-block percentile CI; development diagnostic, not an official run."""
    if frame.empty or time_col not in frame:
        raise ValueError("nonempty frame with time_col required")
    if replicates < 2 or not (0 < confidence < 1):
        raise ValueError("replicates >= 2 and 0 < confidence < 1 required")
    random = np.random.default_rng(seed)
    estimates = np.array([
        statistic(frame.iloc[block_bootstrap_indices(frame[time_col], block_days=block_days,
                                                    rng=random)])
        for _ in range(replicates)
    ], dtype=float)
    if not np.isfinite(estimates).all():
        raise ValueError("bootstrap statistic returned nonfinite estimates")
    alpha = (1 - confidence) / 2
    return {
        "estimate": float(statistic(frame)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1 - alpha)),
        "confidence": confidence,
        "block_days": block_days,
        "replicates": replicates,
        "seed": seed,
        "time_basis": "UTC calendar days",
    }
