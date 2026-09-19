"""Time-causal development/test splitting for the formal v2 protocol.

All intervals are half-open in code. The public test year is fixed to
2025-09-01 through 2026-08-31. Model selection uses expanding-window,
purged rolling-origin folds. The 10-day purge is derived from the 168-hour
maximum sequence lookback plus the 72-hour maximum forecast lead.

``month_block_folds`` remains only so legacy result scripts can be reproduced;
new training and tuning code must use ``nested_rolling_folds``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

DEV_START = pd.Timestamp("2024-02-01", tz="UTC")
TEST_START = pd.Timestamp("2025-09-01", tz="UTC")
TEST_END = pd.Timestamp("2026-09-01", tz="UTC")
MAX_SEQUENCE_LOOKBACK_HOURS = 168
MAX_LEAD_HOURS = 72
PURGE_HOURS = MAX_SEQUENCE_LOOKBACK_HOURS + MAX_LEAD_HOURS
PURGE = pd.Timedelta(hours=PURGE_HOURS)

FINAL_EARLY_STOP_START = pd.Timestamp("2025-07-01", tz="UTC")
FINAL_EARLY_STOP_END = pd.Timestamp("2025-08-01", tz="UTC")
FINAL_CALIBRATION_START = pd.Timestamp("2025-08-11", tz="UTC")
FINAL_CALIBRATION_END = TEST_START


@dataclass(frozen=True)
class RollingFoldSpec:
    fold_id: str
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


OUTER_SPECS = (
    RollingFoldSpec("outer_1", pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-06-11", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC")),
    RollingFoldSpec("outer_2", pd.Timestamp("2024-09-01", tz="UTC"),
                    pd.Timestamp("2024-09-11", tz="UTC"), pd.Timestamp("2024-11-01", tz="UTC")),
    RollingFoldSpec("outer_3", pd.Timestamp("2024-12-01", tz="UTC"),
                    pd.Timestamp("2024-12-11", tz="UTC"), pd.Timestamp("2025-02-01", tz="UTC")),
    RollingFoldSpec("outer_4", pd.Timestamp("2025-03-01", tz="UTC"),
                    pd.Timestamp("2025-03-11", tz="UTC"), pd.Timestamp("2025-05-01", tz="UTC")),
    RollingFoldSpec("outer_5", pd.Timestamp("2025-06-01", tz="UTC"),
                    pd.Timestamp("2025-06-11", tz="UTC"), pd.Timestamp("2025-08-01", tz="UTC")),
)


def _time(df: pd.DataFrame) -> pd.Series:
    if "target_time_utc" not in df:
        raise KeyError("target_time_utc is required")
    return pd.to_datetime(df["target_time_utc"], utc=True)


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    t = _time(df)
    return df[(t >= start) & (t < end)].copy()


def split_test(df: pd.DataFrame):
    """Return the bounded development period and untouched 12-month test."""
    return _slice(df, DEV_START, TEST_START), _slice(df, TEST_START, TEST_END)


def default_split(df: pd.DataFrame):
    """Return (training prefix, later calibration, final test).

    The 1--10 August interval is an explicit purge between the training prefix
    and calibration. Call :func:`split_early_stop` on the training prefix when
    an estimator requires early stopping.
    """
    training = _slice(df, DEV_START, FINAL_EARLY_STOP_END)
    calibration = _slice(df, FINAL_CALIBRATION_START, FINAL_CALIBRATION_END)
    test = _slice(df, TEST_START, TEST_END)
    return training, calibration, test


def split_early_stop(training: pd.DataFrame):
    """Split a final training prefix into fit and temporally later early-stop."""
    fit_end = FINAL_EARLY_STOP_START - PURGE
    fit = _slice(training, DEV_START, fit_end)
    early_stop = _slice(training, FINAL_EARLY_STOP_START, FINAL_EARLY_STOP_END)
    return fit, early_stop


def nested_rolling_folds(pool: pd.DataFrame, specs=OUTER_SPECS):
    """Return expanding outer folds with a strict 10-day train/validation gap."""
    folds = []
    for spec in specs:
        train = _slice(pool, DEV_START, spec.train_end)
        valid = _slice(pool, spec.validation_start, spec.validation_end)
        if train.empty or valid.empty:
            raise ValueError(f"{spec.fold_id} is empty for the supplied data")
        if _time(train).max() + PURGE >= _time(valid).min():
            raise AssertionError(f"{spec.fold_id} violates the required purge")
        folds.append((train, valid, spec))
    return folds


def inner_rolling_folds(outer_training: pd.DataFrame, n_folds=3,
                        validation_days=30, purge=PURGE):
    """Create expanding inner folds wholly inside an outer-training prefix."""
    t = _time(outer_training)
    start = max(DEV_START, t.min().floor("D"))
    end = t.max().floor("D") + pd.Timedelta(days=1)
    val_span = pd.Timedelta(days=validation_days)
    folds = []
    for offset in reversed(range(n_folds)):
        val_end = end - offset * val_span
        val_start = val_end - val_span
        train_end = val_start - purge
        train = _slice(outer_training, start, train_end)
        valid = _slice(outer_training, val_start, val_end)
        if train.empty or valid.empty:
            raise ValueError("insufficient history for requested inner rolling folds")
        folds.append((train, valid))
    return folds


def month_block_folds(pool, n_folds=5, block_days=4, seed=0):
    """Legacy-only month-balanced folds; forbidden for formal model selection."""
    warnings.warn(
        "month_block_folds is legacy-only; use nested_rolling_folds",
        DeprecationWarning,
        stacklevel=2,
    )
    del block_days, seed
    local = _time(pool).dt.tz_convert("Asia/Shanghai")
    work = pool.assign(_date=local.dt.normalize(),
                       _month=local.dt.tz_localize(None).dt.to_period("M"))
    val_dates = {i: set() for i in range(n_folds)}
    for _, group in work.groupby("_month"):
        segments = np.array_split(np.sort(group["_date"].unique()), n_folds)
        for k, segment in enumerate(segments):
            val_dates[k].update(pd.to_datetime(segment))
    folds = []
    for i in range(n_folds):
        mask = work["_date"].isin(val_dates[i]).to_numpy()
        folds.append((work[~mask].copy(), work[mask].copy()))
    return folds
