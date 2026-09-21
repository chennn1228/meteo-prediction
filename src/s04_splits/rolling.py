"""Expanding outer folds and fit/purge/early-stop/purge/scoring inner folds.

No tuner may reuse the scoring block as its early-stop block. The configured
three 14-day scoring blocks, two 10-day purges and one 14-day early-stop block
leave a nonempty first fit prefix without changing any outer window.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .split_registry import manifest, protocol, purge_days, utc


class InsufficientHistoryError(ValueError):
    """A configured fold cannot accommodate all ordered train blocks."""


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    fit: pd.DataFrame
    score: pd.DataFrame
    fit_end: pd.Timestamp
    score_start: pd.Timestamp


@dataclass(frozen=True)
class InnerFold:
    fold_id: str
    fit: pd.DataFrame
    early_stop: pd.DataFrame
    score: pd.DataFrame
    fit_end: pd.Timestamp
    early_stop_start: pd.Timestamp
    early_stop_end: pd.Timestamp
    score_start: pd.Timestamp


def times(frame: pd.DataFrame) -> pd.Series:
    if "target_time_utc" not in frame:
        raise KeyError("target_time_utc")
    return pd.to_datetime(frame["target_time_utc"], utc=True)


def window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    t = times(frame)
    return frame.loc[(t >= start) & (t < end)].copy()


def assert_gap(left: pd.DataFrame, right: pd.DataFrame, days: int) -> None:
    if left.empty or right.empty:
        raise InsufficientHistoryError("empty fit/early-stop/scoring block")
    # Boundaries are half-open; a whole-day gap is measured from the next hour.
    left_next = times(left).max() + pd.Timedelta(hours=1)
    if left_next + pd.Timedelta(days=days) > times(right).min():
        raise AssertionError(f"required {days}-day purge violated")


def development_and_test(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = manifest()
    dev = window(frame, utc(m["development_period"]["start"]),
                 utc(m["development_period"]["end"]) + pd.Timedelta(seconds=1))
    test = window(frame, utc(m["test_period"]["start"]),
                  utc(m["test_period"]["end"]) + pd.Timedelta(seconds=1))
    return dev, test


def outer_folds(frame: pd.DataFrame, *, gap_days: int | None = None) -> list[OuterFold]:
    gap = purge_days() if gap_days is None else int(gap_days)
    allowed = tuple(int(v) for v in protocol()["gap_sensitivity_days"])
    if gap not in allowed:
        raise ValueError(f"gap_days must be one of {allowed}")
    development, _ = development_and_test(frame)
    folds = []
    for spec in protocol()["outer_folds"]:
        score_start = utc(spec["validation_start"])
        # Keep the score interval fixed in each sensitivity analysis. A 7-day
        # gap admits three extra fit days; 14 days removes four fit days.
        fit_end = score_start - pd.Timedelta(days=gap)
        fit = window(development, utc(spec["train_start"]), fit_end)
        score = window(development, score_start,
                       utc(spec["validation_end"], inclusive_date_end=True))
        assert_gap(fit, score, gap)
        folds.append(OuterFold(spec["id"], fit, score, fit_end, score_start))
    return folds


def inner_folds(outer: OuterFold | pd.DataFrame, *, n_folds: int | None = None,
                score_days: int | None = None, early_stop_days: int | None = None,
                gap_days: int | None = None) -> list[InnerFold]:
    """Generate expanding inner folds wholly inside the outer-training prefix.

    Scoring blocks are consecutive and disjoint. The early-stop block is a
    separate earlier block, with one purge on either side. The manifest fixes
    the number of folds, purge, and both block durations.
    """
    frame = outer.fit if isinstance(outer, OuterFold) else outer
    n = int(manifest()["tuning_budget"]["common_inner_folds"]) if n_folds is None else n_folds
    gap = purge_days() if gap_days is None else int(gap_days)
    score_days = int(protocol()["inner_scoring_days"]) if score_days is None else score_days
    early_stop_days = int(protocol()["inner_early_stop_days"]) if early_stop_days is None else early_stop_days
    if n < 1 or score_days < 1 or early_stop_days < 1:
        raise ValueError("positive n_folds and block durations required")
    t = times(frame)
    if t.empty:
        raise InsufficientHistoryError("empty outer training prefix")
    origin = utc(manifest()["development_period"]["start"])
    end = t.max().floor("D") + pd.Timedelta(days=1)
    score_span = pd.Timedelta(days=score_days)
    early_span = pd.Timedelta(days=early_stop_days)
    purge = pd.Timedelta(days=gap)
    folds = []
    for index in range(n):
        score_end = end - (n - index - 1) * score_span
        score_start = score_end - score_span
        early_end = score_start - purge
        early_start = early_end - early_span
        fit_end = early_start - purge
        if fit_end <= origin:
            raise InsufficientHistoryError(
                f"inner_{index + 1}: fit_end={fit_end.date()} <= development start="
                f"{origin.date()}; {n}×{score_days}d scoring + {early_stop_days}d "
                f"early stop + 2×{gap}d purge cannot fit outer prefix"
            )
        fit = window(frame, origin, fit_end)
        early = window(frame, early_start, early_end)
        score = window(frame, score_start, score_end)
        assert_gap(fit, early, gap)
        assert_gap(early, score, gap)
        folds.append(InnerFold(f"inner_{index + 1}", fit, early, score,
                               fit_end, early_start, early_end, score_start))
    return folds
