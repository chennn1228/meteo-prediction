"""RunConfig-driven nested purged rolling-origin splits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


class InsufficientHistoryError(ValueError):
    """A registered fold cannot accommodate all ordered blocks."""


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


def utc(value: str, *, inclusive_date_end: bool = False) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    if inclusive_date_end and len(str(value)) == 10:
        stamp += pd.Timedelta(days=1)
    return stamp


def times(frame: pd.DataFrame) -> pd.Series:
    if "target_time_utc" not in frame:
        raise KeyError("target_time_utc")
    return pd.to_datetime(frame["target_time_utc"], utc=True)


def window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    target = times(frame)
    return frame.loc[(target >= start) & (target < end)].copy()


def purge_days(validation: Mapping[str, Any]) -> int:
    hours = int(validation["purge_hours"])
    if hours != int(validation["maximum_sequence_lookback_hours"]) + int(
        validation["maximum_lead_hours"]
    ):
        raise ValueError(
            "purge_hours disagrees with sequence lookback plus maximum lead"
        )
    if hours % 24:
        raise ValueError("day-based rolling windows require an integral-day purge")
    return hours // 24


def _validated_gap(validation: Mapping[str, Any], gap_days: int | None) -> int:
    gap = purge_days(validation) if gap_days is None else int(gap_days)
    allowed = tuple(int(value) for value in validation["gap_sensitivity_days"])
    if gap not in allowed:
        raise ValueError(f"gap_days must be one of {allowed}")
    return gap


def assert_gap(left: pd.DataFrame, right: pd.DataFrame, days: int) -> None:
    if left.empty or right.empty:
        raise InsufficientHistoryError("empty fit, early-stop, or scoring block")
    left_next = times(left).max() + pd.Timedelta(hours=1)
    if left_next + pd.Timedelta(days=days) > times(right).min():
        raise AssertionError(f"required {days}-day purge violated")


def development_and_test(
    frame: pd.DataFrame,
    development_period: Mapping[str, str],
    test_period: Mapping[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = window(
        frame,
        utc(development_period["start"]),
        utc(development_period["end"]) + pd.Timedelta(seconds=1),
    )
    testing = window(
        frame,
        utc(test_period["start"]),
        utc(test_period["end"]) + pd.Timedelta(seconds=1),
    )
    return development, testing


def final_training_calibration_test(
    frame: pd.DataFrame,
    protocol_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the final training prefix, later calibration block, and test."""
    development = protocol_config["development_period"]
    testing = protocol_config["test_period"]
    final = protocol_config["validation"]["final_fit"]
    training = window(
        frame,
        utc(development["start"]),
        utc(final["early_stop_end"], inclusive_date_end=True),
    )
    calibration = window(
        frame,
        utc(final["calibration_start"]),
        utc(final["calibration_end"], inclusive_date_end=True),
    )
    test = window(
        frame,
        utc(testing["start"]),
        utc(testing["end"]) + pd.Timedelta(seconds=1),
    )
    return training, calibration, test


def final_fit_early_stop(
    training: pd.DataFrame,
    protocol_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the final training prefix into purged fit and early-stop blocks."""
    validation = protocol_config["validation"]
    final = validation["final_fit"]
    gap = purge_days(validation)
    early_start = utc(final["early_stop_start"])
    early_end = utc(final["early_stop_end"], inclusive_date_end=True)
    fit_end = early_start - pd.Timedelta(days=gap)
    registered_fit_end = utc(final["fit_end"], inclusive_date_end=True)
    if fit_end != registered_fit_end:
        raise ValueError("final fit_end disagrees with early-stop start minus purge")
    fit = window(training, utc(protocol_config["development_period"]["start"]), fit_end)
    early_stop = window(training, early_start, early_end)
    assert_gap(fit, early_stop, gap)
    return fit, early_stop


def outer_folds(
    frame: pd.DataFrame,
    validation: Mapping[str, Any],
    development_period: Mapping[str, str],
    *,
    gap_days: int | None = None,
) -> list[OuterFold]:
    gap = _validated_gap(validation, gap_days)
    development = window(
        frame,
        utc(development_period["start"]),
        utc(development_period["end"]) + pd.Timedelta(seconds=1),
    )
    folds = []
    for spec in validation["outer_folds"]:
        score_start = utc(spec["validation_start"])
        fit_end = score_start - pd.Timedelta(days=gap)
        fit = window(development, utc(spec["train_start"]), fit_end)
        score = window(
            development,
            score_start,
            utc(spec["validation_end"], inclusive_date_end=True),
        )
        assert_gap(fit, score, gap)
        folds.append(OuterFold(spec["id"], fit, score, fit_end, score_start))
    return folds


def inner_folds(
    outer: OuterFold | pd.DataFrame,
    validation: Mapping[str, Any],
    development_period: Mapping[str, str],
    *,
    n_folds: int | None = None,
    score_days: int | None = None,
    early_stop_days: int | None = None,
    gap_days: int | None = None,
) -> list[InnerFold]:
    """Generate disjoint fit/purge/early-stop/purge/scoring blocks."""
    frame = outer.fit if isinstance(outer, OuterFold) else outer
    n = int(validation["inner_folds"]) if n_folds is None else int(n_folds)
    gap = _validated_gap(validation, gap_days)
    scoring = int(validation["scoring_days"]) if score_days is None else int(score_days)
    stopping = (
        int(validation["early_stop_days"])
        if early_stop_days is None
        else int(early_stop_days)
    )
    if n < 1 or scoring < 1 or stopping < 1:
        raise ValueError("positive n_folds and block durations required")
    target = times(frame)
    if target.empty:
        raise InsufficientHistoryError("empty outer training prefix")
    origin = utc(development_period["start"])
    end = target.max().floor("D") + pd.Timedelta(days=1)
    score_span = pd.Timedelta(days=scoring)
    early_span = pd.Timedelta(days=stopping)
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
                f"{origin.date()}; {n}x{scoring}d scoring + {stopping}d early stop "
                f"+ 2x{gap}d purge cannot fit outer prefix"
            )
        fit = window(frame, origin, fit_end)
        early = window(frame, early_start, early_end)
        score = window(frame, score_start, score_end)
        assert_gap(fit, early, gap)
        assert_gap(early, score, gap)
        folds.append(
            InnerFold(
                f"inner_{index + 1}",
                fit,
                early,
                score,
                fit_end,
                early_start,
                early_end,
                score_start,
            )
        )
    return folds
