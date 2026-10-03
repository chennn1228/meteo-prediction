from __future__ import annotations

import pandas as pd
import pytest

from nwp.core.config import load_bundle
from nwp.splits.rolling import InsufficientHistoryError, inner_folds, outer_folds


def test_outer_folds_follow_registered_gap_and_inner_blocks_do_not_overlap():
    bundle = load_bundle()
    frame = pd.DataFrame({"target_time_utc": pd.date_range("2024-02-01", "2025-08-31 23:00", freq="h", tz="UTC")})
    folds = outer_folds(frame, bundle["protocol"]["validation"], bundle["protocol"]["development_period"])
    assert len(folds) == 5
    assert folds[0].fit_end + pd.Timedelta(days=10) == folds[0].score_start
    nested = inner_folds(folds[1], bundle["protocol"]["validation"], bundle["protocol"]["development_period"])
    assert len(nested) == 3
    assert all(fold.early_stop_end <= fold.score_start - pd.Timedelta(days=10) for fold in nested)


def test_first_outer_uses_registered_boundaries_without_a_silent_window_change():
    bundle = load_bundle()
    frame = pd.DataFrame({"target_time_utc": pd.date_range("2024-02-01", "2025-08-31 23:00", freq="h", tz="UTC")})
    first = outer_folds(frame, bundle["protocol"]["validation"], bundle["protocol"]["development_period"])[0]
    nested = inner_folds(first, bundle["protocol"]["validation"], bundle["protocol"]["development_period"])
    assert nested[0].fit_end > pd.Timestamp("2024-02-01", tz="UTC")
