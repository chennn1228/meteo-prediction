"""No full training: split and selection invariants on synthetic time series."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s04_splits.diagnostics import first_outer_sample_report  # noqa: E402
from s04_splits.rolling import (InsufficientHistoryError, inner_folds,
                                outer_folds, times)  # noqa: E402
from s05_tuning.runner import IncompleteTrialsError, run_trials  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402


@pytest.fixture(scope="module")
def hourly():
    t = pd.date_range("2024-02-01", "2025-09-01", freq="h", tz="UTC", inclusive="left")
    return pd.DataFrame({"target_time_utc": t, "station_id": "S1", "lead_time": 24,
                         "solar_elevation": np.where((t.hour >= 6) & (t.hour <= 18), 1., -1.),
                         "y": np.sin(np.arange(len(t)) / 24)})


def test_gap_sensitivity_keeps_scoring_fixed(hourly):
    f7, f10, f14 = (outer_folds(hourly, gap_days=g)[0] for g in (7, 10, 14))
    assert len(f7.fit) > len(f10.fit) > len(f14.fit)
    assert times(f7.score).min() == times(f10.score).min() == times(f14.score).min()
    assert times(f7.fit).max() < times(f7.score).min()


def test_gap_sensitivity_inner_selector_accepts_registered_variant(hourly):
    outer = outer_folds(hourly, gap_days=7)[0]
    folds = inner_folds(outer, gap_days=7)
    chosen, ledger = run_trials(
        "ridge_mos", "outer_1_gap7", folds,
        lambda parameters, fit, early, score, seed, taus:
        (np.repeat(score.y.to_numpy()[:, None], len(taus), axis=1), None, None),
        smoke=True, gap_days=7)
    assert chosen == -1 and len(ledger) == 1 and ledger[0].status == "ok"
    with pytest.raises(ValueError, match="preregistered sensitivity"):
        run_trials("ridge_mos", "outer_1_gap5", folds,
                   lambda *args: None, smoke=True, gap_days=5)


def test_first_outer_keeps_three_independent_purged_inner_folds(hourly):
    outer = outer_folds(hourly)[0]
    folds = inner_folds(outer)
    assert len(folds) == 3
    assert len(folds[0].fit) == 45 * 24
    assert all(len(fold.score) == 14 * 24 for fold in folds)
    for fold in folds:
        assert times(fold.fit).max() < times(fold.early_stop).min()
        assert times(fold.early_stop).max() < times(fold.score).min()
    report = first_outer_sample_report(hourly)
    assert len(report) == 3
    assert report.protocol_feasible.all()
    assert {"raw_rows", "daytime_rows", "sequence_168_valid", "fit_rows",
            "early_stop_rows", "scoring_rows"}.issubset(report.columns)


def test_inner_early_stop_is_disjoint_from_score(hourly):
    long_prefix = hourly[hourly.target_time_utc < "2024-09-01"].copy()
    folds = inner_folds(long_prefix)
    assert len(folds) == 3
    for fold in folds:
        assert times(fold.fit).max() < times(fold.early_stop).min()
        assert times(fold.early_stop).max() < times(fold.score).min()
        assert len(fold.fit) > 0 and len(fold.early_stop) > 0 and len(fold.score) > 0


def test_six_candidates_and_seven_quantile_selector(hourly):
    folds = inner_folds(hourly[hourly.target_time_utc < "2024-09-01"].copy())
    assert len(candidates("ridge_mos")) == len(candidates("lgbm")) == 6

    def fit_predict(params, fit, early, score, seed, taus):
        assert seed == 0 and len(taus) == 7
        assert times(fit).max() < times(early).min() < times(score).min()
        # Candidate alpha=1 gives the best offset on this synthetic target.
        offset = abs(np.log10(params["alpha"]))
        return np.repeat((score.y.to_numpy() + offset)[:, None], 7, axis=1), None, None

    chosen, ledger = run_trials("ridge_mos", "outer_synthetic", folds, fit_predict)
    assert chosen == 3
    assert len(ledger) == 18
    assert all(row.status == "ok" and row.mean_pinball is not None for row in ledger)


def test_smoke_never_selects_candidate(hourly):
    folds = inner_folds(hourly[hourly.target_time_utc < "2024-09-01"].copy())
    chosen, ledger = run_trials(
        "ridge_mos", "synthetic", folds,
        lambda p, fit, early, score, seed, taus:
        (np.repeat(score.y.to_numpy()[:, None], 7, axis=1), None, None),
        smoke=True)
    assert chosen == -1 and len(ledger) == 1


def test_failed_candidate_blocks_fair_selection(hourly):
    folds = inner_folds(hourly[hourly.target_time_utc < "2024-09-01"].copy())

    def fit_predict(params, fit, early, score, seed, taus):
        if params["alpha"] == 1e-5:
            raise RuntimeError("synthetic candidate failure")
        return np.repeat(score.y.to_numpy()[:, None], 7, axis=1), None, None

    with pytest.raises(IncompleteTrialsError) as caught:
        run_trials("ridge_mos", "outer_synthetic", folds, fit_predict)
    assert len(caught.value.ledger) == 18
    assert sum(row.status == "error" for row in caught.value.ledger) == 3
