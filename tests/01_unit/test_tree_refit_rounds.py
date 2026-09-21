"""Tree refits consume all outer fit rows and only inner-selected epochs."""
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s05_tuning import estimators  # noqa: E402


TAUS = (.05, .10, .25, .50, .75, .90, .95)
PARAMETERS = {"learning_rate": .05, "num_leaves": 31}


def receipt(day, rounds, *, parameters=PARAMETERS):
    return {
        "algorithm": "lgbm", "parameters": parameters, "seed": 0,
        "quantiles": list(TAUS), "best_rounds": list(rounds),
        "fit_max_target_utc": f"2024-04-{day - 3:02d}T23:00:00+00:00",
        "early_start_target_utc": f"2024-04-{day - 2:02d}T00:00:00+00:00",
        "early_max_target_utc": f"2024-04-{day - 2:02d}T23:00:00+00:00",
        "score_start_target_utc": f"2024-04-{day:02d}T00:00:00+00:00",
        "score_max_target_utc": f"2024-04-{day:02d}T23:00:00+00:00",
    }


def test_selected_rounds_are_per_quantile_median_and_require_complete_folds():
    receipts = [receipt(10, [10, 30, 5, 8, 9, 11, 12]),
                receipt(12, [20, 10, 7, 8, 9, 11, 12]),
                receipt(14, [30, 20, 9, 8, 9, 11, 12])]
    rounds = estimators.selected_tree_rounds("lgbm", PARAMETERS, 0, TAUS, receipts)
    assert rounds == (20, 20, 7, 8, 9, 11, 12)
    with pytest.raises(ValueError, match="expected 3"):
        estimators.selected_tree_rounds("lgbm", PARAMETERS, 0, TAUS, receipts[:2])
    duplicate = [receipts[0], receipts[0], receipts[2]]
    with pytest.raises(ValueError, match="overlap"):
        estimators.selected_tree_rounds("lgbm", PARAMETERS, 0, TAUS, duplicate)


def test_refit_has_no_outer_validation_truth_or_early_stop(monkeypatch):
    fit = pd.DataFrame({"target_time_utc": pd.date_range(
        "2024-02-01", periods=100 * 24, freq="h", tz="UTC"), "y": 1.0})
    score = pd.DataFrame({"target_time_utc": pd.date_range(
        "2024-06-11", periods=24, freq="h", tz="UTC"), "y": 999999.0})
    monkeypatch.setattr(estimators, "fit_fold_preprocessing",
                        lambda f, e, s, seed=0: ((np.ones((len(f), 1)),
                                                  np.empty((0, 1)),
                                                  np.ones((len(s), 1))),
                                                 {"fit_rows": len(f)}))
    seen = []

    class FakeRegressor:
        def __init__(self, *, n_estimators, alpha, **kwargs):
            self.n_estimators = n_estimators
            self.alpha = alpha

        def fit(self, x, y, **kwargs):
            assert not kwargs  # no eval_set or early-stopping callback
            assert len(y) == len(fit) and np.max(y) == 1.0
            seen.append((self.alpha, self.n_estimators))
            return self

        def predict(self, x):
            return np.full(len(x), self.n_estimators, dtype=float)

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMRegressor=FakeRegressor))
    receipts = [receipt(10, [10] * 7), receipt(12, [20] * 7), receipt(14, [30] * 7)]
    prediction, audit = estimators.refit_tree_quantiles(
        "lgbm", PARAMETERS, fit, score, 0, TAUS, receipts)
    assert prediction.shape == (24, 7)
    assert np.all(prediction == 20)
    assert seen == [(tau, 20) for tau in TAUS]
    assert audit["outer_fit_rows"] == 100 * 24
    assert audit["round_selection_rule"] == "median_of_three_inner_best_rounds_per_quantile"
    score_gap7 = score.copy()
    score_gap7["target_time_utc"] = pd.date_range(
        fit.target_time_utc.max() + pd.Timedelta(hours=1, days=7),
        periods=len(score), freq="h", tz="UTC")
    with pytest.raises(AssertionError, match="10-day purge"):
        estimators.refit_tree_quantiles("lgbm", PARAMETERS, fit, score_gap7,
                                       0, TAUS, receipts)
    sensitivity, _ = estimators.refit_tree_quantiles(
        "lgbm", PARAMETERS, fit, score_gap7, 0, TAUS, receipts, gap_days=7)
    assert sensitivity.shape == prediction.shape


def test_tree_adapter_records_seven_early_stop_rounds(monkeypatch):
    def preprocessed(f, e, s, seed=0):
        return ((np.ones((len(f), 1)), np.ones((len(e), 1)),
                 np.ones((len(s), 1))), {})
    monkeypatch.setattr(estimators, "fit_fold_preprocessing", preprocessed)

    class FakeRegressor:
        def __init__(self, *, alpha, **kwargs):
            self.alpha = alpha
            self.best_iteration_ = int(alpha * 100) + 1

        def fit(self, x, y, **kwargs):
            assert "eval_set" in kwargs
            return self

        def predict(self, x, **kwargs):
            return np.full(len(x), self.alpha)

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(
        LGBMRegressor=FakeRegressor, early_stopping=lambda *a, **k: object()))
    fit = pd.DataFrame({"target_time_utc": pd.date_range("2024-02-01", periods=10,
                                                        freq="h", tz="UTC"), "y": 1.0})
    early = pd.DataFrame({"target_time_utc": pd.date_range("2024-02-12", periods=10,
                                                          freq="h", tz="UTC"), "y": 1.0})
    score = pd.DataFrame({"target_time_utc": pd.date_range("2024-02-23", periods=10,
                                                          freq="h", tz="UTC"), "y": 1.0})
    adapter = estimators.tree_fit_predict("lgbm")
    output, epoch, memory = adapter(PARAMETERS, fit, early, score, 0, TAUS)
    assert output.shape == (10, 7) and memory is None
    assert adapter.receipts[0]["best_rounds"] == [int(t * 100) + 1 for t in TAUS]
    assert epoch == max(adapter.receipts[0]["best_rounds"])
