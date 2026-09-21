"""Fold-local reference adapters for Ridge, LightGBM and XGBoost.

Deep models must supply their own adapter to the same runner contract after
their validated architecture audit. No station/location identity is encoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from s03_features.preprocessing import fit_fold_preprocessing
from s04_splits.rolling import assert_gap
from s04_splits.split_registry import purge_days


def ridge_fit_predict(parameters, fit, early_stop, score, seed, taus):
    """Seven quantiles from fold-fit Ridge plus earlier residual distribution.

    The early-stop block here calibrates residual offsets for this statistical
    baseline (Ridge has no epoch). It never acts as the scoring block.
    """
    del seed
    x_fit, x_early, x_score = fit_fold_preprocessing(fit, early_stop, score)[0]
    estimator = Ridge(alpha=parameters["alpha"])
    estimator.fit(x_fit, fit["y"].to_numpy())
    residual = early_stop["y"].to_numpy() - estimator.predict(x_early)
    offsets = np.quantile(residual, taus)
    point = estimator.predict(x_score)
    return point[:, None] + offsets[None, :], None, None


def tree_fit_predict(algorithm):
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)

    def evaluate(parameters, fit, early_stop, score, seed, taus):
        x_fit, x_early, x_score = fit_fold_preprocessing(
            fit, early_stop, score, seed=seed)[0]
        y_fit, y_early = fit["y"].to_numpy(), early_stop["y"].to_numpy()
        predictions, epochs = [], []
        for tau in taus:
            if algorithm == "lgbm":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=tau, metric="quantile",
                    n_estimators=3000, random_state=seed, verbose=-1, **parameters)
                model.fit(x_fit, y_fit, eval_set=[(x_early, y_early)],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
                predictions.append(model.predict(x_score, num_iteration=model.best_iteration_))
                epochs.append(model.best_iteration_)
            else:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=tau,
                    n_estimators=3000, tree_method="hist", early_stopping_rounds=80,
                    random_state=seed, **parameters)
                model.fit(x_fit, y_fit, eval_set=[(x_early, y_early)], verbose=False)
                predictions.append(model.predict(
                    x_score, iteration_range=(0, model.best_iteration + 1)))
                epochs.append(model.best_iteration + 1)
        # run_trials retains its original (prediction, scalar epoch, memory)
        # callback contract.  The complete seven-quantile training durations
        # live in an explicit audit receipt for leakage-free outer refitting.
        evaluate.receipts.append({
            "algorithm": algorithm, "parameters": dict(parameters), "seed": int(seed),
            "quantiles": [float(tau) for tau in taus],
            "best_rounds": [int(epoch) for epoch in epochs],
            "fit_max_target_utc": pd.to_datetime(fit.target_time_utc, utc=True).max().isoformat(),
            "early_start_target_utc": pd.to_datetime(
                early_stop.target_time_utc, utc=True).min().isoformat(),
            "early_max_target_utc": pd.to_datetime(
                early_stop.target_time_utc, utc=True).max().isoformat(),
            "score_start_target_utc": pd.to_datetime(
                score.target_time_utc, utc=True).min().isoformat(),
            "score_max_target_utc": pd.to_datetime(
                score.target_time_utc, utc=True).max().isoformat(),
        })
        return np.column_stack(predictions), max(epochs), None

    evaluate.receipts = []
    return evaluate


def selected_tree_rounds(algorithm, parameters, seed, taus, receipts,
                         *, expected_inner_folds=3):
    """Select per-quantile median best rounds from complete inner-fold receipts.

    Only receipts for the already-selected candidate and seed are considered.
    This is *not* another search: the median rule is deterministic, and the
    caller must keep every inner score strictly inside its outer fit prefix.
    """
    levels = [float(tau) for tau in taus]
    chosen = [receipt for receipt in receipts
              if receipt.get("algorithm") == algorithm
              and receipt.get("parameters") == dict(parameters)
              and receipt.get("seed") == int(seed)]
    if len(chosen) != expected_inner_folds:
        raise ValueError(f"expected {expected_inner_folds} selected-candidate inner receipts, "
                         f"received {len(chosen)}")
    intervals = []
    rounds = []
    for receipt in chosen:
        if receipt.get("quantiles") != levels:
            raise ValueError("inner receipt quantiles differ from requested levels")
        values = np.asarray(receipt.get("best_rounds"), dtype=float)
        if (values.shape != (len(levels),) or not np.isfinite(values).all()
                or (values <= 0).any() or (values % 1 != 0).any()):
            raise ValueError("inner receipt requires positive integer rounds per quantile")
        fit_max = pd.Timestamp(receipt["fit_max_target_utc"])
        early_start = pd.Timestamp(receipt["early_start_target_utc"])
        early_max = pd.Timestamp(receipt["early_max_target_utc"])
        score_start = pd.Timestamp(receipt["score_start_target_utc"])
        score_max = pd.Timestamp(receipt["score_max_target_utc"])
        if not fit_max < early_start <= early_max < score_start <= score_max:
            raise ValueError("inner receipt has noncausal fit/early/score order")
        intervals.append((score_start, score_max))
        rounds.append(values)
    intervals.sort()
    if any(previous[1] >= following[0]
           for previous, following in zip(intervals, intervals[1:])):
        raise ValueError("inner scoring receipts overlap")
    # Three-fold median is an observed integer and does not need rounding.
    median = np.median(np.stack(rounds), axis=0)
    return tuple(int(np.floor(value + 0.5)) for value in median)


def refit_tree_quantiles(algorithm, parameters, fit, score, seed, taus, receipts,
                         *, gap_days=None):
    """Fit all outer-training rows with fixed inner-selected rounds, then score.

    No outer-validation truth or early-stopping callback reaches the learner.
    The caller may serialize the returned receipt with the prediction output.
    """
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)
    if fit.empty or score.empty:
        raise ValueError("outer fit and score must be nonempty")
    assert_gap(fit, score, purge_days() if gap_days is None else int(gap_days))
    rounds = selected_tree_rounds(algorithm, parameters, seed, taus, receipts)
    outer_fit_max = pd.to_datetime(fit.target_time_utc, utc=True).max()
    if any(pd.Timestamp(receipt["score_max_target_utc"]) > outer_fit_max
           for receipt in receipts if receipt.get("algorithm") == algorithm
           and receipt.get("parameters") == dict(parameters)
           and receipt.get("seed") == int(seed)):
        raise ValueError("inner scoring receipt extends beyond outer fit")
    (x_fit, _, x_score), preprocessing = fit_fold_preprocessing(
        fit, fit.iloc[:0], score, seed=seed)
    y_fit = fit.y.to_numpy(dtype=float)
    predictions = []
    for tau, count in zip(taus, rounds):
        if algorithm == "lgbm":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=tau, metric="quantile",
                n_estimators=count, random_state=seed, verbose=-1, **parameters)
        else:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=tau,
                n_estimators=count, tree_method="hist", random_state=seed,
                **parameters)
        model.fit(x_fit, y_fit)
        predictions.append(model.predict(x_score))
    return np.column_stack(predictions), {
        "algorithm": algorithm, "parameters": dict(parameters), "seed": int(seed),
        "round_selection_rule": "median_of_three_inner_best_rounds_per_quantile",
        "refit_rounds_by_quantile": dict(zip((float(t) for t in taus), rounds)),
        "outer_fit_rows": len(fit), "outer_score_rows": len(score),
        "outer_fit_max_target_utc": outer_fit_max.isoformat(),
        "preprocessing": preprocessing,
    }
