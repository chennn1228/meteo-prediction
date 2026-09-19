"""Fold-local reference adapters for Ridge, LightGBM and XGBoost.

Deep models must supply their own adapter to the same runner contract after
their validated architecture audit. No station/location identity is encoded.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from s03_features.preprocessing import fit_fold_preprocessing


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
        return np.column_stack(predictions), max(epochs), None

    return evaluate
