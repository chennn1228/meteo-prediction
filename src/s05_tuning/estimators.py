"""Fold-local reference adapters for Ridge, LightGBM and XGBoost.

Deep models must supply their own adapter to the same runner contract after
their validated architecture audit. No station/location identity is encoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def _shared_features():
    # Retained feature definitions are temporary until the formal feature
    # registry has completed its forecast-issue availability audit.
    import sys
    from pathlib import Path
    root = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
    sys.path.insert(0, str(root / "src" / "s03_models" / "train"))
    import train_v1
    return train_v1


def ridge_fit_predict(parameters, fit, early_stop, score, seed, taus):
    """Seven quantiles from fold-fit Ridge plus earlier residual distribution.

    The early-stop block here calibrates residual offsets for this statistical
    baseline (Ridge has no epoch). It never acts as the scoring block.
    """
    del seed
    tv = _shared_features()
    columns = tv.FEATURES_NUM + tv.CAT_COLS
    estimator = tv.lin_pipe(parameters["alpha"], kind="ridge")
    estimator.fit(fit[columns], fit["y"].to_numpy())
    residual = early_stop["y"].to_numpy() - estimator.predict(early_stop[columns])
    offsets = np.quantile(residual, taus)
    point = estimator.predict(score[columns])
    return point[:, None] + offsets[None, :], None, None


def _tree_matrices(fit, early_stop, score):
    tv = _shared_features()
    raw = [tv.tree_matrix(frame) for frame in (fit, early_stop, score)]
    imputer = SimpleImputer(strategy="median").fit(raw[0])
    return [pd.DataFrame(imputer.transform(x), columns=raw[0].columns,
                         index=x.index) for x in raw]


def tree_fit_predict(algorithm):
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)

    def evaluate(parameters, fit, early_stop, score, seed, taus):
        x_fit, x_early, x_score = _tree_matrices(fit, early_stop, score)
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
