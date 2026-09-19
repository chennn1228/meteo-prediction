"""Nested purged rolling-origin tuning for Ridge, LightGBM and XGBoost.

The outer folds estimate development-period generalization. Hyperparameters are
selected only on inner folds contained within each outer-training prefix. The
final test year is loaded only to report its size and is never evaluated here.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(ROOT / "src" / "s02_experiment"))
sys.path.insert(0, str(ROOT / "src" / "s03_models" / "train"))

from split_protocol import inner_rolling_folds, nested_rolling_folds, split_test
from cv_month_balanced_quantile import (LGB_GRID, RIDGE_GRID, TAUS, XGB_GRID,
                                        fit_quantile, load, pinball)
import train_v1 as tv


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("cv_nested_rolling_quantile")


def matrices(train, valid):
    from sklearn.impute import SimpleImputer
    x_train_raw, x_valid_raw = tv.tree_matrix(train), tv.tree_matrix(valid)
    imputer = SimpleImputer(strategy="median").fit(x_train_raw)
    return (pd.DataFrame(imputer.transform(x_train_raw), columns=x_train_raw.columns),
            pd.DataFrame(imputer.transform(x_valid_raw), columns=x_train_raw.columns))


def tune_tree(algorithm, grid, inner_folds, target, smoke=False):
    rows = []
    candidates = grid[:1] if smoke else grid
    taus = [0.5] if smoke else TAUS
    for candidate_id, config in enumerate(candidates):
        for fold_id, (train, valid) in enumerate(inner_folds):
            x_train, x_valid = matrices(train, valid)
            y_train, y_valid = train[target].to_numpy(), valid[target].to_numpy()
            losses = [pinball(y_valid, fit_quantile(
                algorithm, tau, config, x_train, y_train, x_valid, y_valid), tau)
                for tau in taus]
            rows.append({"algorithm": algorithm, "candidate": candidate_id,
                         "inner_fold": fold_id, "mean_pinball": float(np.mean(losses))})
    table = pd.DataFrame(rows)
    selected = int(table.groupby("candidate").mean_pinball.mean().idxmin())
    return selected, table


def tune_ridge(inner_folds, target, smoke=False):
    rows = []
    candidates = RIDGE_GRID[:1] if smoke else RIDGE_GRID
    for candidate_id, alpha in enumerate(candidates):
        for fold_id, (train, valid) in enumerate(inner_folds):
            model = tv.lin_pipe(alpha, kind="ridge")
            model.fit(train[tv.FEATURES_NUM + tv.CAT_COLS], train[target].to_numpy())
            prediction = model.predict(valid[tv.FEATURES_NUM + tv.CAT_COLS])
            error = prediction - valid[target].to_numpy()
            rows.append({"algorithm": "ridge", "candidate": candidate_id,
                         "inner_fold": fold_id,
                         "mae": float(np.mean(np.abs(error))),
                         "rmse": float(np.sqrt(np.mean(error ** 2)))})
    table = pd.DataFrame(rows)
    selected = int(table.groupby("candidate").mae.mean().idxmin())
    return selected, table


def fit_predict_tree(algorithm, tau, config, x_train, y_train,
                     x_early, y_early, x_output):
    """Fit with an inner early-stop set and predict a disjoint outer set."""
    if algorithm == "lgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=tau, metric="quantile",
            n_estimators=3000, random_state=0, verbose=-1, **config)
        model.fit(x_train, y_train, eval_set=[(x_early, y_early)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
        return model.predict(x_output, num_iteration=model.best_iteration_)
    import xgboost as xgb
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=tau,
        n_estimators=3000, tree_method="hist", early_stopping_rounds=80,
        random_state=0, **config)
    model.fit(x_train, y_train, eval_set=[(x_early, y_early)], verbose=False)
    return model.predict(x_output, iteration_range=(0, model.best_iteration + 1))


def evaluate_outer_tree(algorithm, config, outer_train, outer_valid, target):
    early_train, early_valid = inner_rolling_folds(outer_train, n_folds=1)[0]
    x_train, x_early = matrices(early_train, early_valid)
    _, x_outer = matrices(early_train, outer_valid)
    y_train = early_train[target].to_numpy()
    y_early = early_valid[target].to_numpy()
    y_outer = outer_valid[target].to_numpy()
    outer_losses = []
    for tau in TAUS:
        prediction = fit_predict_tree(algorithm, tau, config, x_train, y_train,
                                      x_early, y_early, x_outer)
        outer_losses.append(pinball(y_outer, prediction, tau))
    return float(np.mean(outer_losses))


def evaluate_outer_ridge(alpha, outer_train, outer_valid, target):
    model = tv.lin_pipe(alpha, kind="ridge")
    model.fit(outer_train[tv.FEATURES_NUM + tv.CAT_COLS], outer_train[target].to_numpy())
    prediction = model.predict(outer_valid[tv.FEATURES_NUM + tv.CAT_COLS])
    error = prediction - outer_valid[target].to_numpy()
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["ghi", "cloud"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data, target = load(args.target)
    development, final_test = split_test(data)
    outer_folds = nested_rolling_folds(development)
    if args.smoke:
        outer_folds = outer_folds[:1]
    inner_tables, outer_rows = [], []
    for outer_train, outer_valid, spec in outer_folds:
        inner = inner_rolling_folds(outer_train, n_folds=1 if args.smoke else 3)
        for algorithm, grid in (("lgbm", LGB_GRID), ("xgb", XGB_GRID)):
            selected, table = tune_tree(algorithm, grid, inner, target, args.smoke)
            table.insert(0, "outer_fold", spec.fold_id)
            inner_tables.append(table)
            loss = evaluate_outer_tree(algorithm, grid[selected], outer_train, outer_valid, target)
            outer_rows.append({"outer_fold": spec.fold_id, "algorithm": algorithm,
                               "selected_candidate": selected, "mean_pinball": loss,
                               "n_train": len(outer_train), "n_valid": len(outer_valid)})
            LOG.info("%s %s candidate=%d outer pinball=%.4f", spec.fold_id, algorithm, selected, loss)
        selected, table = tune_ridge(inner, target, args.smoke)
        table.insert(0, "outer_fold", spec.fold_id)
        inner_tables.append(table)
        mae, rmse = evaluate_outer_ridge(RIDGE_GRID[selected], outer_train, outer_valid, target)
        outer_rows.append({"outer_fold": spec.fold_id, "algorithm": "ridge",
                           "selected_candidate": selected, "mae": mae, "rmse": rmse,
                           "n_train": len(outer_train), "n_valid": len(outer_valid)})
        LOG.info("%s ridge candidate=%d outer MAE=%.4f", spec.fold_id, selected, mae)
    out = ROOT / "reports" / "02_experiment" / "cv" / ("00_smoke_nested" if args.smoke else "nested_rolling") / args.target
    out.mkdir(parents=True, exist_ok=True)
    inner_results = pd.concat(inner_tables, ignore_index=True)
    inner_results.to_csv(out / "inner_results.csv", index=False)
    outer = pd.DataFrame(outer_rows)
    outer.to_csv(out / "outer_results.csv", index=False)
    selections = {}
    for algorithm, grid in (("lgbm", LGB_GRID), ("xgb", XGB_GRID)):
        group = inner_results[inner_results.algorithm == algorithm]
        chosen = int(group.groupby("candidate").mean_pinball.mean().idxmin())
        selections[algorithm] = grid[chosen]
        selections[f"{algorithm}_encoding"] = "onehot"
    ridge = inner_results[inner_results.algorithm == "ridge"]
    ridge_chosen = int(ridge.groupby("candidate").mae.mean().idxmin())
    selections["ridge"] = {"alpha": RIDGE_GRID[ridge_chosen]}
    selections.update({
        "protocol": "nested_purged_rolling_origin",
        "outer_folds": len(outer_folds),
        "inner_folds": 1 if args.smoke else 3,
        "trials_per_model": 1 if args.smoke else 6,
        "taus": [0.5] if args.smoke else TAUS,
        "final_test_rows_evaluated": 0,
        "final_test_rows_present": len(final_test),
        "smoke": bool(args.smoke),
        "status": "smoke" if args.smoke else "complete",
    })
    (out / "selection.json").write_text(json.dumps(selections, indent=2), encoding="utf-8")
    LOG.info("done -> %s", out)


if __name__ == "__main__":
    main()
