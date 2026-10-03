"""Tree point adapters and leakage-safe quantile fitting/refitting."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nwp.features.engineering import formal_feature_columns
from nwp.features.preprocessing import FoldPreprocessor, fit_fold_preprocessing, fit_fold_preprocessor
from nwp.features.preprocessing import FoldCloudImputer
from nwp.splits.rolling import assert_gap, purge_days

from .base import BaseModel, ModelError


class TreeModel(BaseModel):
    def __init__(self, model_id: str, implementation: str,
                 feature_config: Mapping[str, Any], *,
                 params: Mapping[str, Any] | None = None) -> None:
        self.model_id = model_id
        self.implementation = implementation
        self.feature_config = feature_config
        self.params = dict(params or {})
        self.preprocessor: FoldPreprocessor | None = None
        self.estimator: Any | None = None

    def fit(self, features: pd.DataFrame,
            target: Any | None = None) -> "TreeModel":
        y = target if target is not None else features.get("y")
        if y is None:
            raise ModelError(f"{self.model_id} requires a target")
        columns = formal_feature_columns(features, self.feature_config)
        self.preprocessor = fit_fold_preprocessor(
            features, columns, self.feature_config["policy"])
        if self.implementation == "lightgbm":
            from lightgbm import LGBMRegressor
            self.estimator = LGBMRegressor(**self.params)
        elif self.implementation == "xgboost":
            from xgboost import XGBRegressor
            self.estimator = XGBRegressor(**self.params)
        else:
            raise ModelError(f"unknown tree implementation: {self.implementation}")
        self.estimator.fit(self.preprocessor.transform(features),
                           np.asarray(y, dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.estimator is None:
            raise ModelError("model must be fit before predict")
        return np.asarray(
            self.estimator.predict(self.preprocessor.transform(features)),
            dtype=float)


class TreeQuantileModel(BaseModel):
    """Seven quantile trees refit on all outer-fit rows at fixed inner rounds."""

    def __init__(self, model_id: str, parameters: Mapping[str, Any],
                 feature_config: Mapping[str, Any], quantiles: Sequence[float],
                 rounds: Sequence[int], seed: int = 0) -> None:
        self.model_id = model_id
        self.parameters = dict(parameters)
        self.feature_config = feature_config
        self.quantiles = tuple(float(value) for value in quantiles)
        self.rounds = tuple(int(value) for value in rounds)
        self.seed = int(seed)
        self.cloud: FoldCloudImputer | None = None
        self.preprocessor: FoldPreprocessor | None = None
        self.estimators: list[Any] = []

    def fit(self, features: pd.DataFrame,
            target: Any | None = None) -> "TreeQuantileModel":
        y = target if target is not None else features.get("y")
        if y is None:
            raise ModelError(f"{self.model_id} requires a target")
        if len(self.rounds) != len(self.quantiles):
            raise ModelError("tree rounds must match the quantile grid")
        policy = self.feature_config["policy"]
        self.cloud = FoldCloudImputer(
            tuple(policy["cloud_imputation_predictors"]), policy,
            seed=self.seed).fit(features)
        transformed = self.cloud.transform(features)
        columns = formal_feature_columns(transformed, self.feature_config)
        self.preprocessor = fit_fold_preprocessor(
            transformed, columns, policy)
        matrix = self.preprocessor.transform(transformed)
        values = np.asarray(y, dtype=float)
        self.estimators = []
        for quantile, count in zip(self.quantiles, self.rounds):
            if self.model_id == "lgbm":
                import lightgbm as lgb
                estimator = lgb.LGBMRegressor(
                    objective="quantile", alpha=quantile, metric="quantile",
                    n_estimators=count, random_state=self.seed, verbose=-1,
                    **self.parameters)
            elif self.model_id == "xgboost":
                import xgboost as xgb
                estimator = xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=quantile,
                    n_estimators=count, tree_method="hist",
                    random_state=self.seed, **self.parameters)
            else:
                raise ModelError(f"unknown quantile tree: {self.model_id}")
            estimator.fit(matrix, values)
            self.estimators.append(estimator)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(features, self.quantiles)[:,
                                                               self.quantiles.index(0.5)]

    def predict_quantiles(self, features: pd.DataFrame,
                          quantiles: tuple[float, ...]) -> np.ndarray:
        if quantiles != self.quantiles:
            raise ModelError("requested quantiles differ from fitted tree grid")
        if self.cloud is None or self.preprocessor is None or not self.estimators:
            raise ModelError("model must be fit before predict")
        matrix = self.preprocessor.transform(self.cloud.transform(features))
        return np.column_stack(
            [estimator.predict(matrix) for estimator in self.estimators])


def final_tree_rounds(
        algorithm: str, parameters: Mapping[str, Any], fit: pd.DataFrame,
        early_stop: pd.DataFrame, seed: int, quantiles: Sequence[float], *,
        feature_config: Mapping[str, Any]) -> tuple[int, ...]:
    """Select final tree rounds only on the declared pre-test early-stop block."""
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)
    fit_max = pd.to_datetime(fit.target_time_utc, utc=True).max()
    early_min = pd.to_datetime(early_stop.target_time_utc, utc=True).min()
    if not fit_max < early_min:
        raise ValueError("final fit must precede the early-stop block")
    (x_fit, x_early, _), _ = fit_fold_preprocessing(
        fit, early_stop, early_stop, feature_config=feature_config, seed=seed)
    y_fit = fit.y.to_numpy(dtype=float)
    y_early = early_stop.y.to_numpy(dtype=float)
    rounds = []
    for quantile in quantiles:
        if algorithm == "lgbm":
            import lightgbm as lgb
            estimator = lgb.LGBMRegressor(
                objective="quantile", alpha=quantile, metric="quantile",
                n_estimators=3000, random_state=seed, verbose=-1,
                **parameters)
            estimator.fit(
                x_fit, y_fit, eval_set=[(x_early, y_early)],
                callbacks=[lgb.early_stopping(80, verbose=False)])
            count = estimator.best_iteration_
        else:
            import xgboost as xgb
            estimator = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=quantile,
                n_estimators=3000, tree_method="hist",
                early_stopping_rounds=80, random_state=seed, **parameters)
            estimator.fit(x_fit, y_fit, eval_set=[(x_early, y_early)],
                          verbose=False)
            count = estimator.best_iteration + 1
        if not isinstance(count, (int, np.integer)) or int(count) < 1:
            raise ModelError("final early stopping returned invalid tree rounds")
        rounds.append(int(count))
    return tuple(rounds)


def tree_fit_predict(algorithm: str, *, feature_config: Mapping[str, Any]):
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)

    def evaluate(parameters, fit, early_stop, score, seed, quantiles):
        (x_fit, x_early, x_score), _ = fit_fold_preprocessing(
            fit, early_stop, score, feature_config=feature_config, seed=seed)
        y_fit = fit["y"].to_numpy(dtype=float)
        y_early = early_stop["y"].to_numpy(dtype=float)
        predictions, epochs = [], []
        for quantile in quantiles:
            if algorithm == "lgbm":
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    objective="quantile", alpha=quantile, metric="quantile",
                    n_estimators=3000, random_state=seed, verbose=-1,
                    **parameters)
                model.fit(x_fit, y_fit, eval_set=[(x_early, y_early)],
                          callbacks=[lgb.early_stopping(80, verbose=False)])
                predictions.append(model.predict(
                    x_score, num_iteration=model.best_iteration_))
                epochs.append(model.best_iteration_)
            else:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    objective="reg:quantileerror", quantile_alpha=quantile,
                    n_estimators=3000, tree_method="hist",
                    early_stopping_rounds=80, random_state=seed, **parameters)
                model.fit(x_fit, y_fit, eval_set=[(x_early, y_early)],
                          verbose=False)
                predictions.append(model.predict(
                    x_score, iteration_range=(0, model.best_iteration + 1)))
                epochs.append(model.best_iteration + 1)
        evaluate.receipts.append({
            "algorithm": algorithm, "parameters": dict(parameters),
            "seed": int(seed),
            "quantiles": [float(value) for value in quantiles],
            "best_rounds": [int(value) for value in epochs],
            "fit_max_target_utc": pd.to_datetime(
                fit.target_time_utc, utc=True).max().isoformat(),
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


def selected_tree_rounds(algorithm: str, parameters: Mapping[str, Any],
                         seed: int, quantiles: Sequence[float],
                         receipts: Sequence[Mapping[str, Any]], *,
                         expected_inner_folds: int = 3) -> tuple[int, ...]:
    levels = [float(value) for value in quantiles]
    chosen = [receipt for receipt in receipts
              if receipt.get("algorithm") == algorithm
              and receipt.get("parameters") == dict(parameters)
              and receipt.get("seed") == int(seed)]
    if len(chosen) != expected_inner_folds:
        raise ValueError(
            f"expected {expected_inner_folds} selected-candidate inner receipts, "
            f"received {len(chosen)}")
    intervals, rounds = [], []
    for receipt in chosen:
        if receipt.get("quantiles") != levels:
            raise ValueError("inner receipt quantiles differ from requested levels")
        values = np.asarray(receipt.get("best_rounds"), dtype=float)
        if (values.shape != (len(levels),) or not np.isfinite(values).all()
                or (values <= 0).any() or (values % 1 != 0).any()):
            raise ValueError(
                "inner receipt requires positive integer rounds per quantile")
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
    median = np.median(np.stack(rounds), axis=0)
    return tuple(int(np.floor(value + 0.5)) for value in median)


def refit_tree_quantiles(
        algorithm: str, parameters: Mapping[str, Any], fit: pd.DataFrame,
        score: pd.DataFrame, seed: int, quantiles: Sequence[float],
        receipts: Sequence[Mapping[str, Any]], *,
        feature_config: Mapping[str, Any],
        validation_config: Mapping[str, Any], gap_days: int | None = None
        ) -> tuple[np.ndarray, dict[str, Any]]:
    if algorithm not in {"lgbm", "xgboost"}:
        raise ValueError(algorithm)
    if fit.empty or score.empty:
        raise ValueError("outer fit and score must be nonempty")
    gap = purge_days(validation_config) if gap_days is None else int(gap_days)
    assert_gap(fit, score, gap)
    rounds = selected_tree_rounds(
        algorithm, parameters, seed, quantiles, receipts)
    outer_fit_max = pd.to_datetime(fit.target_time_utc, utc=True).max()
    if any(pd.Timestamp(receipt["score_max_target_utc"]) > outer_fit_max
           for receipt in receipts
           if receipt.get("algorithm") == algorithm
           and receipt.get("parameters") == dict(parameters)
           and receipt.get("seed") == int(seed)):
        raise ValueError("inner scoring receipt extends beyond outer fit")
    (x_fit, _, x_score), preprocessing = fit_fold_preprocessing(
        fit, fit.iloc[:0], score, feature_config=feature_config, seed=seed)
    y_fit = fit.y.to_numpy(dtype=float)
    predictions = []
    for quantile, count in zip(quantiles, rounds):
        if algorithm == "lgbm":
            import lightgbm as lgb
            model = lgb.LGBMRegressor(
                objective="quantile", alpha=quantile, metric="quantile",
                n_estimators=count, random_state=seed, verbose=-1,
                **parameters)
        else:
            import xgboost as xgb
            model = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=quantile,
                n_estimators=count, tree_method="hist", random_state=seed,
                **parameters)
        model.fit(x_fit, y_fit)
        predictions.append(model.predict(x_score))
    return np.column_stack(predictions), {
        "algorithm": algorithm, "parameters": dict(parameters),
        "seed": int(seed),
        "round_selection_rule":
            "median_of_three_inner_best_rounds_per_quantile",
        "refit_rounds_by_quantile": dict(
            zip((float(value) for value in quantiles), rounds)),
        "outer_fit_rows": len(fit), "outer_score_rows": len(score),
        "outer_fit_max_target_utc": outer_fit_max.isoformat(),
        "preprocessing": preprocessing,
    }
