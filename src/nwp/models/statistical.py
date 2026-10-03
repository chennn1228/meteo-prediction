"""Statistical point and quantile models with fold-local preprocessing."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

from nwp.features.engineering import formal_feature_columns
from nwp.features.preprocessing import (
    FoldCloudImputer, FoldPreprocessor, fit_fold_preprocessing,
    fit_fold_preprocessor)

from .base import BaseModel, ModelError


class StatisticalModel(BaseModel):
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
            target: Any | None = None) -> "StatisticalModel":
        y = target if target is not None else features.get("y")
        if y is None:
            raise ModelError(f"{self.model_id} requires a target")
        columns = formal_feature_columns(features, self.feature_config)
        self.preprocessor = fit_fold_preprocessor(
            features, columns, self.feature_config["policy"])
        if self.implementation == "ridge":
            self.estimator = Ridge(**self.params)
        elif self.implementation == "linear":
            self.estimator = LinearRegression(**self.params)
        else:
            raise ModelError(
                f"unknown statistical implementation: {self.implementation}")
        self.estimator.fit(self.preprocessor.transform(features),
                           np.asarray(y, dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.preprocessor is None or self.estimator is None:
            raise ModelError("model must be fit before predict")
        return np.asarray(
            self.estimator.predict(self.preprocessor.transform(features)),
            dtype=float)


def ridge_fit_predict(parameters: Mapping[str, Any], fit: pd.DataFrame,
                      early_stop: pd.DataFrame, score: pd.DataFrame, seed: int,
                      quantiles: Sequence[float], *,
                      feature_config: Mapping[str, Any]
                      ) -> tuple[np.ndarray, None, None]:
    """Fit Ridge on the fold prefix and calibrate offsets on early-stop rows."""
    del seed
    (x_fit, x_early, x_score), _ = fit_fold_preprocessing(
        fit, early_stop, score, feature_config=feature_config)
    estimator = Ridge(alpha=float(parameters["alpha"]))
    estimator.fit(x_fit, fit["y"].to_numpy(dtype=float))
    residual = (early_stop["y"].to_numpy(dtype=float)
                - estimator.predict(x_early))
    offsets = np.quantile(residual, np.asarray(quantiles, dtype=float))
    point = estimator.predict(x_score)
    return point[:, None] + offsets[None, :], None, None


class RidgeQuantileModel(BaseModel):
    """Outer-fit Ridge point relation plus inner early-stop residual offsets."""

    def __init__(self, model_id: str, feature_config: Mapping[str, Any], alpha: float,
                 quantiles: Sequence[float], seed: int = 0) -> None:
        self.model_id = model_id
        self.feature_config = feature_config
        self.alpha = float(alpha)
        self.quantiles = tuple(float(value) for value in quantiles)
        self.seed = int(seed)
        self.cloud: FoldCloudImputer | None = None
        self.preprocessor: FoldPreprocessor | None = None
        self.estimator: Ridge | None = None
        self.offsets: np.ndarray | None = None

    def fit(self, features: Any,
            target: Any | None = None) -> "RidgeQuantileModel":
        if not isinstance(features, Mapping):
            raise ModelError(
                "Ridge quantile fit requires {'outer_fit', 'inner_folds'}")
        outer_fit = features["outer_fit"]
        folds = features["inner_folds"]
        residuals = []
        for fold in folds:
            (x_fit, x_early, _), _ = fit_fold_preprocessing(
                fold.fit, fold.early_stop, fold.early_stop,
                feature_config=self.feature_config, seed=self.seed)
            estimator = Ridge(alpha=self.alpha).fit(
                x_fit, fold.fit["y"].to_numpy(dtype=float))
            residuals.append(
                fold.early_stop["y"].to_numpy(dtype=float)
                - estimator.predict(x_early))
        pooled = np.concatenate(residuals)
        if not len(pooled) or not np.isfinite(pooled).all():
            raise ModelError("inner early-stop residual pool is empty or nonfinite")
        self.offsets = np.quantile(pooled, self.quantiles)
        policy = self.feature_config["policy"]
        predictors = tuple(policy["cloud_imputation_predictors"])
        self.cloud = FoldCloudImputer(predictors, policy, seed=self.seed).fit(
            outer_fit)
        transformed = self.cloud.transform(outer_fit)
        columns = formal_feature_columns(transformed, self.feature_config)
        self.preprocessor = fit_fold_preprocessor(
            transformed, columns, policy)
        self.estimator = Ridge(alpha=self.alpha).fit(
            self.preprocessor.transform(transformed),
            outer_fit["y"].to_numpy(dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(features, self.quantiles)[:,
                                                               self.quantiles.index(0.5)]

    def predict_quantiles(self, features: pd.DataFrame,
                          quantiles: tuple[float, ...]) -> np.ndarray:
        if quantiles != self.quantiles:
            raise ModelError("requested quantiles differ from fitted Ridge grid")
        if self.cloud is None or self.preprocessor is None or self.estimator is None or self.offsets is None:
            raise ModelError("model must be fit before predict")
        transformed = self.cloud.transform(features)
        point = self.estimator.predict(self.preprocessor.transform(transformed))
        return point[:, None] + self.offsets[None, :]
