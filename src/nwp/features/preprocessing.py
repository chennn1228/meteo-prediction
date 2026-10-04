"""Preprocessing fitted only on the current training fold."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from nwp.core.schema import ContractError, assert_model_features

from .engineering import formal_feature_columns


def impute_cloud_forecast(*args: Any, **kwargs: Any) -> None:
    """Reject the unavailable whole-dataset imputation entry point."""
    del args, kwargs
    raise RuntimeError(
        "whole-dataset cloud imputation is unavailable; fit FoldCloudImputer "
        "on each fit fold")


@dataclass
class FoldPreprocessor:
    columns: tuple[str, ...]
    policy: Mapping[str, Any]
    medians: pd.Series | None = None
    means: pd.Series | None = None
    scales: pd.Series | None = None
    fit_end_utc: pd.Timestamp | None = None
    train_rows: int = 0

    def fit(
        self,
        training: pd.DataFrame,
        *,
        fit_end_utc: str | pd.Timestamp,
    ) -> "FoldPreprocessor":
        assert_model_features(self.columns, self.policy)
        boundary = pd.Timestamp(fit_end_utc)
        boundary = (
            boundary.tz_localize("UTC")
            if boundary.tzinfo is None
            else boundary.tz_convert("UTC")
        )
        issue = pd.to_datetime(training["forecast_issue_time_utc"], utc=True)
        if training.empty or issue.isna().any() or (issue > boundary).any():
            raise ContractError("preprocessor fit contains later/invalid issue times")
        matrix = training.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        if matrix.notna().sum().eq(0).any():
            raise ContractError("training fold contains all-missing formal feature")
        self.medians = matrix.median()
        imputed = matrix.fillna(self.medians)
        self.means = imputed.mean()
        scale = imputed.std(ddof=0)
        self.scales = scale.mask(scale.eq(0), 1.0)
        self.fit_end_utc = boundary
        self.train_rows = len(training)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None or self.means is None or self.scales is None:
            raise ContractError("preprocessor must be fit on the training fold")
        matrix = frame.loc[:, list(self.columns)].apply(pd.to_numeric, errors="coerce")
        values = (matrix.fillna(self.medians) - self.means) / self.scales
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ContractError("non-finite value after frozen fold preprocessing")
        return values


@dataclass
class FoldCloudImputer:
    """Cloud imputation fitted on the current fit block only."""

    feature_columns: tuple[str, ...]
    policy: Mapping[str, Any]
    seed: int = 0
    model_version: str = "fold-cloud-lgbm-v2"

    def fit(self, fit_rows: pd.DataFrame) -> "FoldCloudImputer":
        assert_model_features(self.feature_columns, self.policy)
        if "cloud_cover_fcst" in self.feature_columns:
            raise ContractError("cloud target cannot predict itself")
        if fit_rows.empty or "cloud_cover_fcst" not in fit_rows:
            raise ContractError("nonempty fit rows with forecast cloud required")
        issue = pd.to_datetime(fit_rows["forecast_issue_time_utc"], utc=True)
        if issue.isna().any():
            raise ContractError("invalid fit issue time for cloud imputer")
        self.fit_start_utc = issue.min()
        self.fit_end_utc = issue.max()
        predictors = fit_rows.loc[:, list(self.feature_columns)].apply(
            pd.to_numeric, errors="coerce"
        )
        self.medians_ = predictors.median()
        if self.medians_.isna().any():
            raise ContractError("cloud imputation predictor entirely missing in fit block")
        cloud = pd.to_numeric(fit_rows["cloud_cover_fcst"], errors="coerce")
        self.global_median_ = float(cloud.median())
        if not np.isfinite(self.global_median_):
            raise ContractError("no observed forecast cloud in fit block")
        self.model_ = None
        known = cloud.notna()
        if int(known.sum()) >= 1000:
            import lightgbm as lgb

            self.model_ = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.seed,
                verbose=-1,
            )
            self.model_.fit(
                predictors.loc[known].fillna(self.medians_), cloud.loc[known]
            )
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise ContractError("fit cloud imputer on fit block first")
        result = rows.copy()
        missing = result["cloud_cover_fcst"].isna()
        method = "lightgbm" if self.model_ is not None else "training_fold_median"
        result["cloud_fcst_imputed"] = missing
        result["cloud_fcst_impute_method"] = np.where(missing, method, "raw")
        if missing.any():
            predictors = result.loc[missing, list(self.feature_columns)].apply(
                pd.to_numeric, errors="coerce"
            )
            values = (
                self.model_.predict(predictors.fillna(self.medians_))
                if self.model_ is not None
                else self.global_median_
            )
            result.loc[missing, "cloud_cover_fcst"] = np.clip(values, 0, 100)
        result.attrs["cloud_imputation"] = {
            "model_version": self.model_version,
            "fit_start_utc": self.fit_start_utc.isoformat(),
            "fit_end_utc": self.fit_end_utc.isoformat(),
            "imputed_fraction": float(missing.mean()),
            "method": method,
        }
        return result


def fit_fold_preprocessor(
    frame: pd.DataFrame,
    columns: Sequence[str],
    policy: Mapping[str, Any],
    *,
    fit_end_utc: str | pd.Timestamp | None = None,
) -> FoldPreprocessor:
    boundary = fit_end_utc
    if boundary is None:
        boundary = pd.to_datetime(frame["forecast_issue_time_utc"], utc=True).max()
    return FoldPreprocessor(tuple(columns), policy).fit(frame, fit_end_utc=boundary)


def fit_fold_preprocessing(
    fit: pd.DataFrame,
    early_stop: pd.DataFrame,
    score: pd.DataFrame,
    *,
    feature_config: Mapping[str, Any],
    seed: int = 0,
) -> tuple[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], dict[str, Any]]:
    """Fit one cloud-imputer and numeric transformer on the fit block only."""
    columns = tuple(formal_feature_columns(fit, feature_config))
    for block in (early_stop, score):
        formal_feature_columns(block, feature_config)
    policy = feature_config["policy"]
    predictors = tuple(policy["cloud_imputation_predictors"])
    cloud = FoldCloudImputer(predictors, policy, seed=seed).fit(fit)
    blocks = tuple(cloud.transform(block) for block in (fit, early_stop, score))
    fit_end = pd.to_datetime(fit["forecast_issue_time_utc"], utc=True).max()
    numeric = FoldPreprocessor(columns, policy).fit(blocks[0], fit_end_utc=fit_end)
    matrices = tuple(numeric.transform(block) for block in blocks)
    return matrices, {
        "formal_feature_columns": columns,
        "cloud_imputation": blocks[0].attrs["cloud_imputation"],
        "preprocessing_fit_end_utc": numeric.fit_end_utc.isoformat(),
        "preprocessing_fit_rows": numeric.train_rows,
        "fold_scope": {
            "fit_start_utc": pd.to_datetime(
                fit["forecast_issue_time_utc"], utc=True).min().isoformat(),
            "fit_end_utc": numeric.fit_end_utc.isoformat(),
        },
    }
