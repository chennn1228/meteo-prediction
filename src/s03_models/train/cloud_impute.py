"""Fold-fitted forecast-cloud imputation, never trained on later rows.

The retired whole-dataset function is deliberately blocked. A forecast issued
at one lead must not borrow another lead's value merely because target times
match: that other run may not yet have been issued.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


IDENTITY_OR_LABEL = frozenset({
    "station_id", "location_id", "source_grid_id", "ghi_obs_sat",
    "cloud_cover_obs", "target_time_utc", "forecast_issue_time_utc",
    "cloud_cover_fcst", "cloud_fcst_imputed", "cloud_fcst_impute_method",
})


@dataclass
class FoldCloudImputer:
    """Fit numeric predictors and fallback statistics solely on a fit block."""

    feature_columns: tuple[str, ...]
    seed: int = 0
    model_version: str = "fold-cloud-lgbm-v1"

    def fit(self, fit_rows: pd.DataFrame) -> "FoldCloudImputer":
        forbidden = set(self.feature_columns) & IDENTITY_OR_LABEL
        if forbidden:
            raise ValueError(f"cloud imputer predictors include identity/label fields: {forbidden}")
        if fit_rows.empty or "cloud_cover_fcst" not in fit_rows:
            raise ValueError("nonempty fit rows with cloud_cover_fcst required")
        self.fit_start_utc = pd.to_datetime(fit_rows["target_time_utc"], utc=True).min()
        self.fit_end_utc = pd.to_datetime(fit_rows["target_time_utc"], utc=True).max()
        self.medians_ = fit_rows[list(self.feature_columns)].median(numeric_only=True)
        self.global_median_ = float(fit_rows["cloud_cover_fcst"].median())
        if not np.isfinite(self.global_median_):
            raise ValueError("fit fold has no observed forecast cloud values")
        known = fit_rows["cloud_cover_fcst"].notna()
        self.model_ = None
        if known.sum() >= 1000:
            import lightgbm as lgb
            x_fit = fit_rows.loc[known, list(self.feature_columns)].fillna(self.medians_)
            self.model_ = lgb.LGBMRegressor(n_estimators=500, learning_rate=.05,
                                            num_leaves=31, random_state=self.seed, verbose=-1)
            self.model_.fit(x_fit, fit_rows.loc[known, "cloud_cover_fcst"])
        return self

    def transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise RuntimeError("fit must precede transform")
        result = rows.copy()
        missing = result["cloud_cover_fcst"].isna()
        result["cloud_fcst_imputed"] = missing
        method = "lightgbm" if self.model_ is not None else "training_fold_median"
        result["cloud_fcst_impute_method"] = np.where(missing, method, "raw")
        if missing.any():
            if self.model_ is None:
                value = self.global_median_
            else:
                value = self.model_.predict(
                    result.loc[missing, list(self.feature_columns)].fillna(self.medians_))
            result.loc[missing, "cloud_cover_fcst"] = np.clip(value, 0, 100)
        result.attrs["cloud_imputation"] = {
            "model_version": self.model_version,
            "fit_start_utc": self.fit_start_utc.isoformat(),
            "fit_end_utc": self.fit_end_utc.isoformat(),
            "imputed_fraction": float(missing.mean()),
            "method": method,
        }
        return result


def impute_cloud_forecast(*args, **kwargs):
    raise RuntimeError(
        "whole-dataset cloud imputation is retired; fit FoldCloudImputer on each fit fold")
