"""Fixed CPU value-ladder models with issue-time-safe persistence."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from nwp.features.preprocessing import fit_fold_preprocessing

from .base import BaseModel, ModelError


def _hour_month(frame: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_datetime(frame["target_time_utc"], utc=True).dt.tz_convert(
        "Asia/Shanghai"
    )
    return pd.DataFrame(
        {"month": target.dt.month.to_numpy(), "hour": target.dt.hour.to_numpy()},
        index=frame.index,
    )


def _climatology(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    keys_fit = _hour_month(fit)
    keys_score = _hour_month(score)
    means = pd.DataFrame(
        {
            "month": keys_fit.month,
            "hour": keys_fit.hour,
            "y": pd.to_numeric(fit.y, errors="raise"),
        }
    ).groupby(["month", "hour"]).y.mean()
    global_mean = float(fit.y.mean())
    return np.array(
        [
            float(means.get((month, hour), global_mean))
            for month, hour in keys_score.itertuples(index=False, name=None)
        ]
    )


def _causal_history(
    fit: pd.DataFrame,
    early_stop: pd.DataFrame,
    score: pd.DataFrame,
) -> pd.DataFrame:
    """Latest same-service/local-hour truth strictly before each issue."""
    full = pd.concat([fit, early_stop, score], ignore_index=True)
    required = {
        "location_id",
        "target_time_utc",
        "forecast_issue_time_utc",
        "y",
        "ghi_clear_sky",
    }
    if not required <= set(full):
        raise ModelError(f"persistence lacks {sorted(required - set(full))}")
    history = full.loc[
        :, ["location_id", "target_time_utc", "y", "ghi_clear_sky"]
    ].copy()
    history["target_time_utc"] = pd.to_datetime(history.target_time_utc, utc=True)
    history = history.loc[
        np.isfinite(pd.to_numeric(history.y, errors="coerce"))
    ].copy()
    if history.duplicated(["location_id", "target_time_utc"]).any():
        disagree = history.groupby(
            ["location_id", "target_time_utc"], dropna=False
        ).agg(y_unique=("y", "nunique"), clear_unique=("ghi_clear_sky", "nunique"))
        if ((disagree.y_unique > 1) | (disagree.clear_unique > 1)).any():
            raise ModelError(
                "conflicting observations for one service point and valid hour"
            )
        history = history.drop_duplicates(["location_id", "target_time_utc"])
    history["local_hour"] = history.target_time_utc.dt.tz_convert(
        "Asia/Shanghai"
    ).dt.hour
    left = score.loc[
        :, ["location_id", "target_time_utc", "forecast_issue_time_utc"]
    ].copy()
    left["forecast_issue_time_utc"] = pd.to_datetime(
        left.forecast_issue_time_utc, utc=True
    )
    left["target_time_utc"] = pd.to_datetime(left.target_time_utc, utc=True)
    left["local_hour"] = left.target_time_utc.dt.tz_convert("Asia/Shanghai").dt.hour
    left["_order"] = np.arange(len(left))
    joined = pd.merge_asof(
        left.sort_values("forecast_issue_time_utc"),
        history.rename(
            columns={"y": "history_y", "ghi_clear_sky": "history_clear"}
        ).sort_values("target_time_utc"),
        left_on="forecast_issue_time_utc",
        right_on="target_time_utc",
        by=["location_id", "local_hour"],
        direction="backward",
        allow_exact_matches=False,
        suffixes=("", "_history"),
    ).sort_values("_order").reset_index(drop=True)
    used = joined["target_time_utc_history"]
    if (used.notna() & (used >= joined.forecast_issue_time_utc)).any():
        raise ModelError("persistence accessed truth not available by forecast issue")
    return joined


def predict_fixed_cpu(
    model_id: str,
    fit: pd.DataFrame,
    early_stop: pd.DataFrame,
    score: pd.DataFrame,
    *,
    model_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return fixed predictions and transparent fit metadata."""
    entry = model_config["registry"][model_id]
    if entry["tuning"]["enabled"]:
        raise ModelError("fixed model cannot require candidate tuning")
    if fit.empty or score.empty or "y" not in fit:
        raise ModelError("nonempty fit/score and fit truth required")
    if model_id == "raw_gfs":
        return score.ghi_fcst.to_numpy(dtype=float), {
            "definition": "uncorrected_GFS_point"
        }
    if model_id == "climatology":
        return _climatology(fit, score), {
            "definition": "fit_block_month_local_hour_mean"
        }
    if model_id == "bias_correction":
        residual = fit.y.to_numpy(dtype=float) - fit.ghi_fcst.to_numpy(dtype=float)
        adjustment = pd.Series(residual).groupby(fit.lead_time.to_numpy()).mean().to_dict()
        fallback = float(np.mean(residual))
        point = score.ghi_fcst.to_numpy(dtype=float) + np.array(
            [adjustment.get(lead, fallback) for lead in score.lead_time]
        )
        return point, {
            "definition": "fit_block_mean_additive_bias",
            "offset_by_lead": adjustment,
        }
    if model_id == "linear_mos":
        (x_fit, _, x_score), receipt = fit_fold_preprocessing(
            fit, early_stop, score, feature_config=feature_config
        )
        model = LinearRegression().fit(x_fit, fit.y.to_numpy(dtype=float))
        return model.predict(x_score), {
            "definition": "unregularized_linear_all_formal_features",
            "preprocessing": receipt,
        }
    history = _causal_history(fit, early_stop, score)
    fallback = _climatology(fit, score)
    past_y = history.history_y.to_numpy(dtype=float)
    persistence = np.where(np.isfinite(past_y), past_y, fallback)
    if model_id == "persistence":
        return persistence, {
            "definition": "latest_same_local_hour_truth_before_issue"
        }
    historical_clear = history.history_clear.to_numpy(dtype=float)
    current_clear = score.ghi_clear_sky.to_numpy(dtype=float)
    valid_ratio = np.isfinite(past_y) & (historical_clear > 50)
    smart = np.where(
        valid_ratio,
        np.clip(
            past_y / np.where(valid_ratio, historical_clear, np.nan), 0, 1.5
        )
        * current_clear,
        fallback,
    )
    if model_id == "smart_persistence":
        return smart, {
            "definition": "issue_safe_clear_sky_scaled_persistence"
        }
    fit_history = _causal_history(fit, fit.iloc[:0], fit)
    fit_past = fit_history.history_y.to_numpy(dtype=float)
    fit_clear = fit_history.history_clear.to_numpy(dtype=float)
    fit_fallback = _climatology(fit, fit)
    fit_valid = np.isfinite(fit_past) & (fit_clear > 50)
    fit_smart = np.where(
        fit_valid,
        np.clip(fit_past / np.where(fit_valid, fit_clear, np.nan), 0, 1.5)
        * fit.ghi_clear_sky.to_numpy(dtype=float),
        fit_fallback,
    )
    raw = fit.ghi_fcst.to_numpy(dtype=float)
    difference = raw - fit_smart
    denominator = float(np.dot(difference, difference))
    weight = (
        float(
            np.clip(
                np.dot(fit.y.to_numpy(dtype=float) - fit_smart, difference)
                / denominator,
                0,
                1,
            )
        )
        if denominator > 0
        else 0.5
    )
    return weight * score.ghi_fcst.to_numpy(dtype=float) + (1 - weight) * smart, {
        "definition": "fit_block_closed_form_convex_raw_and_smart_persistence",
        "raw_gfs_weight": weight,
    }


class FixedBaselineModel(BaseModel):
    def __init__(
        self,
        model_id: str,
        model_config: Mapping[str, Any],
        feature_config: Mapping[str, Any],
    ) -> None:
        self.model_id = model_id
        self.model_config = model_config
        self.feature_config = feature_config
        self.training_: pd.DataFrame | None = None
        self.history_: pd.DataFrame | None = None

    def fit(
        self, features: Any, target: Any | None = None
    ) -> "FixedBaselineModel":
        if self.model_id == "raw_gfs":
            fit = features["fit"] if isinstance(features, Mapping) else features
            self.training_ = fit.iloc[:0].copy()
            self.history_ = fit.iloc[:0].copy()
            return self
        if isinstance(features, Mapping):
            fit = features["fit"]
            history = features.get("history", fit.iloc[:0])
        else:
            fit, history = features, features.iloc[:0]
        values = target if target is not None else fit.get("y")
        if values is None:
            raise ModelError(f"{self.model_id} requires a target")
        self.training_ = fit.copy()
        self.training_["y"] = np.asarray(values, dtype=float)
        self.history_ = history.copy()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model_id == "raw_gfs":
            if "ghi_fcst" not in features:
                raise ModelError("raw_gfs requires ghi_fcst")
            return features["ghi_fcst"].to_numpy(dtype=float)
        if self.training_ is None:
            raise ModelError("model must be fit before predict")
        score = features.copy()
        if "y" not in score:
            score["y"] = np.nan
        return predict_fixed_cpu(
            self.model_id,
            self.training_,
            self.history_ if self.history_ is not None else self.training_.iloc[:0],
            score,
            model_config=self.model_config,
            feature_config=self.feature_config,
        )[0]


class UnavailableBaseline(BaseModel):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def fit(self, features: Any, target: Any | None = None) -> "UnavailableBaseline":
        raise ModelError(f"{self.model_id} has no validated unified implementation")

    def predict(self, features: Any) -> np.ndarray:
        raise ModelError(f"{self.model_id} has no validated unified implementation")


BASELINE_IMPLEMENTATIONS = {
    name: FixedBaselineModel for name in (
        "climatology", "persistence", "smart_persistence", "optimal_convex",
        "raw_gfs", "bias_correction", "linear",
    )
}
