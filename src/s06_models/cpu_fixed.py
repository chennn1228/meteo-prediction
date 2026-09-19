"""Pre-registered CPU point baselines; no six-candidate pseudo-tuning."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from s01_core.config_loader import ProtocolError, load_manifest
from s03_features.preprocessing import fit_fold_preprocessing

FIXED_MODEL_IDS = frozenset({
    "climatology", "persistence", "smart_persistence", "optimal_convex",
    "raw_gfs", "bias_correction", "linear_mos",
})


def _hour_month(frame: pd.DataFrame) -> pd.DataFrame:
    target = pd.to_datetime(frame["target_time_utc"], utc=True).dt.tz_convert("Asia/Shanghai")
    return pd.DataFrame({"month": target.dt.month.to_numpy(),
                         "hour": target.dt.hour.to_numpy()}, index=frame.index)


def _climatology(fit: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    keys_fit = _hour_month(fit)
    keys_score = _hour_month(score)
    means = pd.DataFrame({"month": keys_fit.month, "hour": keys_fit.hour,
                          "y": pd.to_numeric(fit.y, errors="raise")}).groupby(
                              ["month", "hour"]).y.mean()
    global_mean = float(fit.y.mean())
    return np.array([float(means.get((month, hour), global_mean))
                     for month, hour in keys_score.itertuples(index=False, name=None)])


def _causal_history(fit: pd.DataFrame, early_stop: pd.DataFrame,
                    score: pd.DataFrame) -> pd.DataFrame:
    """Latest same-service/hour truth strictly before each forecast issue."""
    full = pd.concat([fit, early_stop, score], ignore_index=True)
    required = {"location_id", "target_time_utc", "forecast_issue_time_utc", "y",
                "ghi_clear_sky"}
    if not required <= set(full):
        raise ProtocolError(f"persistence lacks {sorted(required - set(full))}")
    history = full.loc[:, ["location_id", "target_time_utc", "y", "ghi_clear_sky"]].copy()
    history["target_time_utc"] = pd.to_datetime(history.target_time_utc, utc=True)
    if history.duplicated(["location_id", "target_time_utc"]).any():
        disagree = history.groupby(["location_id", "target_time_utc"], dropna=False).agg(
            y_unique=("y", "nunique"), clear_unique=("ghi_clear_sky", "nunique"))
        if ((disagree.y_unique > 1) | (disagree.clear_unique > 1)).any():
            raise ProtocolError("conflicting observations for one service point and valid hour")
        history = history.drop_duplicates(["location_id", "target_time_utc"])
    history["local_hour"] = history.target_time_utc.dt.tz_convert("Asia/Shanghai").dt.hour
    left = score.loc[:, ["location_id", "target_time_utc", "forecast_issue_time_utc"]].copy()
    left["forecast_issue_time_utc"] = pd.to_datetime(left.forecast_issue_time_utc, utc=True)
    left["target_time_utc"] = pd.to_datetime(left.target_time_utc, utc=True)
    left["local_hour"] = left.target_time_utc.dt.tz_convert("Asia/Shanghai").dt.hour
    left["_order"] = np.arange(len(left))
    joined = pd.merge_asof(
        left.sort_values("forecast_issue_time_utc"),
        history.rename(columns={"y": "history_y", "ghi_clear_sky": "history_clear"})
        .sort_values("target_time_utc"),
        left_on="forecast_issue_time_utc", right_on="target_time_utc",
        by=["location_id", "local_hour"], direction="backward", allow_exact_matches=False,
        suffixes=("", "_history"),
    )
    joined = joined.sort_values("_order").reset_index(drop=True)
    used = joined["target_time_utc_history"]
    if (used.notna() & (used >= joined.forecast_issue_time_utc)).any():
        raise ProtocolError("persistence accessed truth not available by forecast issue")
    return joined


def predict_fixed_cpu(model_id: str, fit: pd.DataFrame, early_stop: pd.DataFrame,
                      score: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Return point forecasts and transparent fit metadata, without selection."""
    if model_id not in FIXED_MODEL_IDS:
        raise ProtocolError(f"not a fixed CPU model: {model_id}")
    item = next(entry for entry in load_manifest()["models"] if entry["id"] == model_id)
    if item["tuning_required"]:
        raise ProtocolError("fixed model cannot require six-candidate tuning")
    if fit.empty or score.empty or "y" not in fit:
        raise ProtocolError("nonempty fit/score and fit truth required")
    if model_id == "raw_gfs":
        return score.ghi_fcst.to_numpy(dtype=float), {"definition": "uncorrected_GFS_point"}
    if model_id == "climatology":
        return _climatology(fit, score), {"definition": "fit_block_month_local_hour_mean"}
    if model_id == "bias_correction":
        residual = fit.y.to_numpy(dtype=float) - fit.ghi_fcst.to_numpy(dtype=float)
        adjustment = pd.Series(residual).groupby(fit.lead_time.to_numpy()).mean().to_dict()
        fallback = float(np.mean(residual))
        point = score.ghi_fcst.to_numpy(dtype=float) + np.array(
            [adjustment.get(lead, fallback) for lead in score.lead_time])
        return point, {"definition": "fit_block_mean_additive_bias", "offset_by_lead": adjustment}
    if model_id == "linear_mos":
        (x_fit, _, x_score), receipt = fit_fold_preprocessing(fit, early_stop, score)
        model = LinearRegression().fit(x_fit, fit.y.to_numpy(dtype=float))
        return model.predict(x_score), {"definition": "unregularized_linear_all_formal_features",
                                          "preprocessing": receipt}
    history = _causal_history(fit, early_stop, score)
    fallback = _climatology(fit, score)
    past_y = history.history_y.to_numpy(dtype=float)
    persistence = np.where(np.isfinite(past_y), past_y, fallback)
    if model_id == "persistence":
        return persistence, {"definition": "latest_same_local_hour_truth_before_issue"}
    historical_clear = history.history_clear.to_numpy(dtype=float)
    current_clear = score.ghi_clear_sky.to_numpy(dtype=float)
    valid_ratio = np.isfinite(past_y) & (historical_clear > 50)
    smart = np.where(valid_ratio,
                     np.clip(past_y / np.where(valid_ratio, historical_clear, np.nan), 0, 1.5)
                     * current_clear, fallback)
    if model_id == "smart_persistence":
        return smart, {"definition": "issue_safe_clear_sky_scaled_persistence"}
    # Closed-form convex fit on the fit block; not a six-candidate hyperparameter search.
    fit_history = _causal_history(fit, fit.iloc[:0], fit)
    fit_past = fit_history.history_y.to_numpy(dtype=float)
    fit_clear = fit_history.history_clear.to_numpy(dtype=float)
    fit_fallback = _climatology(fit, fit)
    fit_valid = np.isfinite(fit_past) & (fit_clear > 50)
    fit_smart = np.where(fit_valid,
                         np.clip(fit_past / np.where(fit_valid, fit_clear, np.nan), 0, 1.5)
                         * fit.ghi_clear_sky.to_numpy(dtype=float), fit_fallback)
    raw = fit.ghi_fcst.to_numpy(dtype=float)
    difference = raw - fit_smart
    denominator = float(np.dot(difference, difference))
    weight = (float(np.clip(np.dot(fit.y.to_numpy(dtype=float) - fit_smart,
                                   difference) / denominator, 0, 1))
              if denominator > 0 else 0.5)
    return weight * score.ghi_fcst.to_numpy(dtype=float) + (1 - weight) * smart, {
        "definition": "fit_block_closed_form_convex_raw_and_smart_persistence",
        "raw_gfs_weight": weight,
    }
