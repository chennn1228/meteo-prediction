"""GFS 云量预报缺失的三级插补（正式协议）。

1) 同 station+target_time，用其他时效 cloud_cover_fcst 的中位数；
2) 仍缺失：LightGBM 回归插补（仅训练集非缺失样本训练）；
3) 最后：station+lead+month 中位数。
返回 df（含 cloud_cover_fcst、cloud_fcst_imputed、cloud_fcst_impute_stage）。
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("cloud_impute")


def impute_cloud_forecast(df, feature_cols=None, seed=0, validate=True):
    d = df.copy()
    col = "cloud_cover_fcst"
    if col not in d.columns:
        return d
    d["cloud_fcst_imputed"] = d[col].isna()
    d["cloud_fcst_impute_stage"] = np.where(d[col].isna(), "missing", "raw")

    # stage 1: other leads at same station/time
    same_time = d.groupby(["station_id", "target_time_utc"])[col].transform("median")
    m1 = d[col].isna() & same_time.notna()
    d.loc[m1, col] = same_time[m1]
    d.loc[m1, "cloud_fcst_impute_stage"] = "other_lead_median"

    # stage 2: LightGBM imputation using other numeric features
    miss = d[col].isna()
    if miss.any():
        import lightgbm as lgb
        if feature_cols is None:
            feature_cols = [c for c in d.columns if c not in
                            (col, "cloud_cover_obs", "ghi_obs_sat", "target_time_utc",
                             "station_id", "cloud_fcst_imputed", "cloud_fcst_impute_stage")
                            and pd.api.types.is_numeric_dtype(d[c])]
        Xall = d[feature_cols].copy()
        Xall = Xall.fillna(Xall.median(numeric_only=True))
        known = ~miss
        if known.sum() > 1000:
            model = np.nan
            # 10% artificial mask validation（只在 known 内部）
            if validate and len(Xall) > 5000:
                rng = np.random.default_rng(seed)
                idx = np.flatnonzero(known.to_numpy())
                val_idx = rng.choice(idx, size=max(1000, int(0.1 * len(idx))), replace=False)
                tr = np.ones(len(d), bool)
                tr[val_idx] = False
                tr &= known.to_numpy()
                mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                        num_leaves=31, random_state=seed, verbose=-1)
                mdl.fit(Xall[tr], d.loc[tr, col])
                pred = mdl.predict(Xall[val_idx])
                yv = d.loc[val_idx, col].to_numpy()
                logger.info("cloud impute validation MAE=%.3f RMSE=%.3f",
                            float(np.mean(np.abs(pred - yv))),
                            float(np.sqrt(np.mean((pred - yv) ** 2))))
            model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                      num_leaves=31, random_state=seed, verbose=-1)
            model.fit(Xall[known], d.loc[known, col])
            pred = np.clip(model.predict(Xall[miss]), 0, 100)
            d.loc[miss, col] = pred
            d.loc[miss, "cloud_fcst_impute_stage"] = "lightgbm"

    # stage 3: station+lead+month median fallback
    miss = d[col].isna()
    if miss.any():
        d["_month"] = pd.to_datetime(d["target_time_utc"], utc=True).dt.month
        fallback = d.groupby(["station_id", "lead_time", "_month"])[col].transform("median")
        d.loc[miss, col] = fallback[miss]
        d.loc[miss, "cloud_fcst_impute_stage"] = "station_lead_month_median"
        d = d.drop(columns=["_month"])
    d[col] = d[col].clip(0, 100)
    logger.info("cloud imputed %d rows; stages=%s", int(d["cloud_fcst_imputed"].sum()),
                d["cloud_fcst_impute_stage"].value_counts().to_dict())
    return d
