# -*- coding: utf-8 -*-
"""v1 家族的共享定义：特征列表、类别 one-hot、线性管道与指标。

正式训练入口是 train_quantile_v1.py（分位数）与 train_deep.py（v2–v15）；
本模块只提供共享常量/函数，不再包含训练入口。
"""
import os
from pathlib import Path

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DAY_FILTER = "solar_elevation > 10"
STATION_CODE = {}
SEASON_CODE = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
LEADS = [24, 48, 72]

FEATURES_NUM = [
    "ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "cloud_cover_fcst",
    "cloud_cover_low_fcst", "cloud_cover_mid_fcst", "cloud_cover_high_fcst",
    "temp_fcst", "rh_fcst", "dewpoint_fcst", "wind_speed_fcst", "wind_dir_fcst",
    "pressure_fcst", "precip_fcst", "sunshine_fcst", "terrestrial_fcst",
    "solar_elevation", "solar_azimuth", "ghi_clear_sky", "dni_clear_sky",
    "kt_fcst", "kni", "ghi_fcst_minus_clear", "diffuse_fraction",
    "hour_local_sin", "hour_local_cos", "doy_sin", "doy_cos", "month",
    "cloud_cover_change", "cloud_cover_roll1", "cloud_cover_roll3",
    "ghi_fcst_lag1", "ghi_fcst_lag2", "lead_time",
]
CAT_COLS = ["station_id", "season"]


def metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    e = yhat - y
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(e ** 2)
    return {"n": int(len(y)), "mae": float(np.mean(np.abs(e))) if len(y) else np.nan,
            "rmse": float(np.sqrt(np.mean(e ** 2))) if len(y) else np.nan,
            "bias": float(np.mean(e)) if len(y) else np.nan,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan}


def tree_matrix(df):
    x = df[FEATURES_NUM].copy()
    # 类别特征 one-hot：避免整数编码引入虚假序关系。
    st_cat = pd.Categorical(df["station_id"], categories=list(STATION_CODE.keys()))
    se_cat = pd.Categorical(df["season"], categories=list(SEASON_CODE.keys()))
    st = pd.get_dummies(st_cat, prefix="station").astype(float)
    se = pd.get_dummies(se_cat, prefix="season").astype(float)
    # get_dummies(Categorical) 丢弃原索引，必须显式对齐，否则 split 后的非连续索引会错位。
    st.index = df.index
    se.index = df.index
    return pd.concat([x, st, se], axis=1)


def tree_matrix_encoded(df, encoding="onehot", maps=None):
    """树模型输入矩阵。encoding ∈ {onehot, native, target}。

    - onehot：20 站 + 4 季 one-hot（现行默认）；
    - native：station_id/season 保留为 pandas category，交给 LightGBM categorical_feature
      或 XGBoost enable_categorical；
    - target：用 maps（station/season → 训练子集目标均值）做 target encoding。
    maps 为 None 且 encoding=target 时抛错，避免在测试集上计算编码统计。
    """
    if encoding == "onehot":
        return tree_matrix(df), None, None
    x = df[FEATURES_NUM].copy()
    if encoding == "native":
        x["station_id"] = df["station_id"].astype("category")
        x["season"] = df["season"].astype("category")
        return x, ["station_id", "season"], None
    if encoding == "target":
        if maps is None:
            raise ValueError("target encoding 需要提供训练子集统计 maps")
        x["station_te"] = df["station_id"].map(maps["station"]).astype(float)
        x["season_te"] = df["season"].map(maps["season"]).astype(float)
        return x, None, maps
    raise ValueError(f"未知编码: {encoding}")


def target_maps(df, target_col):
    """在给定（训练）子集上计算 station/season 的目标均值编码。"""
    return {"station": df.groupby("station_id")[target_col].mean().to_dict(),
            "season": df.groupby("season")[target_col].mean().to_dict()}


def lin_pipe(alpha, kind="ridge"):
    """线性族管道：kind=ols（最小二乘）/ ridge（L2）/ huber（L1 的光滑代理）。"""
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num, FEATURES_NUM), ("cat", cat, CAT_COLS)])
    if kind == "huber":
        return Pipeline([("pre", pre), ("huber", HuberRegressor(epsilon=1.0, alpha=alpha, max_iter=300))])
    if kind == "ols":
        return Pipeline([("pre", pre), ("ols", LinearRegression())])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


def mean_pinball(y, qmat, taus):
    """七个分位数的平均 pinball 损失（未加权，非 CRPS）。"""
    y = np.asarray(y, float)
    qmat = np.asarray(qmat, float)
    return float(np.mean([np.mean(np.maximum(t * (y - qmat[:, i]),
                                            (t - 1) * (y - qmat[:, i])))
                          for i, t in enumerate(taus)]))


def crps_quantile(y, qmat, taus):
    """分位数网格上的 CRPS 近似（梯形加权，截尾到 [tau_min, tau_max]）。

    CRPS = 2∫ρ_τ dτ；本函数只覆盖给定分位数区间，尾部未建模，
    因此它是可比的相对指标，不是完整 CRPS。完整 CRPS 需全分位函数或集合预报。
    """
    y = np.asarray(y, float)
    qmat = np.asarray(qmat, float)
    taus = np.asarray(taus, float)
    rho = np.array([np.mean(np.maximum(t * (y - qmat[:, i]), (t - 1) * (y - qmat[:, i])))
                    for i, t in enumerate(taus)])
    w = np.zeros(len(taus))
    w[0] = (taus[1] - taus[0]) / 2
    w[-1] = (taus[-1] - taus[-2]) / 2
    w[1:-1] = (taus[2:] - taus[:-2]) / 2
    return float(2.0 * np.sum(w * rho))


def crossing_rate(qmat):
    """相邻分位数逆序的样本比例（输入为未排序的分位数矩阵）。"""
    qmat = np.asarray(qmat, float)
    bad = np.zeros(len(qmat), dtype=bool)
    for j in range(qmat.shape[1] - 1):
        bad |= qmat[:, j] > qmat[:, j + 1]
    return float(np.mean(bad))


def season_of(ts):
    m = ts.month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"
