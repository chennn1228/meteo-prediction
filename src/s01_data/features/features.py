# -*- coding: utf-8 -*-
"""特征工程：清洗长表 -> 建模特征 + 分析专用字段。

输入：data/02_clean/{site}_clean_{tag}.parquet
输出：data/03_featured/{site}_featured_{tag}.parquet

建模特征（A 组，进模型）：
  太阳几何（pvlib）: solar_elevation, solar_azimuth
  晴空基准（pvlib Ineichen，默认气溶胶）: ghi_clear_sky, dni_clear_sky
  衍生: kt_fcst, kni, ghi_fcst_minus_clear, diffuse_fraction
  云场: total/low/mid/high cloud cover + total-cloud change/rolling features
  时序: ghi_fcst_lag1/2
  时间: hour_local_sin/cos, doy_sin/cos, month, season, lead_time(原始)

分析专用（B 组，禁入模型）：
  kt_obs, ghi_err_fcst, sky_type（云量真值列沿用清洗表）

用法：
  python src/s01_data/features/features.py --site all --start 2024-02-01 --end 2026-01-31
"""
import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("features")

SEASON_MAP = {12: "winter", 1: "winter", 2: "winter",
              3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer",
              9: "autumn", 10: "autumn", 11: "autumn"}


def add_solar_and_clear_sky(df, lat, lon):
    """太阳几何 + 晴空基准（pvlib Ineichen，默认 Linke 浑浊度 3.0）。"""
    loc = pvlib.location.Location(latitude=lat, longitude=lon, tz="UTC")
    # pvlib 需要 DatetimeIndex（pandas 3 下传 Series 会报错）
    times = pd.DatetimeIndex(df["target_time_utc"])
    solpos = loc.get_solarposition(times)
    clearsky = loc.get_clearsky(times, model="ineichen")
    df["solar_elevation"] = solpos["apparent_elevation"].to_numpy()
    df["solar_azimuth"] = solpos["azimuth"].to_numpy()
    df["ghi_clear_sky"] = clearsky["ghi"].to_numpy()
    df["dni_clear_sky"] = clearsky["dni"].to_numpy()
    return df


def add_derived_features(df):
    """基于预报值与晴空基准的衍生特征（白天才有意义，夜间为 NaN）。

    分母门槛用 50 W/m2（而非 >0）：日出日落边缘晴空值趋近 0 会让比值爆炸
    （实测 kt 曾达 1.9e5）。再 clip 到物理范围兜底。见 docs/07 N1/N2。
    """
    CLEAR_MIN = 50.0
    clear_ok = df["ghi_clear_sky"] > CLEAR_MIN
    dni_ok = df["dni_clear_sky"] > CLEAR_MIN
    ghi_ok = df["ghi_fcst"] > CLEAR_MIN
    kt = np.where(clear_ok, df["ghi_fcst"] / df["ghi_clear_sky"], np.nan)
    kni = np.where(dni_ok, df["dni_fcst"] / df["dni_clear_sky"], np.nan)
    df["kt_fcst"] = np.clip(kt, 0.0, 1.5)
    df["kni"] = np.clip(kni, 0.0, 1.5)
    df["diffuse_fraction"] = np.clip(
        np.where(ghi_ok, df["dhi_fcst"] / df["ghi_fcst"], np.nan), 0.0, 1.0)
    df["ghi_fcst_minus_clear"] = df["ghi_fcst"] - df["ghi_clear_sky"]
    return df


def add_time_features(df):
    """时间特征：本地小时/年积日的 sin-cos 周期编码 + 月份/季节。"""
    local = df["target_time_utc"].dt.tz_convert("Asia/Shanghai")
    hour = local.dt.hour
    doy = df["target_time_utc"].dt.dayofyear
    df["hour_local_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_local_cos"] = np.cos(2 * np.pi * hour / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = local.dt.month
    df["season"] = df["month"].map(SEASON_MAP)
    return df


def add_sequence_features(df):
    """时序特征：按（站点, 时效）分组，云量差分/滚动、预报 GHI 滞后。"""
    df = df.sort_values(["station_id", "lead_time", "target_time_utc"]).reset_index(drop=True)
    g = df.groupby(["station_id", "lead_time"], group_keys=False)
    df["cloud_cover_change"] = g["cloud_cover_fcst"].diff()
    df["cloud_cover_roll1"] = g["cloud_cover_fcst"].transform(lambda s: s.rolling(2, min_periods=1).mean())
    df["cloud_cover_roll2"] = g["cloud_cover_fcst"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["cloud_cover_roll3"] = g["cloud_cover_fcst"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["ghi_fcst_lag1"] = g["ghi_fcst"].shift(1)
    df["ghi_fcst_lag2"] = g["ghi_fcst"].shift(2)
    return df


def add_analysis_fields(df, kt_bins):
    """分析专用字段：kt_obs、GHI 误差、天气类型（禁入建模特征）。"""
    clear_ok = df["ghi_clear_sky"] > 50.0
    df["kt_obs"] = np.clip(
        np.where(clear_ok, df["ghi_obs_sat"] / df["ghi_clear_sky"], np.nan), 0.0, 1.5)
    df["ghi_err_fcst"] = df["ghi_fcst"] - df["ghi_obs_sat"]
    day = df["solar_elevation"] > 0
    kt = df["kt_obs"]
    b0, b1, b2 = (float(v) for v in kt_bins)
    df["sky_type"] = np.where(day, np.select(
        [kt <= b0, kt <= b1, kt <= b2],
        ["overcast", "partly_cloudy", "partly_clear"],
        default="clear"), "night")
    return df


def main():
    parser = argparse.ArgumentParser(description="特征工程：清洗长表 -> 特征表")
    parser.add_argument("--site", default="all")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else CODE_ROOT / "data"
    cfg_dir = CODE_ROOT / "config"
    sites = yaml.safe_load((cfg_dir / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    tag = f"{start:%Y-%m}_{end:%Y-%m}"
    site_ids = [s["id"] for s in sites] if args.site == "all" else [args.site]

    out_dir = data_root / "03_featured"
    out_dir.mkdir(parents=True, exist_ok=True)

    for site_id in site_ids:
        clean_path = data_root / "02_clean" / f"{site_id}_clean_{tag}.parquet"
        if not clean_path.exists():
            logger.error("缺少清洗表: %s", clean_path)
            sys.exit(1)
        df = pd.read_parquet(clean_path)
        meta = df[["lat", "lon"]].iloc[0]
        df = add_solar_and_clear_sky(df, meta["lat"], meta["lon"])
        df = add_derived_features(df)
        df = add_time_features(df)
        df = add_sequence_features(df)
        df = add_analysis_fields(df, cfg["kt_weather_bins"])
        out_path = out_dir / f"{site_id}_featured_{tag}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("%s: %d 行 %d 列 -> %s", site_id, len(df), len(df.columns), out_path)


if __name__ == "__main__":
    main()
