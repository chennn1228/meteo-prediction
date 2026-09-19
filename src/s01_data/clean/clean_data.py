# -*- coding: utf-8 -*-
"""三源 raw JSON → 站点长表（Parquet）+ 质量报告。

长表结构：一行 = 站点 × 目标时刻 × 时效（lead_time）。
列 = 预报端 18 组（_fcst）+ 卫星真值 5（_obs_sat）+ ERA5 真值 5（_obs_era5）
      + ERA5 云量真值 4（_obs）。
清洗规则：云量夹取 [0,100]；辐射负值置 0；缺失保留为 NaN 并计入质量报告。

用法：
  python src/s01_data/02_clean/clean_data.py --site all --start 2024-02-01 --end 2026-01-31
  python src/s01_data/02_clean/clean_data.py --site nanjing --start 2024-02-01 --end 2026-01-31
"""
import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())

# raw 数据源目录（docs/04 目录规范：01_gfs / 02_era5 / 03_satellite）
SOURCE_DIRS = {"previous_runs": "01_gfs", "era5": "02_era5", "satellite": "03_satellite"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clean")

LEADS = [1, 2, 3]

# 原始 API 变量名 -> 长表列名
FCST_RENAME = {
    "shortwave_radiation": "ghi_fcst",
    "diffuse_radiation": "dhi_fcst",
    "direct_normal_irradiance": "dni_fcst",
    "global_tilted_irradiance": "gti_fcst",
    "cloud_cover": "cloud_cover_fcst",
    "cloud_cover_low": "cloud_cover_low_fcst",
    "cloud_cover_mid": "cloud_cover_mid_fcst",
    "cloud_cover_high": "cloud_cover_high_fcst",
    "temperature_2m": "temp_fcst",
    "relative_humidity_2m": "rh_fcst",
    "dewpoint_2m": "dewpoint_fcst",
    "wind_speed_10m": "wind_speed_fcst",
    "wind_direction_10m": "wind_dir_fcst",
    "surface_pressure": "pressure_fcst",
    "precipitation": "precip_fcst",
    "sunshine_duration": "sunshine_fcst",
    "terrestrial_radiation": "terrestrial_fcst",
    "is_day": "is_day_fcst",
}
SAT_RENAME = {
    "shortwave_radiation": "ghi_obs_sat",
    "direct_radiation": "direct_obs_sat",
    "diffuse_radiation": "dhi_obs_sat",
    "direct_normal_irradiance": "dni_obs_sat",
    "global_tilted_irradiance": "gti_obs_sat",
}
ERA5_RENAME = {
    "shortwave_radiation": "ghi_obs_era5",
    "direct_radiation": "direct_obs_era5",
    "diffuse_radiation": "dhi_obs_era5",
    "direct_normal_irradiance": "dni_obs_era5",
    "global_tilted_irradiance": "gti_obs_era5",
}
ERA5_CLOUD_RENAME = {
    "cloud_cover": "cloud_cover_obs",
    "cloud_cover_low": "cloud_cover_low_obs",
    "cloud_cover_mid": "cloud_cover_mid_obs",
    "cloud_cover_high": "cloud_cover_high_obs",
}

RADIATION_COLS = [
    "ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst",
    "ghi_obs_sat", "direct_obs_sat", "dhi_obs_sat", "dni_obs_sat", "gti_obs_sat",
    "ghi_obs_era5", "direct_obs_era5", "dhi_obs_era5", "dni_obs_era5", "gti_obs_era5",
    "terrestrial_fcst", "sunshine_fcst",
]
CLOUD_COLS = [
    "cloud_cover_fcst", "cloud_cover_low_fcst", "cloud_cover_mid_fcst", "cloud_cover_high_fcst",
    "cloud_cover_obs", "cloud_cover_low_obs", "cloud_cover_mid_obs", "cloud_cover_high_obs",
]


def month_files(data_root, source, site, start, end):
    cur = start.replace(day=1)
    while cur <= end:
        p = data_root / "01_raw" / source / site / f"{site}_{cur:%Y-%m}.json"
        if p.exists():
            yield p
        cur = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)


def read_prev_runs(path, cfg):
    payload = json.loads(path.read_text(encoding="utf-8"))
    h = payload["hourly"]
    time = pd.to_datetime(h["time"], utc=True)
    frames = []
    for lead in LEADS:
        df = pd.DataFrame({
            "target_time_utc": time,
            "lead_time": lead * 24,
            "source_grid_latitude": payload.get("latitude"),
            "source_grid_longitude": payload.get("longitude"),
            "source_grid_elevation": payload.get("elevation"),
        })
        for var in cfg["forecast_variables"]:
            new = FCST_RENAME[var]
            df[new] = h[f"{var}_previous_day{lead}"]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["fcst_issue_time_utc"] = out["target_time_utc"] - pd.to_timedelta(out["lead_time"], unit="h")
    return out


def read_simple(path, rename_map):
    h = json.loads(path.read_text(encoding="utf-8"))["hourly"]
    df = pd.DataFrame({"target_time_utc": pd.to_datetime(h["time"], utc=True)})
    for raw, new in rename_map.items():
        df[new] = h[raw]
    return df


def clean(df):
    df = df.copy()
    for col in CLOUD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)
    for col in RADIATION_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0)
    return df


def quality_report(df, site, start, end):
    lines = [
        f"# 数据质量报告：{site}（{start} ~ {end}）",
        "",
        f"- 总行数：{len(df)}（站点 × 目标时刻 × 时效）",
        f"- 时间范围：{df['target_time_utc'].min()} ~ {df['target_time_utc'].max()}",
        f"- 时效：{sorted(df['lead_time'].unique())}",
        "",
        "| 列 | 非空 | 缺失率 | min | max |",
        "|---|---|---|---|---|",
    ]
    for col in df.columns:
        n_null = df[col].isna().sum()
        non_null = df[col].dropna()
        vmin = non_null.min() if len(non_null) else None
        vmax = non_null.max() if len(non_null) else None
        lines.append(
            f"| `{col}` | {len(df) - n_null} | {n_null / len(df):.2%} | {vmin} | {vmax} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="三源 raw JSON 合并为站点长表")
    parser.add_argument("--site", default="all")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else CODE_ROOT / "data"
    cfg_dir = CODE_ROOT / "config"
    sites = yaml.safe_load((cfg_dir / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    site_by_id = {s["id"]: s for s in sites}
    cfg = yaml.safe_load((cfg_dir / "02_variables.yaml").read_text(encoding="utf-8"))
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    site_ids = [s["id"] for s in sites] if args.site == "all" else [args.site]

    out_dir = data_root / "02_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    for site_id in site_ids:
        prev_files = list(month_files(data_root, SOURCE_DIRS["previous_runs"], site_id, start, end))
        sat_files = list(month_files(data_root, SOURCE_DIRS["satellite"], site_id, start, end))
        era5_files = list(month_files(data_root, SOURCE_DIRS["era5"], site_id, start, end))
        if not (prev_files and sat_files and era5_files):
            logger.error("%s 三源文件不齐，跳过", site_id)
            continue

        prev = pd.concat([read_prev_runs(p, cfg) for p in prev_files], ignore_index=True)
        sat = pd.concat([read_simple(p, SAT_RENAME) for p in sat_files], ignore_index=True)
        era5 = pd.concat([read_simple(p, ERA5_RENAME | ERA5_CLOUD_RENAME) for p in era5_files], ignore_index=True)
        df = prev.merge(sat, on="target_time_utc", how="left").merge(era5, on="target_time_utc", how="left")
        df = clean(df)

        meta = json.loads(prev_files[0].read_text(encoding="utf-8"))
        requested = site_by_id[site_id]
        df.insert(0, "station_id", site_id)
        df.insert(1, "lat", requested["lat"])
        df.insert(2, "lon", requested["lon"])
        df.insert(3, "elevation", meta.get("elevation"))
        df.insert(4, "region", requested.get("region"))
        df = df.sort_values(["target_time_utc", "lead_time"]).reset_index(drop=True)

        tag = f"{start:%Y-%m}_{end:%Y-%m}"
        parquet_path = out_dir / f"{site_id}_clean_{tag}.parquet"
        df.to_parquet(parquet_path, index=False)
        report = quality_report(df, site_id, start, end)
        (out_dir / f"quality_report_{site_id}.md").write_text(report, encoding="utf-8")
        logger.info("%s: %d 行 -> %s", site_id, len(df), parquet_path)


if __name__ == "__main__":
    main()
