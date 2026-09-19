"""数据可用性诊断：按年块统计每个特征的非空比例（用于核对 GFS 历史覆盖）。

输出：
  reports/01_data_audit/diagnostics/data_availability_featured_<tag>.csv
默认 tag=2019-02_2026-01（历史完整档案证据）；对有效窗口请传：
  --tag 2024-02_2026-08
"""
import logging
import argparse
import os
from pathlib import Path

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("data_availability")

def featured_report(tag="2019-02_2026-01"):
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    feats = ["ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "cloud_cover_fcst", "temp_fcst",
             "rh_fcst", "dewpoint_fcst", "wind_speed_fcst", "pressure_fcst", "precip_fcst",
             "sunshine_fcst", "terrestrial_fcst", "kt_fcst"]
    start_s, end_s = tag.split("_")
    start_y, end_y = int(start_s[:4]), int(end_s[:4])
    bounds = pd.to_datetime([f"{y}-02-01" for y in range(start_y, end_y + 2)], utc=True)
    rows = []
    for s in sites:
        p = CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_{tag}.parquet"
        df = pd.read_parquet(p, columns=["target_time_utc", "ghi_obs_sat"] + feats)
        t = pd.to_datetime(df["target_time_utc"], utc=True)
        yi = np.searchsorted(bounds.to_numpy(), t.to_numpy(), side="right") - 1
        for y, sub in df.assign(yi=yi).groupby("yi"):
            if y < 0 or y >= len(bounds) - 1:
                continue
            row = {"site": s["id"], "year_block": f"{bounds[y]:%Y-%m}~{bounds[y+1]:%Y-%m}",
                   "n": len(sub)}
            for c in feats:
                row[c] = sub[c].notna().mean()
            rows.append(row)
    res = pd.DataFrame(rows)
    out = CODE_ROOT / "reports" / "01_data_audit" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    if tag == "2019-02_2026-01":
        fname = "data_availability_featured.csv"
    else:
        fname = f"data_availability_featured_{tag}.csv"
    res.to_csv(out / fname, index=False)
    agg = res.groupby("year_block")[feats].mean().round(3)
    print(agg.to_string())
    logger.info("done -> %s", out / fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="2019-02_2026-01")
    args = parser.parse_args()
    featured_report(tag=args.tag)
