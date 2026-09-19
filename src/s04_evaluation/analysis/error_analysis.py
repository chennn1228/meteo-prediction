# -*- coding: utf-8 -*-
"""阶段 4：误差规律分析（原始预报 vs 卫星真值）。

分析内容：
  1. 总体指标（MAE/RMSE/Bias/R²）分时效 D+1/2/3；
  2. 分站点总体指标与分站点 × 时效指标，检验站点间差异；
  3. 分季节、分天气类型（sky_type）的 Bias/MAE/RMSE；
  4. 云量误差（fcst-obs）与 GHI 误差（fcst-obs）的关系（分箱 + 相关）；
  5. 输出 CSV 表、PNG 图、summary.md。

口径说明：四项指标均针对主要订正对象 GHI（预报值 vs 卫星真值）计算；
其余变量不参与指标计算，仅用于分组与云量—辐照度耦合分析。

有效白天口径：solar_elevation > 10°（见 README §5.4）。

用法：
  python src/s04_evaluation/analysis/error_analysis.py --site all --start 2024-02-01 --end 2026-01-31
"""
import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analysis")

SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
SKY_ORDER = ["clear", "partly", "overcast"]
DAY_FILTER = "solar_elevation > 10"


def metrics(y, yhat):
    """返回一行指标（n/MAE/RMSE/Bias/R²）。"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[mask], yhat[mask]
    e = yhat - y
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_res = np.sum(e ** 2)
    return {
        "n": len(y),
        "mae": float(np.mean(np.abs(e))) if len(y) else np.nan,
        "rmse": float(np.sqrt(np.mean(e ** 2))) if len(y) else np.nan,
        "bias": float(np.mean(e)) if len(y) else np.nan,  # >0 高估
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
    }


def group_metrics(df, by):
    rows = []
    for key, sub in df.groupby(by, sort=False):
        m = metrics(sub["ghi_obs_sat"], sub["ghi_fcst"])
        if not isinstance(key, tuple):
            key = (key,)
        rows.append({**dict(zip(by, key)), **m})
    return pd.DataFrame(rows)


def md_table(df, index=False):
    """把 DataFrame 转成 Markdown 表格（避免 tabulate 依赖）。"""
    body = df.reset_index() if index else df
    header = [str(c) for c in body.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for _, row in body.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="误差规律分析")
    parser.add_argument("--site", default="all")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else CODE_ROOT / "data"
    out_dir = Path(args.out_dir) if args.out_dir else CODE_ROOT / "reports" / "04_error_analysis" / "error"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    cfg_dir = CODE_ROOT / "config"
    sites = yaml.safe_load((cfg_dir / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    tag = f"{start:%Y-%m}_{end:%Y-%m}"
    site_ids = [s["id"] for s in sites] if args.site == "all" else [args.site]

    frames = []
    for site_id in site_ids:
        p = data_root / "03_featured" / f"{site_id}_featured_{tag}.parquet"
        if not p.exists():
            logger.error("缺少特征表: %s", p)
            sys.exit(1)
        frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.query(DAY_FILTER).copy()
    logger.info("有效白天样本（全部站点合并）: %d 行", len(df))

    # 1) 分时效
    by_lead = group_metrics(df, ["lead_time"]).sort_values("lead_time")
    by_lead.to_csv(out_dir / "error_metrics_by_lead.csv", index=False)
    logger.info("分时效指标:\n%s", by_lead.to_string(index=False))

    # 1b) 分站点（总体与 ×时效）
    by_station = group_metrics(df, ["station_id"]).sort_values("station_id")
    by_station.to_csv(out_dir / "error_metrics_by_station.csv", index=False)
    by_station_lead = group_metrics(df, ["station_id", "lead_time"]).sort_values(["station_id", "lead_time"])
    by_station_lead.to_csv(out_dir / "error_metrics_by_station_lead.csv", index=False)
    logger.info("分站点指标:\n%s", by_station.to_string(index=False))

    # 2) 分季节 / 分天气类型
    by_season = group_metrics(df, ["lead_time", "season"])
    by_season["season"] = pd.Categorical(by_season["season"], SEASON_ORDER)
    by_season = by_season.sort_values(["lead_time", "season"])
    by_season.to_csv(out_dir / "error_metrics_by_season.csv", index=False)

    by_sky = group_metrics(df[df["sky_type"].isin(SKY_ORDER)], ["lead_time", "sky_type"])
    by_sky["sky_type"] = pd.Categorical(by_sky["sky_type"], SKY_ORDER)
    by_sky = by_sky.sort_values(["lead_time", "sky_type"])
    by_sky.to_csv(out_dir / "error_metrics_by_skytype.csv", index=False)

    # 3) 云量误差 vs GHI 误差
    cloud_err = df["cloud_cover_fcst"] - df["cloud_cover_obs"]
    ghi_err = df["ghi_fcst"] - df["ghi_obs_sat"]
    corr_rows = []
    for lead in sorted(df["lead_time"].unique()):
        sub = df[df["lead_time"] == lead]
        ce = sub["cloud_cover_fcst"] - sub["cloud_cover_obs"]
        ge = sub["ghi_fcst"] - sub["ghi_obs_sat"]
        ok = np.isfinite(ce) & np.isfinite(ge)
        r_p = stats.pearsonr(ce[ok], ge[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        r_s = stats.spearmanr(ce[ok], ge[ok]) if ok.sum() > 2 else (np.nan, np.nan)
        corr_rows.append({"lead_time": lead, "pearson_r": r_p[0], "pearson_p": r_p[1],
                          "spearman_r": r_s[0], "spearman_p": r_s[1], "n": int(ok.sum())})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "cloud_ghi_correlation.csv", index=False)

    bins = pd.cut(cloud_err, bins=12)
    binned = (pd.DataFrame({"cloud_err": cloud_err, "ghi_err": ghi_err})
              .groupby(bins, observed=True)["ghi_err"].agg(["mean", "count", "median"]))
    binned.to_csv(out_dir / "cloud_ghi_bins.csv")

    # 4) 图
    x = by_lead["lead_time"]
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, by_lead["mae"], width=w, label="MAE")
    ax.bar(x + w / 2, by_lead["rmse"], width=w, label="RMSE")
    ax.set_xticks(x)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("W/m2")
    ax.set_title("Raw forecast error by lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_error_by_lead.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for season in SEASON_ORDER:
        sub = by_season[by_season["season"] == season]
        ax.plot(sub["lead_time"], sub["bias"], marker="o", label=season)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("Bias (W/m2)")
    ax.set_title("Bias by season and lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_bias_by_season.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for sky in SKY_ORDER:
        sub = by_sky[by_sky["sky_type"] == sky]
        ax.plot(sub["lead_time"], sub["bias"], marker="o", label=sky)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("Bias (W/m2)")
    ax.set_title("Bias by sky type and lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_bias_by_skytype.png", dpi=150)
    plt.close(fig)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(cloud_err), size=min(20000, len(cloud_err)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(cloud_err.iloc[idx], ghi_err.iloc[idx], s=4, alpha=0.25)
    ax.scatter(binned.index.categories.mid, binned["mean"], color="red", s=40, label="bin mean")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("cloud cover error (fcst - obs, %)")
    ax.set_ylabel("GHI error (fcst - obs, W/m2)")
    ax.set_title("Cloud cover error vs GHI error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_cloud_vs_ghi_error.png", dpi=150)
    plt.close(fig)

    # 5) summary.md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("# 误差规律分析摘要\n\n")
        fh.write(f"- 样本：{', '.join(site_ids)}，{start} ~ {end}，有效白天（高度角>10°）共 {len(df)} 行\n")
        fh.write("- 口径说明：四项指标均针对主要订正对象 GHI（预报值对比卫星真值）计算；其余变量不参与指标计算，仅用于分组与云量—辐照度耦合分析。\n\n")
        fh.write("## 分时效（原始预报）\n\n")
        fh.write(md_table(by_lead) + "\n\n")
        fh.write("## 分站点（总体）\n\n")
        fh.write(md_table(by_station) + "\n\n")
        fh.write("## 分站点 × 时效\n\n")
        fh.write(md_table(by_station_lead) + "\n\n")
        fh.write("## 云量误差—GHI 误差相关\n\n")
        fh.write(md_table(corr) + "\n\n")
        fh.write("## 分箱（云量误差 → GHI 误差均值）\n\n")
        fh.write(md_table(binned, index=True) + "\n")

    logger.info("输出目录: %s", out_dir)


if __name__ == "__main__":
    main()