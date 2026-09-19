"""AutoPV 缺陷 #4 落地：分天气 × 时效 × 逐时 MAE 廓线（有效窗口，11 折逐月预测池）。

数据：reports/06_probability/prototype/quantile_proto/predictions.csv 的中位数预测 q50
      作为订正点预测，与原始 GFS ghi_fcst 对照；天气分类用 featured 自带 sky_type。
输出：
  reports/01_data_audit/features/weather_profiles/{summary.csv,summary.md}
  figs/03_modeling/00_comparison/fig_weather_profiles.{svg,png}
"""
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src"))
from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub, panel_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("plot_weather_profiles")


def main():
    apply_pub_style(font_size=8)
    pred = pd.read_csv(CODE_ROOT / "reports" / "06_probability" / "prototype" / "quantile_proto" / "predictions.csv",
                       parse_dates=["target_time_utc"])
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    cols = ["station_id", "target_time_utc", "lead_time", "ghi_fcst", "sky_type"]
    frames = []
    for s in sites:
        df = pd.read_parquet(CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2019-02_2026-01.parquet",
                             columns=cols)
        df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True)
        frames.append(df)
    meta = pd.concat(frames, ignore_index=True)
    d = pred.merge(meta, on=["station_id", "target_time_utc", "lead_time"], how="left")
    d["local_hour"] = d["target_time_utc"].dt.tz_convert("Asia/Shanghai").dt.hour
    d["err_raw"] = (d["ghi_fcst"] - d["y"]).abs()
    d["err_corr"] = (d["q50"] - d["y"]).abs()
    d = d.dropna(subset=["sky_type"])

    rows = []
    for group, g in d.groupby(["sky_type", "lead_time"]):
        rows.append(dict(sky_type=group[0], lead_time=int(group[1]), n=len(g),
                         mae_raw=g.err_raw.mean(), mae_corr=g.err_corr.mean(),
                         impr=(1 - g.err_corr.mean() / g.err_raw.mean()) * 100))
    res = pd.DataFrame(rows)
    out = CODE_ROOT / "reports" / "01_data_audit" / "features" / "weather_profiles"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "summary.csv", index=False)
    lines = ["# 分天气 × 时效误差（11 折预测池；q50 作订正点预测）\n",
             "| sky_type | lead | n | raw MAE | corr MAE | impr % |",
             "|---|---|---|---|---|---|"]
    for _, r in res.sort_values(["lead_time", "sky_type"]).iterrows():
        lines.append(f"| {r.sky_type} | D+{int(r.lead_time)//24} | {int(r.n)} | "
                     f"{r.mae_raw:.1f} | {r.mae_corr:.1f} | {r.impr:.1f} |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    sky_order = ["clear", "partly", "overcast"] if set(["clear", "partly", "overcast"]) <= set(d.sky_type) \
        else sorted(d.sky_type.unique())
    colors = {s: c for s, c in zip(sky_order, [PALETTE["green_3"], PALETTE["blue_main"], PALETTE["red_strong"]])}
    width = 0.35
    leads = sorted(d.lead_time.unique())
    x = np.arange(len(leads))
    for k, (label, col, field) in enumerate([("raw GFS", PALETTE["neutral_mid"], "err_raw"),
                                             ("corrected", PALETTE["blue_main"], "err_corr")]):
        vals = [d[d.lead_time == L][field].mean() for L in leads]
        axes[0].bar(x + (k - 0.5) * width, vals, width, color=col, label=label,
                    edgecolor="black", lw=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"D+{int(L)//24}" for L in leads])
    axes[0].set_ylabel("MAE (W m$^{-2}$)")
    axes[0].set_title("MAE by lead time (pooled 11 folds)")
    axes[0].legend(fontsize=7)
    panel_label(axes[0], "a")

    for sky in sky_order:
        g = d[d.sky_type == sky]
        axes[1].plot(g.groupby("local_hour").err_corr.mean(), color=colors[sky], lw=1.5,
                     marker="o", ms=3, label=sky)
    axes[1].set_xlabel("local hour (CST)")
    axes[1].set_ylabel("corrected MAE (W m$^{-2}$)")
    axes[1].set_title("Hourly corrected MAE profile by sky type")
    axes[1].legend(fontsize=7, ncol=1)
    panel_label(axes[1], "b")
    fig.set_size_inches(11.2, 3.8)
    fig.suptitle("Fig.6 Error structure", fontsize=10)
    save_pub(fig, CODE_ROOT / "figs" / "03_modeling" / "00_comparison", "fig_weather_profiles")
    logger.info("done -> %s", out)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
