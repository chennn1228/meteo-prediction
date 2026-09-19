# -*- coding: utf-8 -*-
"""数据审计图件（nature-figure Python 风格）。

产出（figs/01_data_audit/）：
  fig_gfs_availability：GFS 预报有效历史热图；
  fig_quantile_proto：分位数原型可靠性/覆盖率/单日示例。

注：33 特征与 PCA95 的共线/消融对照由 collinearity_ablation.py 统一产出
（fig_collinearity_valid），本脚本不再重复出图。
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src"))
from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub, panel_label

apply_pub_style(font_size=8)
FIG = CODE_ROOT / "figs" / "01_data_audit"
R = CODE_ROOT / "reports"

def save(fig, name, w=7.2, h=4.2):
    fig.set_size_inches(w, h)
    return save_pub(fig, FIG, name)


def fig_availability():
    feats = ["ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "cloud_cover_fcst",
             "temp_fcst", "rh_fcst", "dewpoint_fcst", "wind_speed_fcst",
             "pressure_fcst", "precip_fcst", "sunshine_fcst", "terrestrial_fcst", "kt_fcst"]
    lab = {"ghi_fcst": "GHI", "dhi_fcst": "DHI", "dni_fcst": "DNI", "gti_fcst": "GTI",
           "cloud_cover_fcst": "CC", "temp_fcst": "T2m", "rh_fcst": "RH", "dewpoint_fcst": "Dew",
           "wind_speed_fcst": "Wind", "pressure_fcst": "P", "precip_fcst": "Precip",
           "sunshine_fcst": "Sun", "terrestrial_fcst": "Ter", "kt_fcst": "kt"}
    panels = [
        ("data_availability_featured.csv",
         "Full archive 2019-02–2026-01 (legacy evidence)"),
        ("data_availability_featured_2024-02_2026-09.csv",
         "Effective window + holdout 2024-02–2026-09-09"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2))
    for ax, (fn, title) in zip(axes, panels):
        path = R / "01_data_audit" / "diagnostics" / fn
        if not path.exists():
            ax.text(0.5, 0.5, f"missing {fn}", ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=8)
            continue
        agg = pd.read_csv(path).groupby("year_block").mean(numeric_only=True)
        M = agg[feats].to_numpy().T
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(agg.index)))
        ax.set_xticklabels([f"{int(y[2:4])}/{y[5:7]}" for y in agg.index],
                           fontsize=6.3, rotation=35, ha="right")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([lab[f] for f in feats], fontsize=8)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "100" if v > 0.9995 else f"{v*100:.1f}",
                        ha="center", va="center", fontsize=5.2,
                        color="black" if v < 0.65 else "white")
        ax.set_title(title, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("GFS feature")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("non-null fraction")
    fig.suptitle("Fig.1 Data availability", fontsize=10)
    save(fig, "fig_gfs_availability", w=13.2, h=4.5)


def fig_quantile():
    pred = pd.read_csv(R / "06_probability" / "prototype" / "quantile_proto" / "predictions.csv",
                       parse_dates=["target_time_utc"])
    met = pd.read_csv(R / "06_probability" / "prototype" / "quantile_proto" / "metrics.csv")
    taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    emp = []
    for tau in taus:
        emp.append((pred["y"] <= pred[f"q{int(tau*100):02d}"]).mean())
    axes[0].plot(taus, taus, ls="--", color=PALETTE["neutral_mid"], lw=1, label="perfect")
    axes[0].plot(taus, emp, marker="o", ms=4, color=PALETTE["blue_main"], lw=1.6, label="empirical")
    axes[0].set_xlabel("nominal quantile $\\tau$")
    axes[0].set_ylabel("observed fraction $y \\leq q_{\\tau}$")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Reliability (11-fold pooled)")
    panel_label(axes[0], "a")

    cov = met.dropna(subset=["coverage"]).groupby("tau").coverage.mean()
    wid = met.dropna(subset=["coverage"]).groupby("tau").width.mean()
    x = np.arange(3)
    axes[1].bar(x, cov.values, 0.55, color=PALETTE["teal"], edgecolor="black", lw=0.7)
    axes[1].plot(x, x, marker="", ls="")
    for xi, (n, c, wv) in enumerate(zip([0.5, 0.8, 0.9], cov.values, wid.values)):
        axes[1].text(xi, c + 0.015, f"width {wv:.0f}", ha="center", fontsize=7)
        axes[1].plot([xi - 0.22, xi + 0.22], [n, n], ls="--", color=PALETTE["red_strong"], lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["50%", "80%", "90%"])
    axes[1].set_ylim(0, 1.18)
    axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("Central interval coverage (dashed=nominal)")
    panel_label(axes[1], "b")

    p = pred[(pred.station_id == "nanjing_1") & (pred.lead_time == 24)]
    day = p[p.target_time_utc.dt.date.astype(str) == "2024-12-10"].sort_values("target_time_utc")
    if len(day):
        h = (day.target_time_utc.dt.tz_convert("Asia/Shanghai").dt.hour +
             day.target_time_utc.dt.tz_convert("Asia/Shanghai").dt.minute / 60).to_numpy()
        axes[2].fill_between(h, day.q05, day.q95, color=PALETTE["blue_main"], alpha=0.18, label="5–95%")
        axes[2].fill_between(h, day.q25, day.q75, color=PALETTE["blue_main"], alpha=0.35, label="25–75%")
        axes[2].plot(h, day.q50, color=PALETTE["blue_main"], lw=1.5, label="median")
        axes[2].plot(h, day.y, color=PALETTE["red_strong"], lw=1.3, marker="o", ms=2.5,
                     label="Himawari obs")
    axes[2].set_xlabel("local hour (CST)")
    axes[2].set_ylabel("GHI (W m$^{-2}$)")
    axes[2].set_title("Example: Nanjing_1, D+1, 2024-12-10")
    axes[2].legend(fontsize=6.5, loc="upper left")
    panel_label(axes[2], "c")
    save(fig, "fig_quantile_proto", w=12.5, h=3.7)


if __name__ == "__main__":
    fig_availability()
    fig_quantile()
