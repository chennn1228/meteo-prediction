# -*- coding: utf-8 -*-
"""Pred/obs agreement figure from CURRENT full v1_ml results (QA-styled).

Replaces the stale lite/old-protocol agreement figures. Uses test-year
per-sample predictions (v1_ml full, per_lead) for lgbm; panels:
  a: pred(q0.5) vs obs scatter (daytime) + 1:1 line
  b: residual (pred-obs) vs obs
  c: marginal histograms of obs and pred
Every axis labelled; legend above; filled marks.

Usage: .venv/Scripts/python.exe src/s04_evaluation/analysis/plot_agreement_full.py
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, save_pub, panel_label)
apply_pub_style(font_size=8)

R = CODE_ROOT / "reports" / "03_modeling" / "v1_ml" / "full" / "quantile"

for target in ["ghi", "cloud"]:
    f = R / target / "per_lead_predictions.csv"
    d = pd.read_csv(f)
    d = d[d.model == "lgbm"]
    y = d["y"].values; p = d["q0.5"].values
    day = y > 0 if target == "ghi" else np.ones(len(y), bool)
    yd, pd_ = y[day], p[day]
    mae = np.abs(pd_ - yd).mean(); rmse = np.sqrt(((pd_ - yd)**2).mean())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axes[0]
    ax.scatter(yd, pd_, s=1, alpha=0.12, color=PALETTE["blue_main"], rasterized=True)
    lim = [0, max(yd.max(), pd_.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("observed"); ax.set_ylabel("predicted (q0.5)")
    ax.set_title(f"lgbm {target} | MAE={mae:.1f} RMSE={rmse:.1f}", fontsize=7)
    panel_label(ax, "a")
    ax = axes[1]
    ax.scatter(yd, pd_ - yd, s=1, alpha=0.12, color=PALETTE["red_strong"], rasterized=True)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("observed"); ax.set_ylabel("residual (pred − obs)")
    ax.set_title("residual vs obs", fontsize=7)
    panel_label(ax, "b")
    ax = axes[2]
    ax.hist(yd, bins=50, alpha=0.6, color=PALETTE["red_strong"], edgecolor="none", label="obs")
    ax.hist(pd_, bins=50, alpha=0.6, color=PALETTE["blue_main"], edgecolor="none", label="pred")
    ax.set_xlabel("value"); ax.set_ylabel("count")
    ax.set_title("marginals", fontsize=7)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)
    panel_label(ax, "c")
    fig.tight_layout(pad=0.6)
    save_pub(fig, CODE_ROOT / "figs" / "03_modeling" / "v1_ml" / "full",
             f"fig_v1_ml_{target}_agreement")
    print(f"wrote figs/03_modeling/v1_ml/full/fig_v1_ml_{target}_agreement")
