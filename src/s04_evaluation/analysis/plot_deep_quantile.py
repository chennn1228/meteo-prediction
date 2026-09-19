"""Deep quantile (v2–v15) summary figures from results.csv."""
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

apply_pub_style(font_size=7)
ROOT = CODE_ROOT / "reports" / "03_modeling"
FIG = CODE_ROOT / "figs" / "06_probability" / "00_comparison"


def main():
    rows = []
    for d in sorted(ROOT.glob("v*_*")):
        for budget in ("full", "lite"):
            for target in ("ghi", "cloud"):
                base = d / budget / "quantile" / target
                p_cal = base / "calibrated" / "results_calibrated.csv"
                p = p_cal if p_cal.exists() else base / "results.csv"
                if not p.exists():
                    continue
                r = pd.read_csv(p)
                a = r[r.group == "all"]
                if not len(a):
                    continue
                rows.append(dict(model=d.name.split("_", 1)[1], target=target,
                                 budget=budget, calibrated=p_cal.exists(),
                                 **a.iloc[0].to_dict()))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "00_summary" / "_deep_quantile_summary.csv", index=False)
    for target in ("ghi", "cloud"):
        g = df[(df.target == target) & (df.budget == "full")].sort_values("mean_pinball")
        if g.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        ax = axes[0]
        ax.barh(np.arange(len(g)), g.mean_pinball, color=PALETTE["blue_main"], edgecolor="black", lw=0.4)
        ax.set_yticks(np.arange(len(g)))
        ax.set_yticklabels(g.model, fontsize=6.5)
        ax.invert_yaxis(); ax.set_xlabel("mean pinball"); ax.set_title("Mean pinball")
        panel_label(ax, "a")
        ax = axes[1]
        for k, col in enumerate(("coverage_50", "coverage_80", "coverage_90")):
            ax.plot(g[col], np.arange(len(g)), marker="o", ms=3, lw=1,
                    color=[PALETTE["blue_secondary"], PALETTE["blue_main"], PALETTE["red_strong"]][k],
                    label=col.replace("coverage_", "") + "%")
        ax.axvline(0.5, ls="--", color=PALETTE["blue_secondary"], lw=0.8)
        ax.axvline(0.8, ls="--", color=PALETTE["blue_main"], lw=0.8)
        ax.axvline(0.9, ls="--", color=PALETTE["red_strong"], lw=0.8)
        ax.set_yticks(np.arange(len(g))); ax.set_yticklabels(g.model, fontsize=6.5)
        ax.invert_yaxis(); ax.set_xlabel("empirical coverage"); ax.set_title("Coverage")
        ax.legend(fontsize=6)
        panel_label(ax, "b")
        ax = axes[2]
        x = np.arange(len(g)); w = 0.38
        ax.bar(x - w/2, g.width_50, w, color=PALETTE["blue_secondary"], label="50%", edgecolor="black", lw=0.4)
        ax.bar(x + w/2, g.width_90, w, color=PALETTE["red_strong"], label="90%", edgecolor="black", lw=0.4)
        ax.set_xticks(x); ax.set_xticklabels(g.model, rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("interval width"); ax.set_title("Sharpness"); ax.legend(fontsize=6)
        panel_label(ax, "c")
        note = "calibrated (conformal)" if bool(g.calibrated.all()) else "uncalibrated"
        fig.suptitle(f"Fig.7 Probabilistic baselines ({target.upper()}, {note})", fontsize=10)
        fig.tight_layout()
        save_pub(fig, FIG / target, "fig_deep_probability")
    print("deep quantile figures", len(df))


if __name__ == "__main__":
    main()
