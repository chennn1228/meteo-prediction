"""v1_ml 分位数正式结果评估与出图（pinball family）。"""
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
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
IN = CODE_ROOT / "reports" / "03_modeling" / "v1_ml" / "full" / "quantile"
FIG = CODE_ROOT / "figs" / "06_probability" / "v1_ml"
MODEL_COLOR = {"raw": PALETTE["neutral_mid"], "bias": PALETTE["teal"],
               "linear": PALETTE["violet"], "ridge": PALETTE["blue_secondary"],
               "lgbm": PALETTE["blue_main"], "xgb": PALETTE["red_strong"]}


def reliability(pred, tau):
    return float(np.mean(pred["y"] <= pred[f"q{tau:g}"]))


def main():
    rows = []
    for target in ("ghi", "cloud"):
        for variant in ("lead_feature", "per_lead"):
            p = IN / target / f"{variant}_predictions.csv"
            if not p.exists():
                continue
            pred = pd.read_csv(p)
            # reliability / pinball figure
            fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
            ax = axes[0, 0]
            ax.plot([0, 1], [0, 1], ls="--", color=PALETTE["neutral_mid"], lw=1)
            for m, g in pred.groupby("model"):
                ax.plot(TAUS, [reliability(g, t) for t in TAUS], marker="o", ms=3,
                        lw=1.1, color=MODEL_COLOR[m], label=m)
            ax.set_xlabel("nominal quantile"); ax.set_ylabel("observed coverage")
            ax.set_title("Reliability"); ax.legend(fontsize=6, ncol=2)
            panel_label(ax, "a")
            ax = axes[0, 1]
            for m, g in pred.groupby("model"):
                vals = [float(np.mean(np.maximum(t * (g.y - g[f"q{t:g}"]),
                                                 (t - 1) * (g.y - g[f"q{t:g}"]))))
                        for t in TAUS]
                ax.plot(TAUS, vals, marker="o", ms=3, lw=1.1, color=MODEL_COLOR[m])
            ax.set_xlabel("quantile"); ax.set_ylabel("pinball loss")
            ax.set_title("Pinball loss"); panel_label(ax, "b")
            axes[0, 2].axis("off")
            for k, (nom, lo, hi) in enumerate(((0.5, 0.25, 0.75), (0.8, 0.10, 0.90),
                                               (0.9, 0.05, 0.95))):
                ax = axes[1, k]
                for m, g in pred.groupby("model"):
                    cov = float(np.mean((g.y >= g[f"q{lo:g}"]) & (g.y <= g[f"q{hi:g}"])))
                    ax.bar([m], [cov], color=MODEL_COLOR[m], width=0.7)
                    rows.append(dict(target=target, variant=variant, model=m,
                                     nominal=nom, coverage=cov))
                ax.axhline(nom, ls="--", color=PALETTE["red_strong"], lw=1)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6)
                ax.set_title(f"{int(nom*100)}% interval")
                ax.set_ylim(0, 1.05)
                panel_label(ax, "cde"[k])
            fig.suptitle(f"Fig.7 Probabilistic verification ({target.upper()}, {variant})",
                         fontsize=10)
            fig.tight_layout()
            save_pub(fig, FIG / target, f"fig_probability_{variant}")
    pd.DataFrame(rows).to_csv(IN / "_coverage_summary.csv", index=False)
    print("quantile figures saved", len(rows))


if __name__ == "__main__":
    main()
