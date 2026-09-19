"""Pred/obs agreement figure: train/test scatter + marginals + residual.

Protocol: train 2024-02~2025-02, early-stop val 2025-03~2025-08,
test 2025-09~2026-08 (full-year), daytime only.
Model: LightGBM MAE-loss selected config (lite run).
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub, panel_label
import train_v1 as tv
sys.path.insert(0, str(CODE_ROOT / "src" / "s03_models" / "train"))
sys.path.insert(0, str(CODE_ROOT / "src" / "s02_experiment"))  # split_protocol
from split_protocol import default_split

apply_pub_style(font_size=7)
CFG = dict(learning_rate=0.05, num_leaves=31, max_depth=-1,
           subsample=0.9, colsample_bytree=0.9)


def load(target):
    sites = yaml.safe_load((CODE_ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    tv.STATION_CODE = {s["id"]: i for i, s in enumerate(sites)}
    obs = "ghi_obs_sat" if target == "ghi" else "cloud_cover_obs"
    frames = []
    for s in sites:
        p = CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2024-02_2026-09.parquet"
        d = pd.read_parquet(p, columns=tv.FEATURES_NUM + ["station_id", "target_time_utc", obs])
        d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    # GHI 主模型筛白天；云量模型是全时次协议，不能筛白天
    if target == "ghi":
        d = d.query(tv.DAY_FILTER)
    d = d.dropna(subset=[obs]).copy()
    d["season"] = d["target_time_utc"].apply(tv.season_of)
    return d.sort_values("target_time_utc").reset_index(drop=True), obs


def main():
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer

    for target in ("ghi", "cloud"):
        d, obs = load(target)
        tr, va, te = default_split(d)
        imp = SimpleImputer(strategy="median").fit(tv.tree_matrix(tr))
        Xtr, Xva, Xte = (imp.transform(tv.tree_matrix(x)) for x in (tr, va, te))
        m = lgb.LGBMRegressor(objective="regression_l1", metric="mae",
                              n_estimators=3000, random_state=0, verbose=-1, **CFG)
        m.fit(Xtr, tr[obs], eval_set=[(Xva, va[obs])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        tr = tr.assign(pred=m.predict(Xtr))
        te = te.assign(pred=m.predict(Xte))
        if target == "cloud":
            tr["pred"] = tr["pred"].clip(0, 100)
            te["pred"] = te["pred"].clip(0, 100)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7.2),
                                 gridspec_kw={"height_ratios": [3, 1.1]})
        for j, lead in enumerate((24, 48, 72)):
            trj = tr[tr.lead_time == lead].sample(n=min(12000, (tr.lead_time == lead).sum()),
                                                  random_state=0)
            tej = te[te.lead_time == lead]
            ax = axes[0, j]
            lim = [0, max(100 if target == "cloud" else tej[obs].quantile(0.99),
                          tej.pred.quantile(0.99))]
            ax.scatter(trj[obs], trj.pred, s=1, alpha=0.05, color=PALETTE["blue_secondary"],
                       rasterized=True)
            tej_s = tej.sample(n=min(15000, len(tej)), random_state=0)
            ax.scatter(tej_s[obs], tej_s.pred, s=1, alpha=0.06, color=PALETTE["red_strong"],
                       rasterized=True)
            ax.plot(lim, lim, ls="--", lw=1, color=PALETTE["neutral_dark"])
            ax.set_xlim(lim); ax.set_ylim(lim)
            e = tej.pred - tej[obs]
            ax.text(0.04, 0.95, f"MAE={np.abs(e).mean():.2f}\nRMSE={np.sqrt((e**2).mean()):.2f}",
                    transform=ax.transAxes, va="top", fontsize=6.5)
            ax.set_title(f"D+{lead//24}")
            if j == 0:
                ax.set_ylabel("prediction")
            handles = [plt.Line2D([0], [0], marker="o", ls="none", ms=7,
                                  mfc=PALETTE["blue_secondary"], mec="none", label="Train"),
                       plt.Line2D([0], [0], marker="o", ls="none", ms=7,
                                  mfc=PALETTE["red_strong"], mec="none", label="Test")]
            ax.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)
            panel_label(ax, "abc"[j])
            # train/test marginals: top=observed, right=predicted
            for which in ("x", "y"):
                axin = (ax.inset_axes([0.0, 1.02, 1.0, 0.32], sharex=ax) if which == "x"
                        else ax.inset_axes([1.02, 0.0, 0.32, 1.0], sharey=ax))
                for dat, col in ((trj, PALETTE["blue_secondary"]),
                                 (tej, PALETTE["red_strong"])):
                    vv = np.asarray(dat[obs] if which == "x" else dat.pred, float)
                    axin.hist(vv, bins=40, density=True, histtype="step", color=col, lw=1.0)
                    if len(np.unique(vv)) > 3:
                        xs = np.linspace(np.nanmin(vv), np.nanmax(vv), 100)
                        axin.plot(xs, gaussian_kde(vv)(xs), color=col, lw=0.9)
                axin.set_yticks([])
                if which == "x":
                    axin.set_xticks([])
                    axin.spines["bottom"].set_visible(False)
                else:
                    axin.set_xticks([])
                    axin.spines["left"].set_visible(False)
            ax = axes[1, j]
            trj_r = tr[tr.lead_time == lead].sample(n=min(5000, (tr.lead_time == lead).sum()),
                                                    random_state=1)
            ax.scatter(trj_r.pred, trj_r.pred - trj_r[obs], s=0.8, alpha=0.03,
                       color=PALETTE["blue_secondary"], rasterized=True)
            tej_r = tej.sample(n=min(8000, len(tej)), random_state=0)
            ax.scatter(tej_r.pred, tej_r.pred - tej_r[obs], s=0.8, alpha=0.03,
                       color=PALETTE["red_strong"], rasterized=True)
            ax.axhline(0, ls="--", color=PALETTE["neutral_dark"], lw=1)
            ax.set_xlabel("prediction")
            if j == 0:
                ax.set_ylabel("residual")
                ax.legend(handles=[plt.Line2D([0], [0], marker="o", ls="none", ms=5,
                                              mfc=c, mec="none", label=l)
                                   for c, l in ((PALETTE["blue_secondary"], "Train"),
                                                (PALETTE["red_strong"], "Test"))],
                          fontsize=7, loc="lower right", frameon=False)
        fig.tight_layout()
        save_pub(fig, CODE_ROOT / "figs" / "03_modeling" / "v1_ml" / "lite",
                 f"fig_v1_ml_{target}_agreement", dpi=300)


if __name__ == "__main__":
    main()
