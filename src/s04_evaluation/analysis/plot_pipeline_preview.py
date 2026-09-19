# -*- coding: utf-8 -*-
"""FULL-PIPELINE figure PREVIEW (QA-compliant styling).

Style rules enforced here (and in all project figures):
  - legends placed ABOVE axes (bbox_to_anchor) so they never cover data;
  - every axes has BOTH xlabel and ylabel;
  - categorical x tick labels rotated 45 deg, ha=right, rotation_mode=anchor;
  - bars/hists FILLED (no hollow step outlines); box/violin facecolor alpha>=0.7;
  - panel labels a/b/c via panel_label.

Numbers are from lite/smoke data => watermarked PREVIEW, NOT final.

Usage: .venv/Scripts/python.exe src/s04_evaluation/analysis/plot_pipeline_preview.py
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
apply_pub_style(font_size=8)

TAU = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QCOLS = [f"q{t:g}" for t in TAU]
OUT = CODE_ROOT / "figs" / "00_preview"
SMOKE = CODE_ROOT / "reports" / "03_modeling" / "00_smoke"

def watermark(fig, txt="PREVIEW — lite/smoke data, numbers NOT final"):
    fig.text(0.99, 0.005, txt, ha="right", va="bottom", fontsize=6,
             color=PALETTE["neutral_mid"], style="italic")

def leg_above(ax, ncol=2, fs=6, y=1.08):
    ax.legend(fontsize=fs, loc="lower center", bbox_to_anchor=(0.5, y),
              ncol=ncol, frameon=False)

def xcat(ax, labels, fs=6):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=fs)

def pinball(y, q, tau):
    e = y - q
    return np.maximum(tau * e, (tau - 1) * e).mean()
def mean_pinball(df):
    return np.mean([pinball(df["y"].values, df[c].values, t) for c, t in zip(QCOLS, TAU)])
def coverage(df, lo, hi):
    return ((df["y"] >= df[lo]) & (df["y"] <= df[hi])).mean()
def width(df, lo, hi):
    return (df[hi] - df[lo]).mean()
def reliability(df):
    return np.array([(t, (df["y"] <= df[c]).mean()) for c, t in zip(QCOLS, TAU)])
def pit(df):
    y = df["y"].values; q = df[QCOLS].values
    return (q < y[:, None]).sum(axis=1) / len(TAU)

deep_models = ["v2_mlp", "v7_autoformer", "v8_informer", "v9_fedformer",
               "v11_patchtst", "v13_timesnet", "v15_pinn"]
preds = {}
for m in deep_models:
    f = SMOKE / m / "quantile" / "ghi" / "test_predictions.csv"
    if f.exists():
        d = pd.read_csv(f)
        d["target_time_utc"] = pd.to_datetime(d["target_time_utc"], utc=True)
        preds[m] = d
print(f"loaded deep smoke models: {list(preds)}")

v1 = pd.read_csv(CODE_ROOT / "reports/03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv")
feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT / "data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))], ignore_index=True)
feat["target_time_utc"] = pd.to_datetime(feat["target_time_utc"])
sky = feat[["station_id", "target_time_utc", "lead_time", "sky_type", "solar_elevation"]].drop_duplicates()

# =============== FIG 5 preview ===============
print("Fig.5 preview...")
rows = []
for m, d in preds.items():
    for lt in [24, 48, 72]:
        sub = d[d["lead_time"] == lt]
        rows.append(dict(model=m, lead=lt, mp=mean_pinball(sub),
                         cov90=coverage(sub, "q0.05", "q0.95"),
                         w90=width(sub, "q0.05", "q0.95"),
                         rmse=np.sqrt(((sub["q0.5"] - sub["y"])**2).mean())))
leadmap = {"D+1": 24, "D+2": 48, "D+3": 72}
for vm in ["lgbm", "xgb"]:
    for u, lt in leadmap.items():
        sub = v1[(v1.model == vm) & (v1.unit == u)]
        if len(sub) == 0: continue
        rows.append(dict(model=f"v1_{vm}", lead=lt, mp=mean_pinball(sub),
                         cov90=coverage(sub, "q0.05", "q0.95"),
                         w90=width(sub, "q0.05", "q0.95"),
                         rmse=np.sqrt(((sub["q0.5"] - sub["y"])**2).mean())))
cmp_df = pd.DataFrame(rows)
model_order = ["v1_lgbm", "v1_xgb"] + deep_models
short = {"v1_lgbm": "LGBM", "v1_xgb": "XGB", "v2_mlp": "MLP", "v7_autoformer": "AutoF",
         "v8_informer": "InfoF", "v9_fedformer": "FEDF", "v11_patchtst": "PatchT",
         "v13_timesnet": "TimeN", "v15_pinn": "PINN"}
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))
x = np.arange(len(model_order)); w = 0.26
for ax, (metric, ylab, nom) in zip(axes, [
        ("mp", "mean pinball", None), ("cov90", "coverage 90%", 0.90),
        ("w90", "width 90%", None), ("rmse", "RMSE (W m$^{-2}$)", None)]):
    for i, lt in enumerate([24, 48, 72]):
        vals = []
        for m in model_order:
            sel = cmp_df.loc[(cmp_df.model == m) & (cmp_df.lead == lt), metric]
            v = sel.mean() if len(sel) else np.nan
            vals.append(0.0 if np.isnan(v) else v)
        ax.bar(x + i*w, vals, w, label=f"D+{lt//24}", color=COLORS[i], alpha=0.9, edgecolor="none")
    xcat(ax, [short[m] for m in model_order])
    ax.set_ylabel(ylab, fontsize=7); ax.set_xlabel("model", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    if nom: ax.axhline(nom, color=PALETTE["red_strong"], ls="--", lw=0.8)
leg_above(axes[0], ncol=3, y=1.16)
for i, ax in enumerate(axes): panel_label(ax, "abcd"[i])
fig.tight_layout(pad=0.6); watermark(fig)
save_pub(fig, OUT, "fig_prev_model_comparison")

# =============== FIG 6 preview ===============
print("Fig.6 preview...")
show = ["v2_mlp", "v7_autoformer", "v11_patchtst", "v15_pinn"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axes[0]
for i, m in enumerate(show):
    rel = reliability(preds[m])
    ax.plot(rel[:, 0], rel[:, 1], marker="o", ms=3, lw=1.2, color=COLORS[i], label=short[m])
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("nominal τ"); ax.set_ylabel("empirical P(y ≤ q̂_τ)")
leg_above(ax, ncol=2)
panel_label(ax, "a")
ax = axes[1]
for i, m in enumerate(show):
    p = pit(preds[m])
    ax.hist(p, bins=10, range=(0, 1), histtype="bar", alpha=0.55, edgecolor="none",
            color=COLORS[i], label=short[m])
ax.set_xlabel("PIT bin"); ax.set_ylabel("count")
leg_above(ax, ncol=2)
panel_label(ax, "b")
ax = axes[2]
data = [preds[m]["q0.95"].values - preds[m]["q0.05"].values for m in show]
bp = ax.boxplot(data, tick_labels=[short[m] for m in show], patch_artist=True,
                showfliers=False, widths=0.6, medianprops=dict(color="black", lw=1))
for patch, c in zip(bp["boxes"], COLORS[:len(show)]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
ax.set_ylabel("q95 − q05 width (W m$^{-2}$)"); ax.set_xlabel("model")
ax.tick_params(axis="x", labelsize=6, rotation=45)
panel_label(ax, "c")
fig.tight_layout(pad=0.6); watermark(fig)
save_pub(fig, OUT, "fig_prev_reliability")

# =============== FIG 7 preview ===============
print("Fig.7 preview...")
d0 = preds["v2_mlp"].merge(sky, on=["station_id", "target_time_utc", "lead_time"], how="left")
d0["err"] = d0["q0.5"] - d0["y"]
day0 = d0[d0.sky_type != "night"].copy()
day0["elev_bin"] = pd.cut(day0.solar_elevation, bins=[0, 15, 45, 90],
                          labels=["low<15", "mid15-45", "high>45"])
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
specs = [("season", ["spring", "summer", "autumn", "winter"], "season"),
         ("sky_type", ["clear", "partly", "overcast"], "weather type"),
         ("elev_bin", ["low<15", "mid15-45", "high>45"], "solar elevation")]
for ax, (col, cats, xlab), lab in zip(axes, specs, "abc"):
    xc = np.arange(len(cats))
    for i, lt in enumerate([24, 48, 72]):
        vals = [day0.loc[(day0[col] == c) & (day0.lead_time == lt), "err"].abs().mean() for c in cats]
        ax.bar(xc + i*0.26, vals, 0.26, label=f"D+{lt//24}", color=COLORS[i], alpha=0.9, edgecolor="none")
    xcat(ax, [str(c).capitalize() if col != "elev_bin" else c for c in cats])
    ax.set_ylabel("MAE (W m$^{-2}$)", fontsize=7); ax.set_xlabel(xlab, fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    leg_above(ax, ncol=3, y=1.16)
    panel_label(ax, lab)
fig.tight_layout(pad=0.6); watermark(fig)
save_pub(fig, OUT, "fig_prev_grouped_error")

# =============== FIG 9 preview ===============
print("Fig.9 preview...")
raw = preds["v2_mlp"]
cal = pd.read_csv(SMOKE / "calib_test" / "test_predictions_calibrated.csv")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axes[0]
rr = reliability(raw); rc = reliability(cal)
ax.plot(rr[:, 0], rr[:, 1], marker="o", ms=3, color=PALETTE["neutral_mid"], label="raw")
ax.plot(rc[:, 0], rc[:, 1], marker="s", ms=3, color=PALETTE["blue_main"], label="calibrated")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("nominal τ"); ax.set_ylabel("empirical")
leg_above(ax, ncol=2); panel_label(ax, "a")
nom = [0.5, 0.8, 0.9]; lo_hi = {0.5: ("q0.25", "q0.75"), 0.8: ("q0.1", "q0.9"), 0.9: ("q0.05", "q0.95")}
xx = np.arange(len(nom)); nlab = [f"{int(n*100)}%" for n in nom]
ax = axes[1]
ax.bar(xx-0.2, [coverage(raw, *lo_hi[n]) for n in nom], 0.38, label="raw",
       color=PALETTE["neutral_mid"], alpha=0.9, edgecolor="none")
ax.bar(xx+0.2, [coverage(cal, *lo_hi[n]) for n in nom], 0.38, label="calibrated",
       color=PALETTE["blue_main"], alpha=0.9, edgecolor="none")
for i, n in enumerate(nom): ax.plot([i-0.4, i+0.4], [n, n], color=PALETTE["red_strong"], lw=1)
xcat(ax, nlab); ax.set_ylabel("coverage"); ax.set_xlabel("nominal level")
leg_above(ax, ncol=2); panel_label(ax, "b")
ax = axes[2]
ax.bar(xx-0.2, [width(raw, *lo_hi[n]) for n in nom], 0.38, label="raw",
       color=PALETTE["neutral_mid"], alpha=0.9, edgecolor="none")
ax.bar(xx+0.2, [width(cal, *lo_hi[n]) for n in nom], 0.38, label="calibrated",
       color=PALETTE["blue_main"], alpha=0.9, edgecolor="none")
xcat(ax, nlab); ax.set_ylabel("interval width (W m$^{-2}$)"); ax.set_xlabel("nominal level")
leg_above(ax, ncol=2); panel_label(ax, "c")
fig.tight_layout(pad=0.6); watermark(fig)
save_pub(fig, OUT, "fig_prev_calibration")

# =============== FIG 10 preview ===============
print("Fig.10 preview...")
site = "nanjing_1"
ds = preds["v2_mlp"][preds["v2_mlp"].station_id == site].merge(
    sky[sky.station_id == site], on=["target_time_utc", "lead_time"], how="left")
ds["date"] = ds.target_time_utc.dt.date
daymean = ds[ds.lead_time == 24].groupby("date")["sky_type"].agg(
    lambda s: s[s != "night"].mode().iat[0] if (s != "night").any() else "night")
pick = {wt: daymean[daymean == wt].index[len(daymean[daymean == wt])//2]
        for wt in ["clear", "partly", "overcast"] if (daymean == wt).any()}
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
for ax, (wt, dt) in zip(axes, pick.items()):
    sub = ds[(ds.date == dt) & (ds.lead_time == 24)].sort_values("target_time_utc")
    t = sub.target_time_utc
    ax.fill_between(t, sub["q0.05"], sub["q0.95"], color=PALETTE["blue_secondary"], alpha=0.22, label="q05–q95")
    ax.fill_between(t, sub["q0.25"], sub["q0.75"], color=PALETTE["blue_main"], alpha=0.40, label="q25–q75")
    ax.plot(t, sub["q0.5"], color=PALETTE["blue_main"], lw=1.2, label="q50")
    ax.plot(t, sub["y"], color=PALETTE["red_strong"], lw=1.0, label="obs")
    ax.set_title(f"{wt} | {dt} | {site}", fontsize=7)
    ax.set_ylabel("GHI (W m$^{-2}$)", fontsize=7); ax.set_xlabel("hour (UTC)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
leg_above(axes[0], ncol=2, y=1.16)
for i, ax in enumerate(axes): panel_label(ax, "abc"[i])
fig.tight_layout(pad=0.6); watermark(fig)
save_pub(fig, OUT, "fig_prev_case_studies")
print("\nPreview figures (QA-styled) done ->", OUT)
