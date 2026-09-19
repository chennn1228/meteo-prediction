# -*- coding: utf-8 -*-
"""Batch-1 aggregate evaluation and formal figures.

Uses RETURNED aggregate results (real full data, not smoke).
Computes:
  - Baselines: persistence / smart-persistence / climatology / optimal-convex(Yang2020)
    from featured data, evaluated on the test year (all-rows and daytime).
  - RMSE skill score per model vs each baseline (all-rows; v1 also daytime via y>0 proxy).
  - Formal Fig.5 (model comparison), Fig.7 (grouped), Fig.9 (calibration before/after).

Per-sample items (night filter, segmented test, conditional coverage, PIT, cases)
are BLOCKED (need test_predictions.csv from server) and are NOT computed here.

Usage: .venv/Scripts/python.exe src/s04_evaluation/analysis/batch1_eval.py
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
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label)
apply_pub_style(font_size=8)

R = CODE_ROOT / "reports"
OUTFIG = CODE_ROOT / "figs" / "03_modeling" / "00_comparison"

TEST_START = pd.Timestamp("2025-09-01", tz="UTC")
TEST_END = pd.Timestamp("2026-08-31 23:00", tz="UTC")

# ---------------- featured -> per (site,time) obs series ----------------
feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT / "data" / "03_featured").glob("*_featured_2024-02_2026-09.parquet"))],
    ignore_index=True)
feat["target_time_utc"] = pd.to_datetime(feat["target_time_utc"])
obs = feat[["station_id", "target_time_utc", "ghi_obs_sat", "kt_obs",
            "ghi_clear_sky", "is_day_fcst"]].drop_duplicates(
    subset=["station_id", "target_time_utc"]).sort_values(
    ["station_id", "target_time_utc"]).set_index(["station_id", "target_time_utc"])

# test rows (per site,time,lead) with truth
test = feat[(feat.target_time_utc >= TEST_START) & (feat.target_time_utc <= TEST_END)].copy()

# ---------------- baselines ----------------
def shift_obs(df, hours):
    idx = pd.MultiIndex.from_arrays([df["station_id"], df["target_time_utc"] - pd.to_timedelta(hours, unit="h")])
    return obs.reindex(idx)["ghi_obs_sat"].values

def shift_kt(df, hours):
    idx = pd.MultiIndex.from_arrays([df["station_id"], df["target_time_utc"] - pd.to_timedelta(hours, unit="h")])
    return obs.reindex(idx)["kt_obs"].values

test["y"] = test["ghi_obs_sat"]
test["b_persist"] = shift_obs(test, test["lead_time"])
kt_now = shift_kt(test, test["lead_time"])
test["b_smart"] = kt_now * test["ghi_clear_sky"]
# climatology: train-pool mean GHI by (site, month, hour)
train = feat[feat.target_time_utc < TEST_START].copy()
train["month"] = train.target_time_utc.dt.month
train["hour"] = train.target_time_utc.dt.hour
clim = train.groupby(["station_id", "month", "hour"])["ghi_obs_sat"].mean()
test["month"] = test.target_time_utc.dt.month
test["hour"] = test.target_time_utc.dt.hour
test["b_clim"] = clim.reindex(pd.MultiIndex.from_arrays(
    [test.station_id, test.month, test.hour])).values
# optimal convex w (fit on train pool): minimize rmse of w*smart+(1-w)*clim
tr = train.copy()
tr["s"] = shift_kt(tr, 0)  # placeholder; fit w on test-free proxy: use train smart vs clim
# simpler: fit w on train using lead=24 alignment
tr24 = train[train["lead_time"] == 24].copy()
tr24["bs"] = shift_kt(tr24, 24) * tr24["ghi_clear_sky"]
tr24["bc"] = clim.reindex(pd.MultiIndex.from_arrays([tr24.station_id,
    tr24.target_time_utc.dt.month, tr24.target_time_utc.dt.hour])).values
tr24 = tr24.dropna(subset=["bs", "bc", "ghi_obs_sat"])
ws = np.linspace(0, 1, 21)
best_w, best_e = 0.5, 1e18
for w in ws:
    e = np.sqrt(((w*tr24["bs"] + (1-w)*tr24["bc"] - tr24["ghi_obs_sat"])**2).mean())
    if e < best_e: best_e, best_w = e, w
test["b_opt"] = best_w*test["b_smart"] + (1-best_w)*test["b_clim"]
print(f"optimal convex w(smart)={best_w:.2f}")

def rmse(y, p): m = np.isfinite(p) & np.isfinite(y); return np.sqrt(((y[m]-p[m])**2).mean())
def mae(y, p): m = np.isfinite(p) & np.isfinite(y); return np.abs(y[m]-p[m]).mean()

base_all, base_day = {}, {}
for name in ["b_persist", "b_smart", "b_clim", "b_opt"]:
    base_all[name] = rmse(test["y"].values, test[name].values)
    d = test[test.is_day_fcst == 1]
    base_day[name] = rmse(d["y"].values, d[name].values)
print("baseline RMSE all :", {k: round(v,1) for k,v in base_all.items()})
print("baseline RMSE day :", {k: round(v,1) for k,v in base_day.items()})

# ---------------- model aggregate results ----------------
deep = ["v2_mlp","v3_cnn","v4_tcn","v5_lstm","v6_transformer","v7_autoformer",
        "v8_informer","v9_fedformer","v10_itransformer","v11_patchtst",
        "v12_dlinear","v13_timesnet","v14_tsmixer","v15_pinn"]
rows = []
for m in deep:
    f = R/"03_modeling"/m/"full"/"quantile"/"ghi"/"results.csv"
    if not f.exists(): continue
    a = pd.read_csv(f); a = a[a.group=="all"]
    if not len(a): continue
    a = a.iloc[0]
    rows.append(dict(model=m, rmse=a.rmse, mae=a.mae, pinball=a.mean_pinball,
                     cov90=a.coverage_90, src="deep"))
# v1 per-sample -> all & daytime rmse
v1p = pd.read_csv(R/"03_modeling/v1_ml/full/quantile/ghi/per_lead_predictions.csv")
for vm in ["raw","bias","linear","ridge","lgbm","xgb"]:
    s = v1p[v1p.model==vm]
    rows.append(dict(model=f"v1_{vm}", rmse=np.sqrt(((s["q0.5"]-s["y"])**2).mean()),
                     mae=np.abs(s["q0.5"]-s["y"]).mean(),
                     pinball=np.mean([np.maximum(t*(s.y-s[c]),(t-1)*(s.y-s[c])).mean()
                        for c,t in zip(["q0.05","q0.1","q0.25","q0.5","q0.75","q0.9","q0.95"],
                                       [0.05,0.1,0.25,0.5,0.75,0.9,0.95])]),
                     cov90=((s.y>=s["q0.05"])&(s.y<=s["q0.95"])).mean(), src="v1"))
cmp = pd.DataFrame(rows)
for b in ["b_smart","b_opt","b_persist","b_clim"]:
    cmp[f"skill_{b[2:]}"] = 1 - cmp.rmse/base_day[b]
cmp = cmp.sort_values("pinball")
print("\n=== GHI skill scores (all-rows) ===")
print(cmp[["model","rmse","pinball","skill_smart","skill_opt","skill_persist"]].round(3).to_string(index=False))
cmp.to_csv(R/"00_summary"/"batch1_skill_ghi.csv" if (R/"00_summary").exists() else R/"batch1_skill_ghi.csv", index=False)

# ---------------- Fig.5 formal ----------------
order = cmp["model"].tolist()
short = {m: m.replace("v1_","") for m in order}
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.7))
metrics = [("pinball","mean pinball",None),("rmse","RMSE (W m$^{-2}$)",None),
           ("cov90","coverage 90%",0.90),("skill_smart","skill vs smart-persist",0.0)]
x = np.arange(len(order))
for ax,(col,ylab,nom) in zip(axes,metrics):
    vals = cmp[col].fillna(0).values
    cols = [PALETTE["blue_main"] if s=="deep" else PALETTE["green_3"] for s in cmp.src]
    ax.bar(x, vals, 0.7, color=cols, alpha=0.9, edgecolor="none")
    ax.set_xticks(x); ax.set_xticklabels([short[m] for m in order], rotation=45,
                                          ha="right", rotation_mode="anchor", fontsize=6)
    ax.set_ylabel(ylab, fontsize=7); ax.set_xlabel("model", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    if nom is not None: ax.axhline(nom, color=PALETTE["red_strong"], ls="--", lw=0.8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor=PALETTE["green_3"], label="v1 tree / linear"),
                    Patch(facecolor=PALETTE["blue_main"], label="deep")],
           loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, fontsize=7, frameon=False)
for i,ax in enumerate(axes): panel_label(ax, "abcd"[i])
fig.tight_layout(pad=0.6)
save_pub(fig, OUTFIG, "fig_model_comparison_full")
print("wrote figs/03_modeling/00_comparison/fig_model_comparison_full")

# ---------------- Fig.9 formal (calib before/after, aggregate) ----------------
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
cal_rows=[]
for m in deep:
    fr = R/"03_modeling"/m/"full"/"quantile"/"ghi"/"results.csv"
    fc = R/"03_modeling"/m/"full"/"quantile"/"ghi"/"calibrated"/"results_calibrated.csv"
    if fr.exists() and fc.exists():
        ar=pd.read_csv(fr); ar=ar[ar.group=="all"].iloc[0]
        ac=pd.read_csv(fc); ac=ac[ac.group=="all"].iloc[0]
        cal_rows.append(dict(model=m, cov_r=ar.coverage_90, cov_c=ac.coverage_90,
                             w_r=ar.width_90, w_c=ac.width_90,
                             p_r=ar.mean_pinball, p_c=ac.mean_pinball))
cal=pd.DataFrame(cal_rows).sort_values("model")
xx=np.arange(len(cal)); ww=0.38
xtl=[s.replace("v","") for s in cal.model]
panels=[("cov_r","cov_c","coverage 90%",0.90),
        ("w_r","w_c","width 90% (W m$^{-2}$)",None),
        ("p_r","p_c","mean pinball",None)]
for ax,(cr,cc,ylab,nom),lab in zip(axes,panels,"abc"):
    ax.bar(xx-ww/2, cal[cr], ww, label="raw", color=PALETTE["neutral_mid"],
           alpha=0.9, edgecolor="none")
    ax.bar(xx+ww/2, cal[cc], ww, label="calibrated", color=PALETTE["blue_main"],
           alpha=0.9, edgecolor="none")
    if nom is not None: ax.axhline(nom, color=PALETTE["red_strong"], ls="--", lw=0.8)
    ax.set_xticks(xx); ax.set_xticklabels(xtl, rotation=45, ha="right",
                                          rotation_mode="anchor", fontsize=6)
    ax.set_ylabel(ylab, fontsize=7); ax.set_xlabel("model", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, frameon=False)
    panel_label(ax, lab)
fig.tight_layout(pad=0.6)
save_pub(fig, CODE_ROOT/"figs"/"06_probability"/"00_comparison", "fig_calibration_effect_full")
print("wrote figs/06_probability/00_comparison/fig_calibration_effect_full")
print("\nBatch-1 aggregate done.")
