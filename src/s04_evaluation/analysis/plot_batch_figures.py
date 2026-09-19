# -*- coding: utf-8 -*-
"""Batch figure generation from featured data (no model results needed).

Produces:
  figs/01_data_audit/fig_data_overview        (Fig.1c,d: time series + seasonal boxplots)
  figs/04_error_analysis/fig_exploratory_error (Fig.3: bias/RMSE by season x lead x sky_type)
  figs/01_data_audit/fig_feature_correlation   (Fig.4a: 33-feature correlation heatmap)
  figs/01_data_audit/fig_cloud_identifiability (Fig.11a: GFS vs ERA5 cloud scatter)

Usage:
  .venv/Scripts/python.exe src/s04_evaluation/analysis/plot_batch_figures.py
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
from matplotlib.colors import TwoSlopeNorm

from s04_evaluation.analysis.plot_common import (
    apply_pub_style, PALETTE, COLORS, save_pub, panel_label
)

def leg_above(ax, ncol=2, fs=6, y=1.08):
    """Legend above the panel title (lower edge anchored) so it covers neither title nor data."""
    ax.legend(fontsize=fs, loc="lower center", bbox_to_anchor=(0.5, y),
              ncol=ncol, frameon=False)

apply_pub_style(font_size=8)

# --------------- Load all 20 sites ---------------
FEAT_DIR = CODE_ROOT / "data" / "03_featured"
files = sorted(FEAT_DIR.glob("*_featured_2024-02_2026-09.parquet"))
dfs = []
for f in files:
    d = pd.read_parquet(f)
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)
df["target_time_utc"] = pd.to_datetime(df["target_time_utc"])
print(f"Loaded {len(files)} sites, {len(df):,} rows total")

# --------------- Derived columns ---------------
# Daytime mask
day = df[df["is_day_fcst"] == 1].copy()

# Fix kt explosion: recompute with threshold 50 W/m2
clear_ok = day["ghi_clear_sky"] > 50.0
day["kt_fcst_clean"] = np.where(clear_ok, day["ghi_fcst"] / day["ghi_clear_sky"], np.nan)
day["kt_obs_clean"] = np.where(clear_ok, day["ghi_obs_sat"] / day["ghi_clear_sky"], np.nan)

# GHI error (forecast - observation), daytime only, valid obs
day_valid = day[day["ghi_obs_sat"].notna()].copy()
day_valid["ghi_err"] = day_valid["ghi_fcst"] - day_valid["ghi_obs_sat"]

# Cloud error (all hours)
df["cloud_err"] = df["cloud_cover_fcst"] - df["cloud_cover_obs"]
day_valid["cloud_err"] = day_valid["cloud_cover_fcst"] - day_valid["cloud_cover_obs"]

# Season mapping for ordered plots
SEASON_ORDER = ["spring", "summer", "autumn", "winter"]
SEASON_CN = {"spring": "Spring", "summer": "Summer", "autumn": "Autumn", "winter": "Winter"}

# --------------- FIG 1cd: Data Overview ---------------
print("Plotting Fig.1cd: data overview...")
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

# (c) Time series example: pick one site, one month (2025-07), D+1
site_example = "nanjing_1"
month_example = "2025-07"
mask_c = ((df["station_id"] == site_example) &
          (df["target_time_utc"].dt.strftime("%Y-%m") == month_example) &
          (df["lead_time"] == 24))
sub_c = df[mask_c].sort_values("target_time_utc")

ax = axes[0, 0]
ax.fill_between(sub_c["target_time_utc"], 0, sub_c["ghi_fcst"],
                alpha=0.3, color=PALETTE["blue_secondary"], label="GFS GHI fcst")
ax.plot(sub_c["target_time_utc"], sub_c["ghi_obs_sat"],
        lw=0.8, color=PALETTE["red_strong"], label="Himawari GHI obs")
ax.set_ylabel("GHI (W m$^{-2}$)")
ax.set_title(f"{site_example} | {month_example} | D+1", fontsize=7, pad=2)
leg_above(ax, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.set_xlabel("day of month", fontsize=7)
panel_label(ax, "c")

# (c2) Cloud cover time series same window
ax = axes[0, 1]
ax.fill_between(sub_c["target_time_utc"], 0, sub_c["cloud_cover_fcst"],
                alpha=0.3, color=PALETTE["teal"], label="GFS cloud fcst")
ax.plot(sub_c["target_time_utc"], sub_c["cloud_cover_obs"],
        lw=0.8, color=PALETTE["violet"], label="ERA5 cloud obs")
ax.set_ylabel("Total Cloud Cover (%)")
ax.set_title(f"{site_example} | {month_example} | D+1", fontsize=7, pad=2)
leg_above(ax, ncol=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
ax.set_xlabel("day of month", fontsize=7)
panel_label(ax, "d")

# (d) Seasonal boxplots: GHI obs (daytime)
ax = axes[1, 0]
box_data = [day_valid.loc[day_valid["season"] == s, "ghi_obs_sat"].dropna().values
            for s in SEASON_ORDER]
bp = ax.boxplot(box_data, tick_labels=[SEASON_CN[s] for s in SEASON_ORDER],
                patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", lw=1))
for patch, c in zip(bp["boxes"], COLORS[:4]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel("GHI$_{obs}$ (W m$^{-2}$)")
ax.set_title("Himawari GHI by Season (daytime)", fontsize=7, pad=2)
ax.set_xlabel("season", fontsize=7)
panel_label(ax, "e")

# (d2) Seasonal boxplots: Cloud cover obs
ax = axes[1, 1]
box_data = [df.loc[df["season"] == s, "cloud_cover_obs"].dropna().values
            for s in SEASON_ORDER]
bp = ax.boxplot(box_data, tick_labels=[SEASON_CN[s] for s in SEASON_ORDER],
                patch_artist=True, widths=0.6, showfliers=False,
                medianprops=dict(color="black", lw=1))
for patch, c in zip(bp["boxes"], COLORS[:4]):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_ylabel("Cloud Cover (%)")
ax.set_title("ERA5 Cloud by Season (all hours)", fontsize=7, pad=2)
ax.set_xlabel("season", fontsize=7)
panel_label(ax, "f")

fig.tight_layout(pad=0.8)
save_pub(fig, CODE_ROOT / "figs" / "01_data_audit", "fig_data_overview")
print("  -> figs/01_data_audit/fig_data_overview.{svg,png}")

# --------------- FIG 3: Exploratory Error Analysis ---------------
print("Plotting Fig.3: exploratory error analysis...")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))

# (a) Bias by season x lead (heatmap)
ax = axes[0, 0]
bias_tbl = day_valid.groupby(["season", "lead_time"])["ghi_err"].mean().unstack()
bias_tbl = bias_tbl.reindex(SEASON_ORDER)
im = ax.imshow(bias_tbl.values, cmap="RdBu_r", aspect="auto",
               norm=TwoSlopeNorm(vcenter=0, vmin=-80, vmax=80))
ax.set_xticks(range(3), ["D+1", "D+2", "D+3"])
ax.set_yticks(range(4), [SEASON_CN[s] for s in SEASON_ORDER])
for i in range(4):
    for j in range(3):
        ax.text(j, i, f"{bias_tbl.values[i,j]:.1f}", ha="center", va="center", fontsize=6)
ax.set_title("Bias (fcst − obs), W m$^{-2}$", fontsize=7, pad=2)
ax.set_xlabel("lead time", fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
panel_label(ax, "a")

# (b) RMSE by sky_type x lead
ax = axes[0, 1]
sky_order = ["clear", "partly", "overcast"]
rmse_tbl = day_valid.groupby(["sky_type", "lead_time"]).apply(
    lambda g: np.sqrt((g["ghi_err"]**2).mean())).unstack()
rmse_tbl = rmse_tbl.reindex(sky_order)
x = np.arange(3)
w = 0.25
for i, sky in enumerate(sky_order):
    ax.bar(x + i*w, rmse_tbl.loc[sky].values, w, label=sky.capitalize(),
           color=COLORS[i], alpha=0.8)
ax.set_xticks(x + w, ["D+1", "D+2", "D+3"])
ax.set_ylabel("RMSE (W m$^{-2}$)")
ax.set_title("RMSE by Weather Type", fontsize=7, pad=2)
ax.set_xlabel("lead time", fontsize=7)
leg_above(ax)
panel_label(ax, "b")

# (c) Cloud error vs GHI error scatter (subsample for visibility)
ax = axes[0, 2]
sub = day_valid[day_valid["cloud_err"].notna()].sample(n=min(8000, len(day_valid)), random_state=42)
ax.scatter(sub["cloud_err"], sub["ghi_err"], s=1, alpha=0.15,
           color=PALETTE["blue_main"], rasterized=True)
# Add binned means
bins = np.linspace(-60, 60, 13)
sub["cloud_bin"] = pd.cut(sub["cloud_err"], bins)
binned = sub.groupby("cloud_bin", observed=True)["ghi_err"].mean()
ax.plot(binned.index.categories.mid, binned.values, color=PALETTE["red_strong"], lw=1.5, label="Binned mean")
r = sub["cloud_err"].corr(sub["ghi_err"])
ax.set_xlabel("Cloud err (GFS − ERA5), %")
ax.set_ylabel("GHI err (GFS − Himawari), W m$^{-2}$")
ax.set_title(f"Cloud-GHI Error Coupling (r={r:.3f})", fontsize=7, pad=2)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.axvline(0, color="grey", lw=0.5, ls="--")
leg_above(ax)
panel_label(ax, "c")

# (d) Diurnal cycle of bias (hour_utc)
ax = axes[1, 0]
day_valid["hour_utc"] = day_valid["target_time_utc"].dt.hour
for sky in sky_order:
    sub_d = day_valid[day_valid["sky_type"] == sky]
    hourly = sub_d.groupby("hour_utc")["ghi_err"].mean()
    ax.plot(hourly.index, hourly.values, marker="o", ms=2, lw=1, label=sky.capitalize())
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel("Hour (UTC)")
ax.set_ylabel("Bias (W m$^{-2}$)")
ax.set_title("Diurnal Bias Cycle", fontsize=7, pad=2)
leg_above(ax)
ax.set_xlim(0, 23)
panel_label(ax, "d")

# (e) Error by lead_time (violin)
ax = axes[1, 1]
lead_data = [day_valid.loc[day_valid["lead_time"] == lt, "ghi_err"].values for lt in [24, 48, 72]]
parts = ax.violinplot(lead_data, positions=[1, 2, 3], showmeans=True, showmedians=True, widths=0.7)
for pc in parts["bodies"]:
    pc.set_facecolor(PALETTE["blue_secondary"])
    pc.set_alpha(0.7)
ax.set_xticks([1, 2, 3], ["D+1", "D+2", "D+3"])
ax.set_ylabel("GHI Error (W m$^{-2}$)")
ax.set_title("Error Distribution by Lead", fontsize=7, pad=2)
ax.set_xlabel("lead time", fontsize=7)
ax.axhline(0, color="grey", lw=0.5, ls="--")
panel_label(ax, "e")

# (f) Spatial bias map (20 sites)
ax = axes[1, 2]
site_bias = day_valid.groupby("station_id").agg(
    bias=("ghi_err", "mean"),
    lat=("lat", "first"),
    lon=("lon", "first")
).reset_index()
sc = ax.scatter(site_bias["lon"], site_bias["lat"],
                c=site_bias["bias"], cmap="RdBu_r", s=40,
                vmin=-80, vmax=80, edgecolors="k", linewidths=0.3, zorder=5)
from s04_evaluation.analysis import geo_jiangsu as _geo
_geo.add_jiangsu_outline(ax, show_cities=True, lw_prov=0.9, lw_city=0.3)
_p, _c, _n = _geo.load_jiangsu(); _b = _p.bounds
ax.set_xlim(_b[0]-0.1, _b[2]+0.1); ax.set_ylim(_b[1]-0.1, _b[3]+0.1)
plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Bias (W m$^{-2}$)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Site-mean Bias (daytime)", fontsize=7, pad=2)
panel_label(ax, "f")

fig.tight_layout(pad=0.8)
save_pub(fig, CODE_ROOT / "figs" / "04_error_analysis", "fig_exploratory_error")
print("  -> figs/04_error_analysis/fig_exploratory_error.{svg,png}")

# --------------- FIG 4a: Feature Correlation Heatmap ---------------
print("Plotting Fig.4a: feature correlation heatmap...")
NUM_FEATURES = [
    "ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "terrestrial_fcst",
    "ghi_clear_sky", "dni_clear_sky", "kt_fcst_clean", "kni",
    "ghi_fcst_minus_clear", "diffuse_fraction",
    "cloud_cover_fcst", "cloud_cover_change", "cloud_cover_roll1", "cloud_cover_roll3",
    "temp_fcst", "rh_fcst", "dewpoint_fcst", "wind_speed_fcst", "wind_dir_fcst",
    "pressure_fcst", "precip_fcst", "sunshine_fcst",
    "solar_elevation", "solar_azimuth", "hour_local_sin", "hour_local_cos",
    "doy_sin", "doy_cos", "month", "lead_time",
    "ghi_fcst_lag1", "ghi_fcst_lag2",
]
# Use daytime data with clean kt, drop rows with any NaN in features
feat_df = day[NUM_FEATURES].dropna()
# Clip extreme kt/kni/diffuse_fraction
feat_df["kt_fcst_clean"] = feat_df["kt_fcst_clean"].clip(0, 1.5)
feat_df["kni"] = feat_df["kni"].clip(0, 1.5)
feat_df["diffuse_fraction"] = feat_df["diffuse_fraction"].clip(0, 1.0)
corr = feat_df.corr()

# Hierarchical clustering for ordering
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
dist = 1 - corr.abs().values
np.fill_diagonal(dist, 0)
dist = (dist + dist.T) / 2
link = linkage(squareform(dist, checks=False), method="average")
order = leaves_list(link)
corr_ordered = corr.iloc[order, order]

fig, ax = plt.subplots(figsize=(7.0, 6.5))
im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xticks(range(len(order)))
ax.set_yticks(range(len(order)))
labels = [corr_ordered.columns[i].replace("_clean", "") for i in range(len(order))]
ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
ax.set_yticklabels(labels, fontsize=5.5)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Pearson r")
ax.set_title("Feature Correlation (daytime, hierarchical order)", fontsize=8, pad=4)
fig.tight_layout()
save_pub(fig, CODE_ROOT / "figs" / "01_data_audit", "fig_feature_correlation")
print("  -> figs/01_data_audit/fig_feature_correlation.{svg,png}")

# --------------- FIG 11a: Cloud Identifiability ---------------
print("Plotting Fig.11a: cloud identifiability...")
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

# (a) Scatter: GFS cloud vs ERA5 cloud (subsample)
ax = axes[0]
cloud_sub = df[["cloud_cover_fcst", "cloud_cover_obs", "season"]].dropna()
cloud_sub = cloud_sub.sample(n=min(15000, len(cloud_sub)), random_state=42)
for i, s in enumerate(SEASON_ORDER):
    ss = cloud_sub[cloud_sub["season"] == s]
    ax.scatter(ss["cloud_cover_fcst"], ss["cloud_cover_obs"], s=1, alpha=0.2,
               color=COLORS[i], label=SEASON_CN[s], rasterized=True)
ax.plot([0, 100], [0, 100], "k--", lw=0.8)
r_cloud = cloud_sub["cloud_cover_fcst"].corr(cloud_sub["cloud_cover_obs"])
bias_cloud = (cloud_sub["cloud_cover_fcst"] - cloud_sub["cloud_cover_obs"]).mean()
rmse_cloud = np.sqrt(((cloud_sub["cloud_cover_fcst"] - cloud_sub["cloud_cover_obs"])**2).mean())
ax.set_xlabel("GFS Cloud Cover (%)")
ax.set_ylabel("ERA5 Cloud Cover (%)")
ax.set_title(f"r={r_cloud:.3f}, bias={bias_cloud:.1f}%, RMSE={rmse_cloud:.1f}%", fontsize=7)
leg_above(ax, ncol=2, fs=5.5)
panel_label(ax, "a")

# (b) Bias by lead_time
ax = axes[1]
for lt, lbl in [(24, "D+1"), (48, "D+2"), (72, "D+3")]:
    sub_lt = df[df["lead_time"] == lt]
    bias_by_cc = sub_lt.groupby(pd.cut(sub_lt["cloud_cover_obs"], bins=np.arange(0, 110, 10)),
                                 observed=True)["cloud_err"].mean()
    ax.plot(bias_by_cc.index.categories.mid, bias_by_cc.values, marker="o", ms=3, lw=1, label=lbl)
ax.axhline(0, color="grey", lw=0.5, ls="--")
ax.set_xlabel("ERA5 Cloud Cover bin (%)")
ax.set_ylabel("GFS − ERA5 bias (%)")
ax.set_title("Cloud Bias by Observed Level", fontsize=7, pad=2)
leg_above(ax)
panel_label(ax, "b")

# (c) Bias by season x lead (grouped bar)
ax = axes[2]
cloud_bias_tbl = df.groupby(["season", "lead_time"])["cloud_err"].mean().unstack().reindex(SEASON_ORDER)
x = np.arange(4)
w = 0.25
for i, lt in enumerate([24, 48, 72]):
    ax.bar(x + i*w, cloud_bias_tbl[lt].values, w, label=f"D+{i+1}",
           color=COLORS[i], alpha=0.8)
ax.set_xticks(x + w, [SEASON_CN[s] for s in SEASON_ORDER])
ax.set_ylabel("Bias (%)")
ax.set_title("Cloud Bias: Season × Lead", fontsize=7, pad=2)
ax.set_xlabel("season", fontsize=7)
ax.axhline(0, color="grey", lw=0.5, ls="--")
leg_above(ax)
panel_label(ax, "c")

fig.tight_layout(pad=0.8)
save_pub(fig, CODE_ROOT / "figs" / "01_data_audit", "fig_cloud_identifiability")
print("  -> figs/01_data_audit/fig_cloud_identifiability.{svg,png}")

print("\nAll 4 figures done.")
