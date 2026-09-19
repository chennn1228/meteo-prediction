# -*- coding: utf-8 -*-
"""Systematic data-quality audit: find ratio-explosion & physical-impossible values.

Scans all 20 featured parquets for:
  1. Ratio features (kt/kni/diffuse_fraction) with physically impossible values
  2. Which denominator is near-zero when the ratio explodes
  3. Target outliers (GHI obs > physical ceiling)
  4. Any other division-derived or out-of-range columns

Usage: .venv/Scripts/python.exe src/s04_evaluation/analysis/audit_data_quality.py
"""
import sys
from pathlib import Path
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
import numpy as np
import pandas as pd

FEAT = CODE_ROOT / "data" / "03_featured"
files = sorted(FEAT.glob("*_featured_2024-02_2026-09.parquet"))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
day = df[df["is_day_fcst"] == 1].copy()
N = len(df); Nd = len(day)
print(f"total rows={N:,}  daytime rows={Nd:,}  ({Nd/N:.1%})\n")

print("="*70)
print("1) RATIO-FEATURE EXPLOSION (physically impossible)")
print("="*70)
ratio_specs = [
    ("kt_fcst",  "ghi_fcst",      "ghi_clear_sky", 1.5),
    ("kt_obs",   "ghi_obs_sat",   "ghi_clear_sky", 1.5),
    ("kni",      "dni_fcst",      "dni_clear_sky", 1.5),
    ("diffuse_fraction", "dhi_fcst", "ghi_fcst",   1.0),
]
for name, num, den, phys_max in ratio_specs:
    v = day[name]
    bad = v > phys_max
    nbad = int(bad.sum())
    if nbad == 0:
        print(f"  {name:18s} OK (max={v.max():.2f})")
        continue
    den_at_bad = day.loc[bad, den]
    num_at_bad = day.loc[bad, num]
    print(f"  {name:18s} BAD n={nbad:,} ({nbad/Nd:.1%} of day)  max={v.max():,.0f}")
    print(f"      denominator[{den}] at bad: median={den_at_bad.median():.3f}  "
          f"p90={den_at_bad.quantile(0.9):.3f}  max={den_at_bad.max():.2f}")
    print(f"      numerator[{num}]   at bad: median={num_at_bad.median():.1f}  "
          f"max={num_at_bad.max():.1f}")
    print(f"      solar_elevation at bad: median={day.loc[bad,'solar_elevation'].median():.2f} deg")

print()
print("="*70)
print("2) MECHANISM CHECK: kt_fcst explosion vs clear-sky threshold")
print("="*70)
for thr in [0, 1, 10, 50, 100]:
    ok = day["ghi_clear_sky"] > thr
    kt = np.where(ok, day["ghi_fcst"] / day["ghi_clear_sky"].where(ok), np.nan)
    kt = pd.Series(kt, index=day.index)
    frac_bad = (kt > 1.5).sum() / ok.sum()
    print(f"  threshold ghi_clear_sky>{thr:3d}: kept={ok.sum():,}  "
          f"kt>1.5 fraction={frac_bad:.2%}  kt_max={kt.max():,.0f}")

print()
print("="*70)
print("3) TARGET OUTLIERS (physical ceiling)")
print("="*70)
# Physical ceiling: clear-sky * 1.25 (cloud enhancement upper bound)
ceil = day["ghi_clear_sky"] * 1.25
for col in ["ghi_obs_sat", "ghi_obs_era5", "ghi_fcst"]:
    v = day[col].dropna()
    over_abs = (v > 1300).sum()
    over_rel = (day[col] > ceil).sum()
    print(f"  {col:14s} max={v.max():8.1f}  >1300 abs: {over_abs:,}  "
          f">1.25x clearsky: {over_rel:,} ({over_rel/Nd:.2%})")
# where are the >1300
big = day[day["ghi_obs_sat"] > 1300]
if len(big):
    print(f"    ghi_obs_sat>1300: solar_elev median={big['solar_elevation'].median():.1f}, "
          f"clearsky median={big['ghi_clear_sky'].median():.1f}, "
          f"ratio median={(big['ghi_obs_sat']/big['ghi_clear_sky']).median():.2f}")

print()
print("="*70)
print("4) OTHER RANGE / NaN SCAN (all columns)")
print("="*70)
RANGES = {
    "cloud_cover_fcst": (0,100), "cloud_cover_obs": (0,100),
    "cloud_cover_low_obs": (0,100), "cloud_cover_mid_obs": (0,100), "cloud_cover_high_obs": (0,100),
    "rh_fcst": (0,100), "is_day_fcst": (0,1),
    "solar_elevation": (-90,90), "solar_azimuth": (0,360), "wind_dir_fcst": (0,360),
}
for col,(lo,hi) in RANGES.items():
    v = df[col].dropna()
    oob = ((v<lo)|(v>hi)).sum()
    nn = df[col].isna().sum()
    flag = "  <-- OUT-OF-RANGE" if oob else ""
    print(f"  {col:24s} range=[{v.min():.1f},{v.max():.1f}] expect[{lo},{hi}] "
          f"oob={oob} nan={nn}{flag}")
# high-NaN columns
print("\n  columns with NaN>0:")
for c in df.columns:
    nn = df[c].isna().sum()
    if nn: print(f"    {c:24s} nan={nn:,} ({nn/N:.2%})")

print()
print("="*70)
print("5) DUPLICATE / MONOTONIC SANITY")
print("="*70)
dup = df.duplicated(subset=["station_id","target_time_utc","lead_time"]).sum()
print(f"  duplicate (site,time,lead) rows: {dup}")
print(f"  lead_time values: {sorted(df['lead_time'].unique())}")
print(f"  sky_type counts: {df['sky_type'].value_counts().to_dict()}")
