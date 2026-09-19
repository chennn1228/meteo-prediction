# -*- coding: utf-8 -*-
"""Jiangsu spatial figures: REAL gridded Himawari annual GHI + province outline,
in a single equal-area projection (no interpolation, no lon/lat squashing).

Inputs:
  data/00_geo/grid_annual_ghi_2025.csv  (lon, lat, annual_kwh)  [from fetch_grid_ghi.py]
  data/00_geo/jiangsu.geojson           (province + city outlines)
  data/03_featured/*                    (20 sites meta)

Outputs (figs/02_site_design/):
  fig_jiangsu_annual_ghi        : continuous REAL-grid annual GHI (projected)
  fig_jiangsu_20sites_geo       : 20 sites on projected province outline
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyproj import CRS, Transformer
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
from s04_evaluation.analysis.plot_common import apply_pub_style, save_pub
from s04_evaluation.analysis import geo_jiangsu as geo
apply_pub_style(font_size=8)

OUT = CODE_ROOT / "figs" / "02_site_design"
GEO = CODE_ROOT / "data" / "00_geo"
# Albers Equal Area for Jiangsu (lon0=119, std parallels 31/35)
CRS_PROJ = CRS.from_proj4("+proj=aea +lat_1=31 +lat_2=35 +lat_0=33 +lon_0=119 "
                          "+ellps=GRS80 +units=m +no_defs")
TR = Transformer.from_crs(CRS.from_epsg(4326), CRS_PROJ, always_xy=True)

def proj(lon, lat):
    return TR.transform(lon, lat)

prov, cities, _ = geo.load_jiangsu()

def draw_outline(ax, lw_prov=1.2, lw_city=0.4):
    for c in cities:
        for ring in geo._rings(c):
            x, y = proj(ring[:, 0], ring[:, 1]); ax.plot(x, y, lw=lw_city, color="#999", zorder=2)
    for ring in geo._rings(prov):
        x, y = proj(ring[:, 0], ring[:, 1]); ax.plot(x, y, lw=lw_prov, color="#222", zorder=3)

# ---------- FIG 1: 20 sites on projected outline ----------
feat = pd.concat([pd.read_parquet(p) for p in sorted(
    (CODE_ROOT/"data/03_featured").glob("*_featured_2024-02_2026-09.parquet"))], ignore_index=True)
meta = feat.groupby("station_id").agg(lat=("lat","first"), lon=("lon","first")).reset_index()
meta["code"] = meta["station_id"].map(geo.site_codes())
codes = sorted(meta["code"].dropna().unique())
RCMAP = dict(zip(codes, ["#6ba3c7","#77b38a","#d0a54f","#b58fc2","#d0786f"]))
fig, ax = plt.subplots(figsize=(5.2, 4.2))
draw_outline(ax)
for cd in codes:
    s = meta[meta.code == cd]
    x, y = proj(s.lon.values, s.lat.values)
    ax.scatter(x, y, s=40, c=RCMAP.get(cd, "#444"), edgecolors="k", linewidths=0.4,
               label=str(cd), zorder=5)
ax.legend(fontsize=6, loc="lower left", frameon=False, title="layer", title_fontsize=6)
ax.set_xlabel("Easting (m, Albers)"); ax.set_ylabel("Northing (m, Albers)")
ax.set_aspect("equal", adjustable="box")
save_pub(fig, OUT, "fig_jiangsu_20sites_geo")
print("wrote fig_jiangsu_20sites_geo (projected)")

# ---------- FIG 2: REAL-grid annual GHI ----------
csv = GEO / "grid_annual_ghi_2025.csv"
if not csv.exists():
    print("!! grid CSV missing; run fetch_grid_ghi.py first"); sys.exit(1)
g = pd.read_csv(csv)
lon_u = np.sort(g.lon.unique()); lat_u = np.sort(g.lat.unique())
gi = g.set_index(["lat", "lon"])["annual_kwh"]
Z = np.full((len(lat_u), len(lon_u)), np.nan)
for (la, lo), v in gi.items():
    Z[np.where(lat_u == la)[0][0], np.where(lon_u == lo)[0][0]] = v
LON, LAT = np.meshgrid(lon_u, lat_u)
XP, YP = proj(LON, LAT)
cmap = LinearSegmentedColormap.from_list("ghi", ["#b40426","#f46d43","#fde725","#8db4e8","#3b4cc0"][::-1])
fig, ax = plt.subplots(figsize=(5.8, 4.4))
mesh = ax.pcolormesh(XP, YP, Z, shading="nearest", cmap=cmap)
draw_outline(ax)
xs, ys = proj(meta.lon.values, meta.lat.values)
ax.scatter(xs, ys, s=12, c="white", edgecolors="k", linewidths=0.4, zorder=6)
cb = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
cb.set_label("Annual GHI (kWh m$^{-2}$ yr$^{-1}$, 2025)", fontsize=7)
ax.set_xlabel("Easting (m, Albers)"); ax.set_ylabel("Northing (m, Albers)")
ax.set_aspect("equal", adjustable="box")
ax.set_title("Jiangsu annual GHI — real Himawari grid (0.1°)", fontsize=8)
save_pub(fig, OUT, "fig_jiangsu_annual_ghi")
print(f"wrote fig_jiangsu_annual_ghi (real grid, {len(g)} cells, "
      f"{g.annual_kwh.min():.0f}-{g.annual_kwh.max():.0f} kWh)")
