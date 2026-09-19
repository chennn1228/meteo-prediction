# -*- coding: utf-8 -*-
"""江苏站点体系图 → figs/02_site_design/。

图1 fig_site_coverage_map_global     : 全域最大最小布点（方法说明）
图2 fig_site_coverage_map_stratified : 分层配额+层内最大最小（当前定稿站点，来自 config）
图3 fig_site_layer_error_current    : 分层误差（数据就绪后生成）
图4 fig_jiangsu_20sites             : 两年平均日间 GHI 分布（新站点，数据就绪后生成）

统一：投影（等经纬度，aspect 修正）、坐标范围、刻度、轴标签；图例使用区域代码。
"""
import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
os.environ.setdefault("MPLCONFIGDIR", str(CODE_ROOT / ".cache" / "matplotlib"))
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(CODE_ROOT / "src"))
from s04_evaluation.analysis.plot_common import apply_pub_style, PALETTE, save_pub
apply_pub_style(font_size=7)
from matplotlib.lines import Line2D as L2
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

REGIONS = ["苏南内陆", "苏中内陆", "苏中沿海", "苏北内陆", "苏北沿海"]
CODE = {"苏南内陆": "SI", "苏中内陆": "MI", "苏中沿海": "MC", "苏北内陆": "NI", "苏北沿海": "NC"}
RC = {"苏南内陆": "#6ba3c7", "苏中内陆": "#77b38a", "苏中沿海": "#d0a54f",
      "苏北内陆": "#b58fc2", "苏北沿海": "#d0786f"}
CITIES = [("南京", "苏南", 32.04, 118.78, 0), ("苏州", "苏南", 31.30, 120.62, 0),
          ("无锡", "苏南", 31.49, 120.31, 0), ("常州", "苏南", 31.81, 119.97, 0),
          ("镇江", "苏南", 32.19, 119.45, 0), ("扬州", "苏中", 32.39, 119.41, 0),
          ("泰州", "苏中", 32.46, 119.92, 0), ("南通", "苏中", 31.98, 120.89, 1),
          ("盐城", "苏北", 33.35, 120.16, 1), ("淮安", "苏北", 33.60, 119.02, 0),
          ("宿迁", "苏北", 33.96, 118.28, 0), ("徐州", "苏北", 34.26, 117.19, 0),
          ("连云港", "苏北", 34.60, 119.22, 1)]


def hav(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parts():
    j = json.load(open(CODE_ROOT / "data/01_raw/jiangsu.geojson", encoding="utf-8-sig"))
    g = j["features"][0]["geometry"]
    ps = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    return [(p[0], p[1:]) for p in ps]


def inside(lon, lat, rings):
    def ir(ring):
        ins, n, jj = False, len(ring), len(ring) - 1
        for i in range(n):
            xi, yi = ring[i]; xj, yj = ring[jj]
            if ((yi > lat) != (yj > lat)) and (lon < (xj - xi) * (lat - yi) / (yj - yi) + xi):
                ins = not ins
            jj = i
        return ins
    for ext, holes in rings:
        if ir(ext):
            return not any(ir(h) for h in holes)
    return False


def candidates():
    rings = parts()
    out = []
    lon = 116.2
    while lon <= 122.0:
        lat = 30.7
        while lat <= 35.2:
            if inside(lon, lat, rings):
                c = min(CITIES, key=lambda c: hav(lon, lat, c[3], c[2]))
                out.append({"lon": round(lon, 4), "lat": round(lat, 4),
                            "layer": {"苏南": "苏南内陆",
                                      "苏中": ("苏中沿海" if c[4] else "苏中内陆"),
                                      "苏北": ("苏北沿海" if c[4] else "苏北内陆")}[c[1]]})
            lat += 0.4
        lon += 0.4
    return out


def maximin(pool, k):
    start = min(pool, key=lambda c: sum(hav(c["lon"], c["lat"], d["lon"], d["lat"]) for d in pool))
    sel = [start]; pool = [c for c in pool if c is not start]
    while len(sel) < k and pool:
        b = max(pool, key=lambda c: min(hav(c["lon"], c["lat"], s["lon"], s["lat"]) for s in sel))
        sel.append(b); pool.remove(b)
    return sel


def style_axes(ax):
    ax.set_xlim(116.2, 122.2); ax.set_ylim(30.6, 35.3)
    ax.set_aspect(1 / math.cos(math.radians(33)))
    ax.set_xticks([117, 118, 119, 120, 121, 122])
    ax.set_yticks([31, 32, 33, 34, 35])
    ax.set_xlabel("Longitude (°E)", fontsize=7)
    ax.set_ylabel("Latitude (°N)", fontsize=7)
    ax.tick_params(labelsize=6)


def coverage_map(sel, title, out_name):
    pop = candidates()
    fig, ax = plt.subplots(figsize=(92 / 25.4, 112 / 25.4))
    for ext, holes in parts():
        rr = np.asarray(ext)
        ax.plot(rr[:, 0], rr[:, 1], lw=0.55, color="#9a9a9a")
    ax.scatter([c["lon"] for c in pop], [c["lat"] for c in pop], s=9, marker=".",
               color="#cccccc", zorder=2)
    for s in sel:
        ax.scatter(s["lon"], s["lat"], s=38, marker="o", color=RC[s["layer"]],
                   edgecolor="#333333", linewidths=0.45, zorder=4)
    style_axes(ax)
    ax.set_title("Fig.2 Site design", fontsize=9, pad=4)
    handles = [L2([0], [0], marker="o", ls="none", mfc=RC[r], mec="#333333", ms=5,
                  label=CODE[r]) for r in REGIONS]
    handles.append(L2([0], [0], marker=".", ls="none", color="#cccccc", ms=7, label="candidates"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.2, borderpad=0.4, handletextpad=0.3)
    save_pub(fig, CODE_ROOT / "figs" / "02_site_design", out_name)


def layer_error_current():
    """分层误差图依赖旧模型结果，已作废；最终模型定稿后重写，不再输出旧图。"""
    print("skip layer_error_current: old v1 results obsolete; will rewrite after final model")


def jiangsu_mean_ghi():
    sites_cfg = yaml.safe_load(open(CODE_ROOT / "config" / "01_sites.yaml", encoding="utf-8"))["sites"]
    ghis = {}
    for s in sites_cfg:
        p = CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2024-02_2026-09.parquet"
        if not p.exists():
            print("skip jiangsu_20sites: featured data not ready")
            return
        df = pd.read_parquet(p, columns=["solar_elevation", "ghi_obs_sat"])
        df = df[df["solar_elevation"] > 10].dropna(subset=["ghi_obs_sat"])
        ghis[s["id"]] = float(df["ghi_obs_sat"].mean())
    vals = np.array(list(ghis.values()))
    fig, ax = plt.subplots(figsize=(92 / 25.4, 112 / 25.4))
    for ext, holes in parts():
        rr = np.asarray(ext)
        ax.plot(rr[:, 0], rr[:, 1], lw=0.55, color="#9a9a9a")
    norm = Normalize(vmin=float(np.floor(vals.min()) - 5), vmax=float(np.ceil(vals.max()) + 5))
    cmap = plt.get_cmap("viridis")
    for s in sites_cfg:
        ax.scatter(s["lon"], s["lat"], s=40, marker="o", color=cmap(norm(ghis[s["id"]])),
                   edgecolor="#333333", linewidths=0.45, zorder=4)
    style_axes(ax)
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Mean daytime GHI (W m$^{-2}$), 2024-02–2026-09", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title("Fig.2 Site design", fontsize=9, pad=4)
    save_pub(fig, CODE_ROOT / "figs" / "02_site_design", "fig_jiangsu_20sites")


def main():
    sites_cfg = yaml.safe_load(open(CODE_ROOT / "config" / "01_sites.yaml", encoding="utf-8"))["sites"]
    cur = [{"lon": s["lon"], "lat": s["lat"], "layer": s["region"]} for s in sites_cfg]
    pop = candidates()
    coverage_map(maximin(pop, 20), "Site selection: global maximin (20 of 62)",
                 "fig_site_coverage_map_global")
    coverage_map(cur, "Site selection: stratified quota + within-layer maximin (20)",
                 "fig_site_coverage_map_stratified")
    layer_error_current()
    jiangsu_mean_ghi()
    print("site figs updated")


if __name__ == "__main__":
    main()
