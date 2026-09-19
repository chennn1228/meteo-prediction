# -*- coding: utf-8 -*-
"""kt 分组阈值轻量稳定性校验：跨站点、跨季节的高斯混合边界离散度。"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.mixture import GaussianMixture

CODE_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "config" / "01_sites.yaml").exists())
sites = yaml.safe_load(open(CODE_ROOT / "config" / "01_sites.yaml", encoding="utf-8"))["sites"]

def season_of(ts):
    m = ts.month
    return "spring" if m in (3,4,5) else "summer" if m in (6,7,8) else "autumn" if m in (9,10,11) else "winter"

def daily_kt(sid):
    df = pd.read_parquet(CODE_ROOT / "data" / "03_featured" / f"{sid}_featured_2019-02_2026-01.parquet",
                         columns=["target_time_utc","solar_elevation","ghi_obs_sat","ghi_clear_sky"])
    df = df[(df["solar_elevation"]>10)].dropna(subset=["ghi_obs_sat","ghi_clear_sky"]).copy()
    df["date"] = pd.to_datetime(df["target_time_utc"], utc=True).dt.normalize()
    return (df["ghi_obs_sat"]/df["ghi_clear_sky"]).groupby(df["date"]).mean()

def boundaries(x):
    x = np.clip(x, 0, 1.4).reshape(-1,1)
    m = GaussianMixture(n_components=4, random_state=0, covariance_type="full").fit(x)
    means = np.sort(m.means_.ravel()); sds = np.sqrt(np.sort(m.covariances_.ravel()))
    out = []
    for i in range(3):
        a, b, sa, sb = means[i], means[i+1], sds[i], sds[i+1]
        out.append((a*sb + b*sa)/(sa+sb))
    return out

rows = {}
# 分站
for s in sites:
    kt = daily_kt(s["id"]).to_numpy()
    if len(kt) < 120:
        continue
    rows[s["id"]] = {"station": boundaries(kt)}
# 分季（合并全站）
seas = {se: [] for se in ["spring","summer","autumn","winter"]}
for s in sites:
    df = pd.read_parquet(CODE_ROOT / "data" / "03_featured" / f"{s['id']}_featured_2019-02_2026-01.parquet",
                         columns=["target_time_utc","solar_elevation","ghi_obs_sat","ghi_clear_sky"])
    df = df[(df["solar_elevation"]>10)].dropna(subset=["ghi_obs_sat","ghi_clear_sky"])
    df["date"] = pd.to_datetime(df["target_time_utc"], utc=True).dt.normalize()
    g = (df["ghi_obs_sat"]/df["ghi_clear_sky"]).groupby(df["date"]).mean()
    se = pd.Series([season_of(d) for d in g.index], index=g.index)
    for k in seas:
        v = g[se == k].to_numpy()
        if len(v) >= 80:
            seas[k] += list(boundaries(v))
        else:
            seas[k] += [np.nan]*3
rows["season"] = {"station": None}
out = []
out.append("# kt 分组阈值稳定性校验（轻量）\n")
out.append("方法：各站/各季单独拟合 4 分量高斯混合，取相邻分量决策边界；报告三组边界的分布。\n")
st = np.array([rows[s]["station"] for s in rows if s != "season"])
for i, name in enumerate(["b1 (阴/多云)", "b2 (多云/少云)", "b3 (少云/晴)"]):
    v = st[:, i]
    out.append(f"- {name}：站点间 均值 {v.mean():.3f}，SD {v.std():.3f}，范围 [{v.min():.3f}, {v.max():.3f}]")
sv = np.array(seas["season"]) if False else np.vstack([np.array(seas[k]) for k in ["spring","summer","autumn","winter"]])
sv = sv[~np.isnan(sv).any(axis=1)]
for i, name in enumerate(["b1", "b2", "b3"]):
    v = sv[:, i]
    if len(v):
        out.append(f"- {name}（季节）：均值 {v.mean():.3f}，SD {v.std():.3f}，范围 [{v.min():.3f}, {v.max():.3f}]")
out.append("\n候选总体边界（全数据）≈ 0.44/0.73/0.97。判定：站点间与季节间 SD 均小于相邻档距的一半（约 0.15）则稳定。")
o = CODE_ROOT / "reports" / "05_robustness" / "verification" / "kt_grouping_stability"
o.mkdir(parents=True, exist_ok=True)
(o / "summary.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))