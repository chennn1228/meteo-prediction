# -*- coding: utf-8 -*-
"""Fetch REAL gridded Himawari GHI over Jiangsu (no interpolation).

Builds a 0.1-deg grid masked to the province, pulls Open-Meteo satellite
(jma_jaxa_himawari) daily shortwave_radiation_sum for calendar 2025 per cell,
and writes annual GHI (kWh m^-2 yr^-1) per grid cell to a CSV cache.

Output: data/00_geo/grid_annual_ghi_2025.csv  (lon, lat, annual_kwh, n_days)
"""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, requests
CODE_ROOT = next(p for p in Path(__file__).resolve().parents
                 if (p / "config" / "01_sites.yaml").exists())
sys.path.insert(0, str(CODE_ROOT / "src"))
from s04_evaluation.analysis import geo_jiangsu as geo

GEO = CODE_ROOT / "data" / "00_geo"
OUT = GEO / "grid_annual_ghi_2025.csv"
STEP = 0.1
YEAR = "2025"
URL = "https://satellite-api.open-meteo.com/v1/archive"

prov, cities, _ = geo.load_jiangsu()
minx, miny, maxx, maxy = prov.bounds
lons = np.arange(minx, maxx + 1e-9, STEP)
lats = np.arange(miny, maxy + 1e-9, STEP)
pts = [(round(lo, 3), round(la, 3)) for la in lats for lo in lons if geo.contains_point(lo, la)]
print(f"grid {STEP}deg: {len(pts)} cells inside province (of {len(lons)*len(lats)})")

def fetch_batch(coords):
    p = {"latitude": ",".join(str(c[1]) for c in coords),
         "longitude": ",".join(str(c[0]) for c in coords),
         "daily": "shortwave_radiation_sum",
         "models": "jma_jaxa_himawari",
         "start_date": f"{YEAR}-01-01", "end_date": f"{YEAR}-12-31",
         "timezone": "Asia/Shanghai"}
    for attempt in range(4):
        try:
            r = requests.get(URL, params=p, timeout=120)
            if r.status_code == 200:
                return r.json()
            r.raise_for_status()
        except Exception as e:
            print("  retry", attempt, e); time.sleep(3 * (attempt + 1))
    return None

rows = []
B = 40
for i in range(0, len(pts), B):
    batch = pts[i:i+B]
    res = fetch_batch(batch)
    if res is None:
        print(f"  batch {i} FAILED, skip"); continue
    if isinstance(res, dict):
        res = [res]
    for (lo, la), rec in zip(batch, res):
        vals = rec.get("daily", {}).get("shortwave_radiation_sum") or []
        vals = [v for v in vals if v is not None]
        if vals:
            rows.append(dict(lon=lo, lat=la, annual_kwh=sum(vals) / 3.6, n_days=len(vals)))
    print(f"  {min(i+B,len(pts))}/{len(pts)} cells, {len(rows)} ok")

df = pd.DataFrame(rows).drop_duplicates(subset=["lon", "lat"])
df.to_csv(OUT, index=False)
print(f"wrote {OUT}: {len(df)} cells | annual kWh min={df.annual_kwh.min():.0f} "
      f"max={df.annual_kwh.max():.0f} mean={df.annual_kwh.mean():.0f}")
