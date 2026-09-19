"""Audit the Open-Meteo location-specific GFS cells used by this project.

The point API does not expose a raw GRIB grid identifier. This audit therefore
records the returned latitude/longitude/elevation and gives each selected cell
a stable local identifier. With ``--discover-province-grid`` it samples all
in-province request locations densely enough to enumerate distinct service
cells selected for Jiangsu; this is not presented as a raw-GFS grid count.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
OUT = ROOT / "reports" / "01_data_audit" / "source_grid"
API = "https://api.open-meteo.com/v1/forecast"


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def read_header(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        prefix = handle.read(1024)
    def number(name):
        match = re.search(rf'"{name}"\s*:\s*(-?[0-9.]+)', prefix)
        return float(match.group(1)) if match else None
    return number("latitude"), number("longitude"), number("elevation")


def local_site_audit():
    sites = yaml.safe_load((ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    rows = []
    for site in sites:
        files = sorted((ROOT / "data" / "01_raw" / "01_gfs" / site["id"]).glob("*.json"))
        observed = [read_header(path) for path in files]
        cells = sorted({(lat, lon, elev) for lat, lon, elev in observed if lat is not None})
        reference = next((cell for path, cell in zip(files, observed)
                          if cell[0] is not None and "2024-02" in path.stem),
                         cells[-1] if cells else (None, None, None))
        lat, lon, elev = reference
        rows.append({
            "location_id": site["id"],
            "requested_latitude": site["lat"],
            "requested_longitude": site["lon"],
            "region": site.get("region"),
            "source_grid_latitude": lat,
            "source_grid_longitude": lon,
            "source_grid_elevation_m": elev,
            "source_grid_id": f"om-gfs-land-{lat:.6f}-{lon:.6f}" if lat is not None else None,
            "request_to_returned_km": haversine_km(site["lat"], site["lon"], lat, lon) if lat is not None else None,
            "months_scanned": len(files),
            "distinct_returned_cells_over_time": len({(a, b) for a, b, _ in cells}),
            "product_class": "openmeteo_location_specific_gfs",
            "cell_selection": "land",
        })
    return pd.DataFrame(rows)


def load_boundary():
    raw = json.loads((ROOT / "data" / "00_geo" / "jiangsu.geojson").read_text(encoding="utf-8"))
    return unary_union([shape(feature["geometry"]) for feature in raw["features"]])


def request_points(coords, attempts=4):
    params = {
        "latitude": ",".join(str(lat) for _, lat in coords),
        "longitude": ",".join(str(lon) for lon, _ in coords),
        "models": "gfs_seamless",
        "forecast_days": 1, "timezone": "UTC", "cell_selection": "land",
    }
    for attempt in range(attempts):
        try:
            response = requests.get(API, params=params, timeout=120)
            if response.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else [payload]
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def discover_province_grid(step=0.05, batch_size=40, workers=1):
    province = load_boundary()
    minx, miny, maxx, maxy = province.bounds
    lons = np.arange(math.floor(minx / step) * step, maxx + step / 2, step)
    lats = np.arange(math.floor(miny / step) * step, maxy + step / 2, step)
    requests_in = [(round(lon, 5), round(lat, 5)) for lat in lats for lon in lons
                   if province.covers(Point(float(lon), float(lat)))]
    batches = [requests_in[i:i + batch_size] for i in range(0, len(requests_in), batch_size)]
    mappings = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(request_points, batch): batch for batch in batches}
        for done, job in enumerate(as_completed(jobs), 1):
            batch = jobs[job]
            payload = job.result()
            if len(payload) != len(batch):
                raise RuntimeError("API response count does not match request count")
            for (requested_lon, requested_lat), response in zip(batch, payload):
                mappings.append({
                    "requested_longitude": requested_lon,
                    "requested_latitude": requested_lat,
                    "source_grid_longitude": float(response["longitude"]),
                    "source_grid_latitude": float(response["latitude"]),
                    "source_grid_elevation_m": response.get("elevation"),
                })
            if done % 20 == 0 or done == len(batches):
                print(f"batches {done}/{len(batches)}", flush=True)
    mapping_df = pd.DataFrame(mappings)
    cells = (mapping_df.groupby(["source_grid_latitude", "source_grid_longitude"], as_index=False)
             .agg(source_grid_elevation_m=("source_grid_elevation_m", "first"),
                  represented_request_points=("requested_latitude", "size")))
    cells["source_grid_id"] = cells.apply(
        lambda row: f"om-gfs-land-{row.source_grid_latitude:.6f}-{row.source_grid_longitude:.6f}", axis=1)
    cells["returned_centroid_inside_jiangsu"] = cells.apply(
        lambda row: province.covers(Point(row.source_grid_longitude, row.source_grid_latitude)), axis=1)

    truth_path = ROOT / "data" / "00_geo" / "grid_annual_ghi_2025.csv"
    if truth_path.exists():
        truth = pd.read_csv(truth_path)
        def nearest_truth(row):
            distance = ((truth["lat"] - row.source_grid_latitude) ** 2
                        + (truth["lon"] - row.source_grid_longitude) ** 2)
            nearest = truth.loc[distance.idxmin()]
            return pd.Series({"nearest_himawari_proxy_lat": nearest.lat,
                              "nearest_himawari_proxy_lon": nearest.lon,
                              "nearest_himawari_proxy_n_days_2025": nearest.n_days,
                              "nearest_himawari_proxy_distance_deg": math.sqrt(float(distance.min()))})
        cells = pd.concat([cells, cells.apply(nearest_truth, axis=1)], axis=1)
    cells["formal_hourly_truth_eligible"] = False
    return mapping_df, cells


def write_outputs(site_df, mapping_df=None, cells=None, step=None):
    OUT.mkdir(parents=True, exist_ok=True)
    site_df.to_csv(OUT / "20site_source_grid_audit.csv", index=False)
    result = {
        "status": "provisional",
        "product_class": "Open-Meteo location-specific GFS product",
        "province_grid_metadata_endpoint": "Forecast API using the same gfs_seamless model and land-cell selection",
        "raw_gfs_grid_claimed": False,
        "sites": len(site_df),
        "unique_returned_site_cells": int(site_df["source_grid_id"].nunique()),
        "maximum_request_distance_km": float(site_df["request_to_returned_km"].max()),
        "province_sampling_step_degrees": step,
        "province_request_points": int(len(mapping_df)) if mapping_df is not None else None,
        "province_distinct_service_cells": int(len(cells)) if cells is not None else None,
        "formal_hourly_himawari_truth_cells": 0,
        "formal_truth_note": "Full-period hourly Himawari availability has not yet been fetched and gated.",
    }
    if mapping_df is not None:
        mapping_df.to_csv(OUT / "province_request_to_service_cell.csv", index=False)
        cells.to_csv(OUT / "jiangsu_gfs_service_cells.csv", index=False)
        features = []
        for row in cells.itertuples(index=False):
            props = row._asdict()
            lon = props.pop("source_grid_longitude")
            lat = props.pop("source_grid_latitude")
            features.append({"type": "Feature", "geometry": mapping(Point(lon, lat)), "properties": props})
        (OUT / "jiangsu_gfs_service_cells.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
            encoding="utf-8")
    (OUT / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# GFS source-grid audit", "",
        "> The current object is an Open-Meteo location-specific GFS product, not a raw GFS GRIB grid.", "",
        f"- 20 requested sites map to **{result['unique_returned_site_cells']} distinct returned cells**.",
        f"- Maximum request-to-returned-coordinate distance: **{result['maximum_request_distance_km']:.2f} km**.",
        "- The API exposes returned coordinates/elevation but no authoritative raw-GRIB source_grid_id; local semantic IDs are used.",
    ]
    if cells is not None:
        lines += [
            f"- Dense in-province sampling step: {step} degree ({len(mapping_df)} request points).",
            f"- Distinct Open-Meteo service cells selected: **{len(cells)}**.",
            "- Formal province-wide evaluation cells: **0 currently eligible** because full-period hourly Himawari truth has not passed the availability gate.",
        ]
    (OUT / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discover-province-grid", action="store_true")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    site_df = local_site_audit()
    mapping_df = cells = None
    if args.discover_province_grid:
        mapping_df, cells = discover_province_grid(step=args.step, workers=args.workers)
    result = write_outputs(site_df, mapping_df, cells, args.step if cells is not None else None)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
