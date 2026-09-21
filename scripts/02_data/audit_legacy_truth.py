"""Read-only 20-site Himawari truth coverage audit on historical raw files.

This quantifies missing hours and returned-coordinate offsets without making
legacy raw payloads official-protocol eligible or inventing a quality threshold.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_data_config, load_manifest  # noqa: E402


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da, db = a2 - a1, math.radians(lon2 - lon1)
    value = math.sin(da / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin(db / 2) ** 2
    return 6371.0088 * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def audit_site(root: Path, site: str, periods: dict[str, tuple[str, str]]) -> dict:
    raw = root / "data" / "01_raw"
    gfs_path = raw / "01_gfs" / site / f"{site}_2024-02.json"
    gfs = json.loads(gfs_path.read_text(encoding="utf-8"))
    gfs_coords = (float(gfs["latitude"]), float(gfs["longitude"]))
    times, values, truth_coords = [], [], set()
    year, month = 2024, 2
    while (year, month) <= (2026, 8):
        sat_path = raw / "03_satellite" / site / f"{site}_{year:04d}-{month:02d}.json"
        sat = json.loads(sat_path.read_text(encoding="utf-8"))
        truth_coords.add((float(sat["latitude"]), float(sat["longitude"])))
        times.extend(sat["hourly"]["time"])
        values.extend(sat["hourly"]["shortwave_radiation"])
        month += 1
        if month == 13:
            year, month = year + 1, 1
    if len(truth_coords) != 1:
        raise ValueError(f"{site}: truth returned coordinates changed")
    truth_lat, truth_lon = next(iter(truth_coords))
    utc = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    expected = pd.date_range("2024-02-01", "2026-09-01", freq="h", inclusive="left", tz="UTC")
    if not utc.equals(expected):
        raise ValueError(f"{site}: missing or unordered satellite timestamps")
    solar = pvlib.location.Location(gfs_coords[0], gfs_coords[1], tz="UTC").get_solarposition(utc)
    day = solar["apparent_elevation"].to_numpy(dtype=float) > 0
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(numeric) & (numeric >= 0)
    groups = {}
    for name, (first, last_exclusive) in periods.items():
        in_period = (utc >= pd.Timestamp(first, tz="UTC")) & (utc < pd.Timestamp(last_exclusive, tz="UTC"))
        in_day = in_period & day
        groups[name] = {
            "expected_hours": int(in_period.sum()),
            "valid_hours": int((in_period & valid).sum()),
            "daylight_hours": int(in_day.sum()),
            "valid_daylight_hours": int((in_day & valid).sum()),
            "missing_daylight_hours": int((in_day & ~valid).sum()),
        }
        if groups[name]["daylight_hours"]:
            groups[name]["valid_daylight_fraction"] = (
                groups[name]["valid_daylight_hours"] / groups[name]["daylight_hours"]
            )
    return {"site": site,
            "gfs_returned_coordinates": list(gfs_coords),
            "himawari_returned_coordinates": [truth_lat, truth_lon],
            "truth_offset_km": distance_km(*gfs_coords, truth_lat, truth_lon),
            "periods": groups}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    periods = {
        "all": ("2024-02-01", "2026-09-01"),
        "development": ("2024-02-01", "2025-09-01"),
        "final_test": ("2025-09-01", "2026-09-01"),
    }
    for fold in manifest["validation_protocol"]["outer_folds"]:
        first = fold["validation_start"]
        last = dt.date.fromisoformat(fold["validation_end"]) + dt.timedelta(days=1)
        periods[fold["id"]] = (first, last.isoformat())
    sites = [site["id"] for site in load_data_config("01_sites.yaml", ROOT)["sites"]]
    rows = [audit_site(ROOT, site, periods) for site in sites]
    report = {
        "scope": "historical_raw_truth_coverage_diagnostic_not_official_provenance",
        "site_count": len(rows),
        "periods": periods,
        "total_missing_daylight_hours": sum(row["periods"]["all"]["missing_daylight_hours"] for row in rows),
        "minimum_daylight_valid_fraction": min(row["periods"]["all"]["valid_daylight_fraction"] for row in rows),
        "maximum_truth_offset_km": max(row["truth_offset_km"] for row in rows),
        "sites": rows,
        "official_eligible": False,
        "threshold_policy": "none chosen; counts only",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sites"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
