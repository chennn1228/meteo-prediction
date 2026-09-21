"""Read-only, legacy-data diagnostic for the first outer-fold duration conflict.

This script never changes the manifest or certifies old files for official use.
It checks candidate INNER scoring durations against real returned-coordinate
daylight, and conservatively restarts the 168-hour continuity count per block.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
import yaml

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())


def audit(score_days: int, early_stop_days: int) -> dict:
    sites = yaml.safe_load((ROOT / "config" / "01_sites.yaml").read_text(encoding="utf-8"))["sites"]
    start = pd.Timestamp("2024-02-01", tz="UTC")
    end = pd.Timestamp("2024-06-01", tz="UTC")
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    gap = pd.Timedelta(days=10)
    score_span = pd.Timedelta(days=score_days)
    early_span = pd.Timedelta(days=early_stop_days)
    per_site = []
    for site in sites:
        payloads = [json.loads((ROOT / "data" / "01_raw" / "01_gfs" / site["id"] /
                               f"{site['id']}_2024-{month:02d}.json").read_text(encoding="utf-8"))
                    for month in range(2, 6)]
        coordinates = {(p["latitude"], p["longitude"], p["elevation"]) for p in payloads}
        if len(coordinates) != 1:
            raise ValueError(f"returned service coordinates changed in first outer: {site['id']}")
        latitude, longitude, elevation = next(iter(coordinates))
        solar = pvlib.location.Location(latitude, longitude, altitude=elevation, tz="UTC")
        daytime = np.asarray(solar.get_solarposition(hours)["apparent_elevation"] > 0)
        for lead in (1, 2, 3):
            gfs = [value for payload in payloads
                   for value in payload["hourly"][f"shortwave_radiation_previous_day{lead}"]]
            sat = [value for month in range(2, 6)
                   for value in json.loads((ROOT / "data" / "01_raw" / "03_satellite" /
                                            site["id"] / f"{site['id']}_2024-{month:02d}.json")
                                           .read_text(encoding="utf-8"))["hourly"]["shortwave_radiation"]]
            if len(gfs) != len(hours) or len(sat) != len(hours):
                raise ValueError(f"hour length mismatch: {site['id']}, D+{lead}")
            valid = np.array([(a is not None and b is not None) for a, b in zip(gfs, sat)])
            for index in range(3):
                score_end = end - (2 - index) * score_span
                score_start = score_end - score_span
                early_end = score_start - gap
                early_start = early_end - early_span
                fit_end = early_start - gap
                blocks = {"fit": (start, fit_end), "early_stop": (early_start, early_end),
                          "score": (score_start, score_end)}
                counts = {}
                for name, (left, right) in blocks.items():
                    mask = np.asarray((hours >= left) & (hours < right))
                    selected = np.flatnonzero(mask)
                    raw = int((valid & mask).sum())
                    day = int((valid & daytime & mask).sum())
                    # Conservative: restart lookback inside every disjoint block.
                    seq = int((valid & daytime)[selected[167:]].sum()) if len(selected) >= 168 else 0
                    counts[name] = {"raw": raw, "daytime": day, "sequence_168": seq}
                per_site.append({"site": site["id"], "lead": lead, "inner_fold": index + 1,
                                 "fit_days": (fit_end - start).days, "counts": counts})
    return {"status": "legacy_diagnostic_not_official", "score_days": score_days,
            "early_stop_days": early_stop_days, "gap_days": 10,
            "site_lead_inner_combinations": len(per_site),
            "minimum_fit_days": min(row["fit_days"] for row in per_site),
            "minimum_by_block": {
                block: {metric: min(row["counts"][block][metric] for row in per_site)
                        for metric in ("raw", "daytime", "sequence_168")}
                for block in ("fit", "early_stop", "score")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-days", type=int, required=True)
    parser.add_argument("--early-stop-days", type=int, default=14)
    args = parser.parse_args()
    print(json.dumps(audit(args.score_days, args.early_stop_days), indent=2))


if __name__ == "__main__":
    main()
