"""Offline comparison of existing and newly fetched 15-variable payloads.

This is a discrepancy audit only; it never upgrades missing legacy request
provenance or writes cache sidecars.
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_data_config  # noqa: E402
from s02_data.cache_contract import expected_hourly_fields  # noqa: E402


def audit(site: str) -> dict:
    cfg = load_data_config("02_variables.yaml", ROOT)
    folders = {"previous_runs": "01_gfs", "satellite": "03_satellite", "era5": "02_era5"}
    results = {}
    for source, folder in folders.items():
        differences = []
        total_values = 0
        equal_values = 0
        month = dt.date(2024, 2, 1)
        while month <= dt.date(2026, 8, 1):
            relative = Path(folder) / site / f"{site}_{month:%Y-%m}.json"
            old = json.loads((ROOT / "data" / "01_raw" / relative).read_text(encoding="utf-8"))
            new = json.loads((ROOT / "data" / "protocol15temp" / "01_raw" / relative)
                             .read_text(encoding="utf-8"))
            for key in ("latitude", "longitude", "elevation"):
                if old.get(key) != new.get(key):
                    differences.append(f"{month:%Y-%m} {key}: {old.get(key)} != {new.get(key)}")
            if old["hourly"]["time"] != new["hourly"]["time"]:
                differences.append(f"{month:%Y-%m} hourly time mismatch")
            for field in expected_hourly_fields(source, cfg):
                older = old["hourly"][field]
                newer = new["hourly"][field]
                total_values += len(newer)
                equal_values += sum(a == b for a, b in zip(older, newer))
                if len(older) != len(newer) or older != newer:
                    differences.append(f"{month:%Y-%m} {field}: "
                                       f"{sum(a != b for a, b in zip(older, newer))} unequal")
            month = (month.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        results[source] = {"total_field_hours": total_values,
                           "exactly_equal_field_hours": equal_values,
                           "first_differences": differences[:12],
                           "difference_entries": len(differences)}
    return {"scope": "offline_single_site_comparison_not_official_admission",
            "site": site, "results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="nanjing_1")
    args = parser.parse_args()
    print(json.dumps(audit(args.site), ensure_ascii=False, indent=2))
