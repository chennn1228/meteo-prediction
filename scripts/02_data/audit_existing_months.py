"""Read-only audit of legacy monthly payloads against the active field contract.

Passing this audit does not establish original request provenance or make
legacy files eligible for an official run.
"""
from __future__ import annotations

import argparse
import calendar
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_data_config, load_manifest  # noqa: E402
from s02_data.cache_contract import expected_hourly_fields, validate_raw_payload  # noqa: E402


def months(start: dt.date, end: dt.date):
    cursor = start.replace(day=1)
    while cursor <= end:
        last = dt.date(cursor.year, cursor.month,
                       calendar.monthrange(cursor.year, cursor.month)[1])
        yield cursor, min(last, end)
        cursor = last + dt.timedelta(days=1)


def audit(root: Path, start: dt.date, end: dt.date) -> dict:
    cfg = load_data_config("02_variables.yaml", root)
    sites = load_data_config("01_sites.yaml", root)["sites"]
    counts = {source: {"valid_payloads": 0, "missing_files": 0,
                       "invalid_payloads": 0, "sidecars": 0,
                       "total_field_hours": 0, "null_field_hours": 0}
              for source in ("previous_runs", "era5", "satellite")}
    field_nulls = {source: {} for source in counts}
    month_nulls = {source: {} for source in counts}
    service_coordinates = defaultdict(set)
    log_rows = defaultdict(list)
    with (root / "data" / "01_raw" / "fetch_log.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        for row in csv.DictReader(stream):
            if row["status"] == "ok":
                log_rows[(row["source"], row["site"], row["start"], row["end"])].append(row)
    log_audit = {"matched_files": 0, "missing_or_mismatched": 0,
                 "nonzero_utc_offset": 0, "unstable_returned_site_coordinates": []}
    problems = []
    folders = {"previous_runs": "01_gfs", "era5": "02_era5",
               "satellite": "03_satellite"}
    for site in sites:
        for first, last in months(start, end):
            for source, folder in folders.items():
                path = root / "data" / "01_raw" / folder / site["id"] / f"{site['id']}_{first:%Y-%m}.json"
                if not path.is_file():
                    counts[source]["missing_files"] += 1
                    problems.append(f"missing: {path}")
                    continue
                if Path(str(path) + ".meta.json").is_file():
                    counts[source]["sidecars"] += 1
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    validate_raw_payload(payload, source=source, cfg=cfg,
                                         start=first, end=last)
                    if payload.get("utc_offset_seconds") != 0:
                        log_audit["nonzero_utc_offset"] += 1
                        problems.append(f"nonzero UTC offset: {path}")
                    coordinates = (float(payload["latitude"]), float(payload["longitude"]))
                    service_coordinates[(source, site["id"])].add(coordinates)
                    rows = log_rows[(source, site["id"], first.isoformat(), last.isoformat())]
                    matched = any(
                        int(row["bytes"]) == path.stat().st_size
                        and abs(float(row["grid_lat"]) - coordinates[0]) < 1e-6
                        and abs(float(row["grid_lon"]) - coordinates[1]) < 1e-6
                        for row in rows
                        if row["bytes"] and row["grid_lat"] and row["grid_lon"]
                    )
                    if matched:
                        log_audit["matched_files"] += 1
                    else:
                        log_audit["missing_or_mismatched"] += 1
                        problems.append(f"no exact matching fetch-log receipt: {path}")
                    hourly = payload["hourly"]
                    for field in expected_hourly_fields(source, cfg):
                        values = hourly[field]
                        count = sum(value is None for value in values)
                        counts[source]["total_field_hours"] += len(values)
                        counts[source]["null_field_hours"] += count
                        field_nulls[source][field] = field_nulls[source].get(field, 0) + count
                        key = f"{first:%Y-%m}"
                        month_nulls[source][key] = month_nulls[source].get(key, 0) + count
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    counts[source]["invalid_payloads"] += 1
                    problems.append(f"invalid: {path}: {exc}")
                else:
                    counts[source]["valid_payloads"] += 1
    log_audit["unstable_returned_site_coordinates"] = [
        {"source": source, "site": site, "coordinate_count": len(values)}
        for (source, site), values in sorted(service_coordinates.items()) if len(values) != 1
    ]
    return {"scope": "legacy_payload_structure_only_not_official_provenance",
            "forecast_variable_count": len(cfg["forecast_variables"]),
            "site_count": len(sites), "months": sum(1 for _ in months(start, end)),
            "data_version": load_manifest(root / "project_manifest.yaml")["data_version"],
            "counts": counts,
            "fetch_log_audit": log_audit,
            "fields_with_most_nulls": {source: sorted(values.items(), key=lambda item: -item[1])[:8]
                                       for source, values in field_nulls.items()},
            "months_with_nulls": {source: sorted([(month, count) for month, count in values.items()
                                                  if count], key=lambda item: item[0])
                                  for source, values in month_nulls.items()},
            "first_problems": problems[:20],
            "official_eligible": False,
            "provenance_limitation": "fetch log has no URL or full request parameters; old files have no sidecars"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-02-01")
    parser.add_argument("--end", default="2026-08-31")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(ROOT, dt.date.fromisoformat(args.start),
                   dt.date.fromisoformat(args.end))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
