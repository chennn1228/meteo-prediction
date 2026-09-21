"""Read-only legacy raw/log receipt audit; never certifies request parameters.

The old fetch log records response coordinates and byte lengths, but neither
the full URL nor cell_selection. This audit checks that every in-window raw
file has a matching log receipt and emits a deterministic SHA-256 cohort
fingerprint. It does not create v2 sidecars or grant official eligibility.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_data_config  # noqa: E402


SOURCE_DIRS = {"previous_runs": "01_gfs", "satellite": "03_satellite", "era5": "02_era5"}


def month_slices(start: dt.date, end: dt.date):
    current = start.replace(day=1)
    while current <= end:
        last = dt.date(current.year, current.month,
                       calendar.monthrange(current.year, current.month)[1])
        yield current, min(last, end)
        current = last + dt.timedelta(days=1)


def audit(root: Path, start: dt.date, end: dt.date) -> dict:
    if start.day != 1 or end.day != calendar.monthrange(end.year, end.month)[1]:
        raise ValueError("This legacy audit requires a whole-month window")
    sites = load_data_config("01_sites.yaml", root)["sites"]
    log_path = root / "data" / "01_raw" / "fetch_log.csv"
    with log_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or [])
        required = {"source", "site", "start", "end", "status", "grid_lat",
                    "grid_lon", "bytes", "error"}
        if not required <= columns:
            raise ValueError(f"Legacy log missing columns: {sorted(required - columns)}")
        rows = defaultdict(list)
        for row in reader:
            if row["status"] == "ok":
                rows[(row["source"], row["site"], row["start"], row["end"])].append(row)

    cohort_digest = hashlib.sha256()
    expected = matched = missing = mismatched = sidecars = bytes_total = 0
    duplicate_keys = conflicting_receipts = 0
    first_problems = []
    for site in sites:
        site_id = site["id"]
        for first, last in month_slices(start, end):
            for source, folder in SOURCE_DIRS.items():
                expected += 1
                path = root / "data" / "01_raw" / folder / site_id / f"{site_id}_{first:%Y-%m}.json"
                if not path.is_file():
                    missing += 1
                    first_problems.append(f"missing: {path}")
                    continue
                if Path(str(path) + ".meta.json").is_file():
                    sidecars += 1
                raw = path.read_bytes()
                bytes_total += len(raw)
                relative = path.relative_to(root).as_posix()
                cohort_digest.update(relative.encode("utf-8") + b"\0" +
                                     hashlib.sha256(raw).digest())
                try:
                    payload = json.loads(raw)
                    latitude, longitude = float(payload["latitude"]), float(payload["longitude"])
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    mismatched += 1
                    first_problems.append(f"unreadable response coordinates: {path}: {exc}")
                    continue
                candidates = rows[(source, site_id, first.isoformat(), last.isoformat())]
                if len(candidates) > 1:
                    duplicate_keys += 1
                signatures = {(row["bytes"], row["grid_lat"], row["grid_lon"])
                              for row in candidates}
                if len(signatures) > 1:
                    conflicting_receipts += 1
                if any(int(row["bytes"]) == len(raw)
                       and abs(float(row["grid_lat"]) - latitude) < 1e-6
                       and abs(float(row["grid_lon"]) - longitude) < 1e-6
                       for row in candidates if row["bytes"] and row["grid_lat"] and row["grid_lon"]):
                    matched += 1
                else:
                    mismatched += 1
                    first_problems.append(f"no matching legacy fetch receipt: {path}")

    return {
        "scope": "legacy_file_and_fetch_log_receipts_only",
        "window": [start.isoformat(), end.isoformat()],
        "site_count": len(sites),
        "expected_files": expected,
        "matched_files": matched,
        "missing_files": missing,
        "mismatched_receipts": mismatched,
        "legacy_sidecar_count": sidecars,
        "duplicate_log_keys": duplicate_keys,
        "conflicting_duplicate_receipts": conflicting_receipts,
        "total_raw_bytes": bytes_total,
        "cohort_sha256": cohort_digest.hexdigest(),
        "log_contains_full_request": {"request", "url", "cell_selection"} & columns ==
                                     {"request", "url", "cell_selection"},
        "explicit_cell_selection_proven": False,
        "official_eligible": False,
        "first_problems": first_problems[:20],
    }


def main() -> None:
    report = audit(ROOT, dt.date(2024, 2, 1), dt.date(2026, 8, 31))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_files"] or report["mismatched_receipts"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
