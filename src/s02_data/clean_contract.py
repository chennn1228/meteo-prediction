"""Strict monthly coverage and temporal audit before any model-ready parquet."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from src.s02_data.cache_contract import DataContractError, cache_metadata, validate_raw_payload


def expected_month_slices(start: dt.date, end: dt.date):
    if start > end:
        raise DataContractError("start exceeds end")
    current = start
    while current <= end:
        following = (current.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        stop = min(end, following - dt.timedelta(days=1))
        yield current, stop
        current = stop + dt.timedelta(days=1)


def validate_month_paths(data_root: Path, source_dir: str, site: str,
                         start: dt.date, end: dt.date) -> list[tuple[Path, dt.date, dt.date]]:
    paths = []
    for first, last in expected_month_slices(start, end):
        path = data_root / "01_raw" / source_dir / site / f"{site}_{first:%Y-%m}.json"
        if not path.is_file():
            raise DataContractError(f"Missing {source_dir} raw month: {path}")
        paths.append((path, first, last))
    return paths


def validate_cached_month(path: Path, *, source: str, cfg: dict, start: dt.date,
                          end: dt.date, data_version: str, requested_lat: float,
                          requested_lon: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_raw_payload(payload, source=source, cfg=cfg, start=start, end=end)
    sidecar = Path(str(path) + ".meta.json")
    if not sidecar.is_file():
        raise DataContractError(f"Missing cache metadata; old raw file cannot be reused: {path}")
    actual = json.loads(sidecar.read_text(encoding="utf-8"))
    request = actual.get("request", {})
    if (request.get("latitude") != requested_lat or request.get("longitude") != requested_lon
            or request.get("cell_selection") != "land" or request.get("timezone") != "UTC"):
        raise DataContractError(f"Requested coordinates/cell/timezone mismatch: {path}")
    expected = cache_metadata(source=source, cfg=cfg, start=start, end=end,
                              params=request, data_version=data_version)
    if actual != expected:
        raise DataContractError(f"Cache data version, request, lead, or schema mismatch: {path}")
    if request.get("start_date") != start.isoformat() or request.get("end_date") != end.isoformat():
        raise DataContractError(f"Cache month date range mismatch: {path}")
    return payload


def audit_clean_frame(df: pd.DataFrame, start: dt.date, end: dt.date,
                      required_columns: tuple[str, ...]) -> dict:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise DataContractError(f"Missing cleaned columns: {sorted(missing)}")
    target = pd.to_datetime(df["target_time_utc"], utc=True)
    issue = pd.to_datetime(df["fcst_issue_time_utc"], utc=True)
    lead = pd.to_numeric(df["lead_time"], errors="coerce")
    if target.isna().any() or issue.isna().any() or lead.isna().any():
        raise DataContractError("Null or unparsable target/issue/lead")
    if set(lead.unique()) != {24, 48, 72}:
        raise DataContractError("Cleaned data lacks D+1/D+2/D+3")
    if not ((target - issue) == pd.to_timedelta(lead, unit="h")).all():
        raise DataContractError("Forecast issue/target/lead inconsistency")
    keys = pd.DataFrame({"target": target, "lead": lead})
    if keys.duplicated().any():
        raise DataContractError("Duplicate target time × lead records")
    expected = pd.date_range(start, end + dt.timedelta(days=1), freq="h", inclusive="left", tz="UTC")
    actual = pd.DatetimeIndex(target.drop_duplicates().sort_values())
    if not actual.equals(expected):
        raise DataContractError("Cleaned hourly timeline is incomplete or out of requested bounds")
    if len(df) != len(expected) * 3:
        raise DataContractError("Each target hour must have three forecast leads")
    for column in ("requested_latitude", "requested_longitude", "gfs_service_latitude",
                   "gfs_service_longitude", "himawari_service_latitude",
                   "himawari_service_longitude", "era5_service_latitude", "era5_service_longitude"):
        if df[column].isna().any():
            raise DataContractError(f"Missing coordinate: {column}")
    return {
        "expected_months": [f"{date:%Y-%m}" for date, _ in expected_month_slices(start, end)],
        "actual_months": sorted(set(actual.strftime("%Y-%m"))),
        "hourly_count": len(expected),
        "row_count": len(df),
        "lead_hours": [24, 48, 72],
        "missing_rate": {col: float(df[col].isna().mean()) for col in required_columns},
        "service_coordinates": {
            source: sorted({(float(lat), float(lon)) for lat, lon in
                            zip(df[f"{source}_service_latitude"], df[f"{source}_service_longitude"])})
            for source in ("gfs", "himawari", "era5")
        },
    }
