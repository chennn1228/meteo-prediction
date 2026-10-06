"""Non-destructive inventory and contract audits for research data."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from nwp.core.hashing import file_sha256
from nwp.core.schema import ContractError
from nwp.evaluation.spatial import service_point

from .contracts import DataCatalog, validate_raw_payload


DATE = re.compile(r"(20\d{2}-\d{2})(?:[_-](20\d{2}-\d{2}))?")
SITE = re.compile(r"(?:^|[/\\])([a-z]+(?:_[a-z]+)*_\d+)(?:[_/\\]|$)")
INVENTORY_ACTIONS = frozenset({"KEEP", "MOVE", "MERGE", "DUPLICATE_DELETE", "UNKNOWN"})
SERVICE_PROBE_API = "https://api.open-meteo.com/v1/forecast"


class ServiceRateLimitError(ContractError):
    """The provider rejected the request budget; completed batches remain valid."""


def _estimated_stage(path: Path, data_root: Path) -> str:
    parts = {item.lower() for item in path.relative_to(data_root).parts}
    if {"raw", "01_raw"} & parts:
        return "RAW"
    if {"clean", "02_clean"} & parts:
        return "CLEAN"
    if {"features", "03_featured"} & parts:
        return "FEATURES"
    if "registry" in parts:
        return "REGISTRY"
    return "UNKNOWN"


def _schema(path: Path) -> str:
    if path.suffix.lower() != ".parquet":
        return ""
    try:
        import pyarrow.parquet as pq

        return str(pq.ParquetFile(path).schema_arrow)
    except Exception as exc:  # unreadable files are inventory findings
        return f"UNREADABLE: {type(exc).__name__}"


def _row(path: Path, data_root: Path) -> dict[str, Any]:
    relative = path.relative_to(data_root).as_posix()
    date_match = DATE.search(path.name)
    site_match = SITE.search(relative)
    stage = _estimated_stage(path, data_root)
    parts = path.relative_to(data_root).parts
    in_quarantine = "quarantine" in parts
    is_receipt = path.name.endswith(".receipt.json")
    canonical_root = parts[0] in {"raw", "clean", "features", "registry"}
    action = (
        "UNKNOWN" if in_quarantine or stage == "UNKNOWN" else
        "KEEP" if canonical_root else
        "MOVE"
    )
    return {
        "path": relative,
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
        "extension": path.suffix.lower(),
        "estimated_stage": stage,
        "site": site_match.group(1) if site_match else "",
        "date_range": " to ".join(item for item in date_match.groups() if item) if date_match else "",
        "schema": _schema(path),
        "provenance_available": path.with_suffix(".receipt.json").exists(),
        "is_receipt": is_receipt,
        "in_quarantine": in_quarantine,
        "action": action,
    }


def inventory_data(
    data_root: Path,
    *,
    output: Path | None = None,
) -> dict[str, int]:
    """Hash/classify every data file without moving or deleting anything."""
    data_root = data_root.resolve()
    output = (output or data_root / "data_inventory.csv").resolve()
    excluded = {output, (data_root / "catalog.json").resolve()}
    files = sorted(
        item
        for item in data_root.rglob("*")
        if item.is_file() and item.resolve() not in excluded
    )
    rows = [_row(item, data_root) for item in files]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    for copies in by_hash.values():
        if len(copies) > 1:
            canonical = sorted(
                (row for row in copies if row["action"] == "KEEP"),
                key=lambda item: item["path"])
            if canonical:
                keep = canonical[0]
                for row in copies:
                    if row is not keep:
                        row["action"] = "DUPLICATE_DELETE"
    if any(row["action"] not in INVENTORY_ACTIONS for row in rows):
        raise AssertionError("inventory produced an unregistered action")
    fields = [
        "path",
        "size",
        "sha256",
        "extension",
        "estimated_stage",
        "site",
        "date_range",
        "schema",
        "provenance_available",
        "is_receipt",
        "in_quarantine",
        "action",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["action"]] += 1
    return dict(counts)


def audit_month_coverage(
    catalog: DataCatalog,
    *,
    stage: str,
    site_ids: Iterable[str],
    months: Iterable[str],
) -> dict[str, Any]:
    """Report cataloged ready coverage; never infer readiness from filenames."""
    expected = {(site, month) for site in site_ids for month in months}
    actual: set[tuple[str, str]] = set()
    for record in catalog.records():
        if record.stage != stage or record.status != "ready" or len(record.sites) != 1:
            continue
        for month in months:
            if record.time_range.startswith(month):
                actual.add((record.sites[0], month))
    missing = sorted(expected - actual)
    return {
        "status": "pass" if not missing else "blocked",
        "stage": stage,
        "expected_partitions": len(expected),
        "ready_partitions": len(actual & expected),
        "missing": [{"site_id": site, "month": month} for site, month in missing],
    }


def audit_truth(data_config: Mapping[str, Any]) -> dict[str, Any]:
    primary = data_config["truth"]["primary"]
    supplementary = data_config["truth"]["supplementary"]
    issues = []
    if primary.get("semantics") != data_config.get("radiation_semantics"):
        issues.append("primary truth and forecast radiation semantics differ")
    if not primary.get("variables"):
        issues.append("primary truth variables are empty")
    if not supplementary.get("variables"):
        issues.append("supplementary truth variables are empty")
    return {
        "status": "pass" if not issues else "blocked",
        "primary_provider": primary.get("provider"),
        "supplementary_provider": supplementary.get("provider"),
        "issues": issues,
    }


def audit_gfs_quality(
    payload_paths: Iterable[Path],
    *,
    data_config: Mapping[str, Any],
    start: dt.date,
    end: dt.date,
) -> dict[str, Any]:
    audits = []
    for path in payload_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(f"cannot read GFS payload: {path}") from exc
        audits.append(
            {
                "path": str(path),
                **validate_raw_payload(
                    payload,
                    source="previous_runs",
                    data_config=data_config,
                    start=start,
                    end=end,
                ),
            }
        )
    return {
        "status": "pass",
        "file_count": len(audits),
        "files": audits,
    }


def service_boundary(path: Path):
    """Load the explicitly supplied province boundary for returned-point probes."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    raw = json.loads(path.read_text(encoding="utf-8"))
    return unary_union([shape(item["geometry"]) for item in raw["features"]])


def service_request_lattice(boundary, step: float) -> list[tuple[float, float]]:
    """Create a deterministic request lattice; returned points remain the study object."""
    import numpy as np
    from shapely.geometry import Point

    if not 0 < step <= .05:
        raise ContractError(
            "probe step must be positive and no coarser than 0.05 degrees")
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    lons = np.arange(math.floor(min_lon / step) * step,
                     max_lon + step / 2, step)
    lats = np.arange(math.floor(min_lat / step) * step,
                     max_lat + step / 2, step)
    return [
        (round(float(lon), 6), round(float(lat), 6))
        for lat in lats for lon in lons
        if boundary.covers(Point(float(lon), float(lat)))]


def fetch_service_probe_batch(
        requests: list[tuple[float, float]], *, model: str,
        retries: int = 3) -> list[dict[str, Any]]:
    """Fetch one bounded service-point probe batch with fail-closed quota handling."""
    params = {
        "latitude": ",".join(str(lat) for _, lat in requests),
        "longitude": ",".join(str(lon) for lon, _ in requests),
        "models": model, "forecast_days": 1, "timezone": "UTC",
        "cell_selection": "land"}
    url = SERVICE_PROBE_API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            rows = payload if isinstance(payload, list) else [payload]
            if len(rows) != len(requests):
                raise ContractError(
                    "multi-location API response count mismatch")
            result = []
            for (requested_lon, requested_lat), returned in zip(requests, rows):
                if returned.get("error"):
                    raise ContractError(
                        f"service probe API error: {returned.get('reason')}")
                point = service_point(
                    returned["latitude"], returned["longitude"], "unassigned",
                    elevation_m=returned.get("elevation"))
                result.append({
                    "requested_longitude": requested_lon,
                    "requested_latitude": requested_lat,
                    "service_longitude": point.longitude,
                    "service_latitude": point.latitude,
                    "service_elevation_m": point.elevation_m,
                    "service_id": point.service_id})
            return result
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ServiceRateLimitError(
                    "Open-Meteo returned HTTP 429; cached probe batches are preserved"
                ) from exc
            if attempt + 1 == retries:
                raise ContractError(
                    f"service probe failed after retries: {exc}") from exc
            time.sleep(2 ** attempt)
        except (OSError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise ContractError(
                    f"service probe failed after retries: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def probe_service_round(
        step: float, *, data_root: Path, boundary_path: Path, model: str,
        batch_size: int = 40, max_batches: int = 0) -> dict[str, Any]:
    """Verify/reuse cached batches and optionally fetch an explicit bounded budget."""
    from shapely.geometry import Point

    if not 1 <= batch_size <= 40 or max_batches < 0:
        raise ContractError(
            "batch size must be 1-40 and budget nonnegative")
    boundary = service_boundary(boundary_path)
    locations = service_request_lattice(boundary, step)
    output = data_root / "registry" / "service_probes" / f"step_{step:g}"
    batches = [locations[index:index + batch_size]
               for index in range(0, len(locations), batch_size)]
    reused = newly_fetched = 0
    stopped_reason = None
    found: dict[str, dict] = {}
    for index, request in enumerate(batches):
        path = output / f"batch_{index:05d}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload["step"] != step
                    or payload["requests"] != [list(item) for item in request]
                    or payload["model"] != model):
                raise ContractError(
                    f"probe cache does not match resolved request: {path}")
            rows, reused = payload["returns"], reused + 1
        elif newly_fetched < max_batches and stopped_reason is None:
            try:
                rows = fetch_service_probe_batch(request, model=model)
            except ServiceRateLimitError:
                stopped_reason = "provider_http_429"
                continue
            output.mkdir(parents=True, exist_ok=True)
            payload = {
                "step": step, "model": model, "cell_selection": "land",
                "requests": [list(item) for item in request], "returns": rows}
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            newly_fetched += 1
        else:
            continue
        if len(rows) != len(request):
            raise ContractError(f"probe batch length mismatch: {path}")
        for row in rows:
            if boundary.covers(Point(
                    row["service_longitude"], row["service_latitude"])):
                found.setdefault(row["service_id"], row)
    complete = reused + newly_fetched == len(batches)
    return {
        "status": "complete_empirical_round" if complete else "partial_not_frozen",
        "step_degrees": step, "request_points": len(locations),
        "batch_size": batch_size, "total_batches": len(batches),
        "cached_batches": reused, "new_batches": newly_fetched,
        "completed_batches": reused + newly_fetched,
        "currently_discovered_inside_boundary": len(found),
        "service_ids": sorted(found) if complete else None,
        "output_dir": str(output.resolve()),
        "boundary_rule": "covers(returned_lon, returned_lat)",
        "stopped_reason": stopped_reason, "frozen": False}
