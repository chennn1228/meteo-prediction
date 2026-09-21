"""Resumable, budgeted Open-Meteo returned-service discovery probes."""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from s01_core.config_loader import ProtocolError, load_manifest, project_root
from .service_registry import service_point

API = "https://api.open-meteo.com/v1/forecast"


class ServiceRateLimitError(ProtocolError):
    """The provider rejected the request budget; preserve completed batches."""


def jiangsu_boundary(path: Path | None = None):
    source = path or project_root() / "data" / "00_geo" / "jiangsu.geojson"
    raw = json.loads(Path(source).read_text(encoding="utf-8"))
    return unary_union([shape(item["geometry"]) for item in raw["features"]])


def request_lattice(boundary, step: float) -> list[tuple[float, float]]:
    if not 0 < step <= .05:
        raise ProtocolError("probe step must be positive and no coarser than 0.05°")
    min_lon, min_lat, max_lon, max_lat = boundary.bounds
    lons = np.arange(math.floor(min_lon / step) * step, max_lon + step / 2, step)
    lats = np.arange(math.floor(min_lat / step) * step, max_lat + step / 2, step)
    return [(round(float(lon), 6), round(float(lat), 6)) for lat in lats for lon in lons
            if boundary.covers(Point(float(lon), float(lat)))]


def fetch_batch(requests: list[tuple[float, float]], *, retries: int = 3) -> list[dict]:
    params = {
        "latitude": ",".join(str(lat) for _, lat in requests),
        "longitude": ",".join(str(lon) for lon, _ in requests),
        "models": load_manifest()["data_sources"]["forecast"]["model"],
        "forecast_days": 1,
        "timezone": "UTC",
        "cell_selection": "land",
    }
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = json.load(response)
            rows = payload if isinstance(payload, list) else [payload]
            if len(rows) != len(requests):
                raise ProtocolError("multi-location API response count mismatch")
            result = []
            for (requested_lon, requested_lat), returned in zip(requests, rows):
                if returned.get("error"):
                    raise ProtocolError(f"service probe API error: {returned.get('reason')}")
                point = service_point(returned["latitude"], returned["longitude"],
                                      "unassigned", elevation_m=returned.get("elevation"))
                result.append({"requested_longitude": requested_lon,
                               "requested_latitude": requested_lat,
                               "service_longitude": point.longitude,
                               "service_latitude": point.latitude,
                               "service_elevation_m": point.elevation_m,
                               "service_id": point.service_id})
            return result
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise ServiceRateLimitError(
                    "Open-Meteo returned HTTP 429; cached probe batches are preserved, "
                    "and this round must pause until the provider's quota permits more requests"
                ) from exc
            if attempt + 1 == retries:
                raise ProtocolError(f"service probe failed after retries: {exc}") from exc
            time.sleep(2 ** attempt)
        except (OSError, TimeoutError) as exc:
            if attempt + 1 == retries:
                raise ProtocolError(f"service probe failed after retries: {exc}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def probe_round(step: float, *, batch_size: int = 40, max_new_batches: int = 0,
                data_root: Path | None = None) -> dict:
    """0 new batches is a dry run. Existing batch files are verified and reused."""
    if not 1 <= batch_size <= 40 or max_new_batches < 0:
        raise ProtocolError("batch size must be 1–40 and budget nonnegative")
    boundary = jiangsu_boundary()
    locations = request_lattice(boundary, step)
    root = data_root or project_root() / load_manifest()["data_layout"]["root"]
    output = Path(root) / "04_service_probes" / f"step_{step:g}"
    batches = [locations[i:i + batch_size] for i in range(0, len(locations), batch_size)]
    reused = newly_fetched = 0
    stopped_reason = None
    found: dict[str, dict] = {}
    for index, request in enumerate(batches):
        path = output / f"batch_{index:05d}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["step"] != step or payload["requests"] != [list(item) for item in request]:
                raise ProtocolError(f"probe cache does not match current lattice: {path}")
            rows = payload["returns"]
            reused += 1
        elif newly_fetched < max_new_batches and stopped_reason is None:
            try:
                rows = fetch_batch(request)
            except ServiceRateLimitError:
                stopped_reason = "provider_http_429"
                continue
            output.mkdir(parents=True, exist_ok=True)
            payload = {"step": step, "model": "gfs_seamless", "cell_selection": "land",
                       "requests": [list(item) for item in request], "returns": rows}
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
            newly_fetched += 1
        else:
            continue
        if len(rows) != len(request):
            raise ProtocolError(f"probe batch length mismatch: {path}")
        for row in rows:
            if boundary.covers(Point(row["service_longitude"], row["service_latitude"])):
                found.setdefault(row["service_id"], row)
    complete = reused + newly_fetched == len(batches)
    return {"status": "complete_empirical_round" if complete else "partial_not_frozen",
            "step_degrees": step, "request_points": len(locations),
            "batch_size": batch_size, "total_batches": len(batches),
            "cached_batches": reused, "new_batches": newly_fetched,
            "completed_batches": reused + newly_fetched,
            "currently_discovered_inside_boundary": len(found),
            "service_ids": sorted(found) if complete else None,
            "output_dir": str(output.resolve()),
            "boundary_rule": "covers(returned_lon, returned_lat)",
            "stopped_reason": stopped_reason,
            "frozen": False}
