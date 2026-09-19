"""Build/freeze Jiangsu registry from API-returned points, never request locations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable


class SpatialContractError(ValueError):
    """A spatial object is not eligible for formal evaluation."""


@dataclass(frozen=True, order=True)
class ServicePoint:
    service_id: str
    latitude: float
    longitude: float
    region: str
    elevation_m: float | None = None
    discovery_round: int = 0


def _coordinate(value: object, bound: int) -> float:
    number = float(value)
    if not math.isfinite(number) or abs(number) > bound:
        raise SpatialContractError("Invalid API-returned service coordinate")
    return round(number, 6)


def service_point(latitude: object, longitude: object, region: str,
                  *, elevation_m: object | None = None,
                  discovery_round: int = 0) -> ServicePoint:
    if not region:
        raise SpatialContractError("Region must be derived from returned service coordinates")
    lat, lon = _coordinate(latitude, 90), _coordinate(longitude, 180)
    elevation = None if elevation_m is None else float(elevation_m)
    if elevation is not None and not math.isfinite(elevation):
        raise SpatialContractError("Invalid API-returned service elevation")
    if discovery_round < 0:
        raise SpatialContractError("discovery round must be nonnegative")
    return ServicePoint(f"om-gfs-land-{lat:.6f}-{lon:.6f}", lat, lon, region,
                        elevation, discovery_round)


def build_registry(probes: Iterable[dict], *,
                   province_covers: Callable[[float, float], bool],
                   region_for_service: Callable[[float, float], str]) -> tuple[ServicePoint, ...]:
    """Deduplicate returned coordinates; predicates take (returned_lon, returned_lat).

    Request coordinates may be retained in the probe log but are deliberately
    never read here, even if the API point falls outside Jiangsu.
    """
    unique = {}
    for probe in probes:
        if "service_latitude" not in probe or "service_longitude" not in probe:
            raise SpatialContractError("Probe lacks API-returned service coordinates")
        lat = _coordinate(probe["service_latitude"], 90)
        lon = _coordinate(probe["service_longitude"], 180)
        if not province_covers(lon, lat):
            continue
        point = service_point(lat, lon, region_for_service(lon, lat),
                              elevation_m=probe.get("service_elevation_m"),
                              discovery_round=int(probe.get("discovery_round", 0)))
        if point.service_id in unique and unique[point.service_id].region != point.region:
            raise SpatialContractError("Conflicting region for one returned service point")
        unique.setdefault(point.service_id, point)
    return tuple(sorted(unique.values()))


def convergence_audit(rounds: Iterable[tuple[float, Iterable[ServicePoint]]],
                      *, stable_rounds: int = 2,
                      boundary_sensitivity_passed: bool = False) -> dict:
    """Audit cumulative discovery; two zero-new rounds plus boundary check freeze.

    The existing 0.05-degree 707-point count is an empirical enumeration only,
    not a frozen final population. No network calls happen here.
    """
    history = []
    cumulative: set[str] = set()
    unchanged = 0
    preceding_step = math.inf
    for step, points in rounds:
        if step <= 0 or step >= preceding_step:
            raise SpatialContractError("Probe steps must be strictly decreasing positive values")
        preceding_step = step
        ids = {p.service_id for p in points}
        additions = sorted(ids - cumulative)
        omissions = sorted(cumulative - ids)
        unchanged = unchanged + 1 if history and not additions else 0
        cumulative |= ids
        history.append({"probe_step_degrees": step, "round_service_count": len(ids),
                        "cumulative_service_count": len(cumulative),
                        "added_service_ids": additions, "missing_from_this_round": omissions,
                        "new_count": len(additions), "missing_from_round_count": len(omissions)})
    frozen = (len(history) >= stable_rounds + 1 and unchanged >= stable_rounds
              and boundary_sensitivity_passed)
    return {"rounds": history, "converged": frozen,
            "final_service_count": len(cumulative) if frozen else None,
            "final_service_ids": sorted(cumulative) if frozen else None,
            "cumulative_discovered_count": len(cumulative),
            "boundary_sensitivity_passed": boundary_sensitivity_passed,
            "boundary_rule": "province_covers(returned_lon, returned_lat)",
            "note": "No final count until repeated denser probes show identical returned-point sets."}


def require_frozen_registry(registry: Iterable[ServicePoint], audit: dict) -> tuple[ServicePoint, ...]:
    points = tuple(registry)
    if not audit.get("converged") or audit.get("final_service_count") != len(points):
        raise SpatialContractError("Jiangsu service registry is not convergence-frozen")
    if len({p.service_id for p in points}) != len(points):
        raise SpatialContractError("Duplicate returned service coordinates")
    if set(audit.get("final_service_ids") or ()) != {p.service_id for p in points}:
        raise SpatialContractError("Frozen registry identities differ from final converged probe")
    return points


def truth_gate(points: Iterable[ServicePoint], truth_available: Callable[[ServicePoint], bool],
               *, minimum_hourly_coverage: float | None = None) -> tuple[ServicePoint, ...]:
    """Keep only points whose independent Himawari hourly-truth test passes.

    `truth_available` is a validated caller-supplied test, not nearest-grid guess.
    """
    if minimum_hourly_coverage is not None and not 0 < minimum_hourly_coverage <= 1:
        raise SpatialContractError("Truth coverage threshold must be in (0, 1]")
    return tuple(p for p in points if truth_available(p))
