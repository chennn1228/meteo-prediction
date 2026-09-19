"""Deterministic, region-balanced 5 ⊂ 10 ⊂ 15 ⊂ 20 farthest-point design."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

from src.s12_spatial.service_registry import ServicePoint, SpatialContractError


def haversine_km(a: ServicePoint, b: ServicePoint) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat = lat2 - lat1
    dlon = math.radians(b.longitude - a.longitude)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _nearest_km(point: ServicePoint, selected: Sequence[ServicePoint]) -> float:
    return min(haversine_km(point, s) for s in selected)


def select_nested_density(points: Iterable[ServicePoint], regions: Sequence[str]) -> dict[int, tuple[ServicePoint, ...]]:
    """One new point per region per layer; farthest legal candidate wins.

    Initialization is the lexicographically smallest service_id in the first
    region. All later distance ties use ascending service_id; no RNG is used.
    """
    candidates = tuple(sorted(points))
    if len(regions) != 5 or len(set(regions)) != 5:
        raise SpatialContractError("Exactly five unique ordered regions are required")
    if len({p.service_id for p in candidates}) != len(candidates):
        raise SpatialContractError("Duplicate service IDs in candidate registry")
    if set(p.region for p in candidates) != set(regions):
        raise SpatialContractError("Region set does not match frozen registry")
    counts = Counter(p.region for p in candidates)
    if any(counts[region] < 4 for region in regions):
        raise SpatialContractError("Each region needs at least four service points")
    selected: list[ServicePoint] = []
    results = {}
    for layer in range(1, 5):
        for region in regions:
            legal = [p for p in candidates if p.region == region and p not in selected]
            if not selected:
                winner = legal[0]
            else:
                winner = min(legal, key=lambda p: (-_nearest_km(p, selected), p.service_id))
            selected.append(winner)
        results[layer * 5] = tuple(selected)
    return results


def coverage_diagnostics(all_points: Iterable[ServicePoint], selected: Sequence[ServicePoint],
                         *, coverage_radius_km: float) -> dict:
    all_points = tuple(all_points)
    if not selected or coverage_radius_km <= 0:
        raise SpatialContractError("Nonempty training set and positive coverage radius required")
    if not set(selected).issubset(all_points):
        raise SpatialContractError("Training sites must come from returned-service registry")
    distances = sorted(_nearest_km(p, selected) for p in all_points)
    def percentile(fraction: float) -> float:
        index = (len(distances) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        return distances[lower] + (distances[upper] - distances[lower]) * (index - lower)
    return {
        "selected_count": len(selected),
        "service_count": len(all_points),
        "maximum_uncovered_distance_km": max(distances),
        "mean_nearest_training_distance_km": sum(distances) / len(distances),
        "nearest_distance_distribution_km": {
            "min": distances[0], "p25": percentile(.25), "median": percentile(.5),
            "p75": percentile(.75), "p95": percentile(.95), "max": distances[-1],
        },
        "region_training_counts": dict(sorted(Counter(p.region for p in selected).items())),
        "covered_service_count": sum(d <= coverage_radius_km for d in distances),
        "coverage_radius_km": coverage_radius_km,
        "service_coverage_fraction": sum(d <= coverage_radius_km for d in distances) / len(distances),
    }
