"""Returned-service-point spatial contracts, selection and evaluation plans."""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import datetime as dt
import math
from typing import Any

import pandas as pd


class SpatialContractError(ValueError):
    """A spatial object is not eligible for the declared evaluation."""


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
        raise SpatialContractError("invalid API-returned service coordinate")
    return round(number, 6)


def service_point(latitude: object, longitude: object, region: str, *,
                  elevation_m: object | None = None,
                  discovery_round: int = 0) -> ServicePoint:
    if not region:
        raise SpatialContractError(
            "region must be derived from returned service coordinates")
    lat, lon = _coordinate(latitude, 90), _coordinate(longitude, 180)
    elevation = None if elevation_m is None else float(elevation_m)
    if elevation is not None and not math.isfinite(elevation):
        raise SpatialContractError("invalid API-returned service elevation")
    if discovery_round < 0:
        raise SpatialContractError("discovery round must be nonnegative")
    return ServicePoint(
        f"om-gfs-land-{lat:.6f}-{lon:.6f}", lat, lon, region,
        elevation, discovery_round)


def build_registry(probes: Iterable[dict], *,
                   province_covers: Callable[[float, float], bool],
                   region_for_service: Callable[[float, float], str]
                   ) -> tuple[ServicePoint, ...]:
    """Deduplicate returned coordinates; request coordinates are provenance only."""
    unique: dict[str, ServicePoint] = {}
    for probe in probes:
        if "service_latitude" not in probe or "service_longitude" not in probe:
            raise SpatialContractError(
                "probe lacks API-returned service coordinates")
        lat = _coordinate(probe["service_latitude"], 90)
        lon = _coordinate(probe["service_longitude"], 180)
        if not province_covers(lon, lat):
            continue
        point = service_point(
            lat, lon, region_for_service(lon, lat),
            elevation_m=probe.get("service_elevation_m"),
            discovery_round=int(probe.get("discovery_round", 0)))
        if point.service_id in unique and unique[point.service_id].region != point.region:
            raise SpatialContractError(
                "conflicting region for one returned service point")
        unique.setdefault(point.service_id, point)
    return tuple(sorted(unique.values()))


def convergence_audit(rounds: Iterable[tuple[float, Iterable[ServicePoint]]], *,
                      stable_rounds: int = 2,
                      boundary_sensitivity_passed: bool = False) -> dict[str, Any]:
    history, cumulative = [], set()
    unchanged, preceding_step = 0, math.inf
    for step, points in rounds:
        if step <= 0 or step >= preceding_step:
            raise SpatialContractError(
                "probe steps must be strictly decreasing positive values")
        preceding_step = step
        ids = {point.service_id for point in points}
        additions, omissions = sorted(ids - cumulative), sorted(cumulative - ids)
        unchanged = unchanged + 1 if history and not additions else 0
        cumulative |= ids
        history.append({
            "probe_step_degrees": step, "round_service_count": len(ids),
            "cumulative_service_count": len(cumulative),
            "added_service_ids": additions,
            "missing_from_this_round": omissions,
            "new_count": len(additions),
            "missing_from_round_count": len(omissions)})
    frozen = (len(history) >= stable_rounds + 1
              and unchanged >= stable_rounds and boundary_sensitivity_passed)
    return {
        "rounds": history, "converged": frozen,
        "final_service_count": len(cumulative) if frozen else None,
        "final_service_ids": sorted(cumulative) if frozen else None,
        "cumulative_discovered_count": len(cumulative),
        "boundary_sensitivity_passed": boundary_sensitivity_passed,
        "boundary_rule": "province_covers(returned_lon, returned_lat)"}


def require_frozen_registry(registry: Iterable[ServicePoint],
                            audit: Mapping[str, Any]) -> tuple[ServicePoint, ...]:
    points = tuple(registry)
    if not audit.get("converged") or audit.get("final_service_count") != len(points):
        raise SpatialContractError(
            "Jiangsu service registry is not convergence-frozen")
    if len({point.service_id for point in points}) != len(points):
        raise SpatialContractError("duplicate returned service coordinates")
    if set(audit.get("final_service_ids") or ()) != {
            point.service_id for point in points}:
        raise SpatialContractError(
            "frozen registry identities differ from final converged probe")
    return points


def truth_gate(points: Iterable[ServicePoint],
               truth_available: Callable[[ServicePoint], bool], *,
               minimum_hourly_coverage: float | None = None
               ) -> tuple[ServicePoint, ...]:
    if (minimum_hourly_coverage is not None
            and not 0 < minimum_hourly_coverage <= 1):
        raise SpatialContractError("truth coverage threshold must be in (0, 1]")
    return tuple(point for point in points if truth_available(point))


def haversine_km(a: ServicePoint, b: ServicePoint) -> float:
    lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
    dlat, dlon = lat2 - lat1, math.radians(b.longitude - a.longitude)
    value = (math.sin(dlat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))


def _nearest_km(point: ServicePoint,
                selected: Sequence[ServicePoint]) -> float:
    return min(haversine_km(point, item) for item in selected)


def _density_contract(selector_config: Mapping[str, Any],
                      regions: Sequence[str]
                      ) -> tuple[tuple[int, ...], dict[int, dict[str, int]]]:
    counts = tuple(int(value) for value in selector_config["allowed_counts"])
    if not counts or tuple(sorted(counts)) != counts:
        raise SpatialContractError("density counts must be strictly increasing")
    quotas = {}
    for count in counts:
        raw = selector_config["region_quotas"].get(
            count, selector_config["region_quotas"].get(str(count)))
        values = ({str(region): int(raw) for region in regions}
                  if isinstance(raw, int)
                  else {str(key): int(value) for key, value in raw.items()})
        if set(values) != set(regions) or sum(values.values()) != count:
            raise SpatialContractError(
                f"density quota does not cover regions or sum to {count}")
        quotas[count] = values
    return counts, quotas


def select_nested_density(points: Iterable[ServicePoint],
                          regions: Sequence[str], *,
                          selector_config: Mapping[str, Any]
                          ) -> dict[int, tuple[ServicePoint, ...]]:
    """Deterministic region-balanced nested farthest-point design."""
    candidates = tuple(sorted(points))
    if not regions or len(set(regions)) != len(regions):
        raise SpatialContractError("ordered regions must be nonempty and unique")
    if len({point.service_id for point in candidates}) != len(candidates):
        raise SpatialContractError("duplicate service IDs in candidate registry")
    if set(point.region for point in candidates) != set(regions):
        raise SpatialContractError("region set does not match frozen registry")
    counts, quotas = _density_contract(selector_config, regions)
    selected: list[ServicePoint] = []
    results: dict[int, tuple[ServicePoint, ...]] = {}
    for count in counts:
        target = quotas[count]
        for region in regions:
            needed = target[region] - sum(
                point.region == region for point in selected)
            for _ in range(needed):
                legal = [point for point in candidates
                         if point.region == region and point not in selected]
                if not legal:
                    raise SpatialContractError(
                        f"region {region} has too few service points")
                if selected:
                    winner = min(
                        legal, key=lambda point: (
                            -_nearest_km(point, selected), point.service_id))
                else:
                    local = [point for point in candidates if point.region == region]
                    winner = min(
                        legal, key=lambda point: (
                            max(haversine_km(point, other) for other in local),
                            point.service_id))
                selected.append(winner)
        if len(selected) != count:
            raise SpatialContractError(
                "density layer does not match configured total")
        results[count] = tuple(selected)
    return results


def coverage_diagnostics(all_points: Iterable[ServicePoint],
                         selected: Sequence[ServicePoint], *,
                         coverage_radius_km: float) -> dict[str, Any]:
    population = tuple(all_points)
    if not population or not selected or coverage_radius_km <= 0:
        raise SpatialContractError(
            "population, training set and positive radius are required")
    if not set(selected).issubset(population):
        raise SpatialContractError(
            "training sites must come from returned-service registry")
    distances = sorted(_nearest_km(point, selected) for point in population)

    def percentile(fraction: float) -> float:
        index = (len(distances) - 1) * fraction
        lower, upper = math.floor(index), math.ceil(index)
        return distances[lower] + (distances[upper] - distances[lower]) * (index - lower)

    covered = sum(distance <= coverage_radius_km for distance in distances)
    return {
        "selected_count": len(selected), "service_count": len(population),
        "maximum_uncovered_distance_km": max(distances),
        "mean_nearest_training_distance_km": sum(distances) / len(distances),
        "nearest_distance_distribution_km": {
            "min": distances[0], "p25": percentile(.25),
            "median": percentile(.5), "p75": percentile(.75),
            "p95": percentile(.95), "max": distances[-1]},
        "region_training_counts": dict(sorted(
            Counter(point.region for point in selected).items())),
        "covered_service_count": covered,
        "coverage_radius_km": coverage_radius_km,
        "service_coverage_fraction": covered / len(distances)}


def assess_himawari_truth(frame: pd.DataFrame, point: ServicePoint, *,
                          start: dt.date, end: dt.date,
                          minimum_valid_fraction: float,
                          maximum_truth_offset_km: float) -> dict[str, Any]:
    if not 0 < minimum_valid_fraction <= 1 or maximum_truth_offset_km < 0:
        raise SpatialContractError(
            "explicit valid-fraction and nonnegative offset policy required")
    required = {
        "target_time_utc", "ghi_obs_sat", "gfs_service_latitude",
        "gfs_service_longitude", "himawari_service_latitude",
        "himawari_service_longitude"}
    if missing := required - set(frame):
        raise SpatialContractError(
            f"missing Himawari truth columns: {sorted(missing)}")
    if set(zip(frame.gfs_service_latitude, frame.gfs_service_longitude)) != {
            (point.latitude, point.longitude)}:
        raise SpatialContractError(
            "frame is not for the selected API-returned GFS point")
    truth_coords = set(zip(
        frame.himawari_service_latitude, frame.himawari_service_longitude))
    if len(truth_coords) != 1:
        raise SpatialContractError(
            "Himawari service coordinate changed or is missing")
    truth_lat, truth_lon = next(iter(truth_coords))
    truth_point = service_point(truth_lat, truth_lon, "truth")
    offset = haversine_km(point, truth_point)
    hours = pd.date_range(
        start, end + dt.timedelta(days=1), freq="h", inclusive="left", tz="UTC")
    grouped = frame.assign(
        _target=pd.to_datetime(frame.target_time_utc, utc=True)).groupby(
            "_target")["ghi_obs_sat"]
    truth = {}
    for target, values in grouped:
        finite = pd.to_numeric(values, errors="coerce").dropna().unique()
        if len(finite) > 1:
            raise SpatialContractError(
                f"inconsistent Himawari truth across leads at {target}")
        truth[target] = float(finite[0]) if len(finite) else math.nan
    if set(truth) != set(hours):
        raise SpatialContractError(
            "Himawari truth timeline has missing or extra target hours")
    valid_count = sum(
        math.isfinite(value) and value >= 0 for value in truth.values())
    fraction = valid_count / len(hours)
    return {
        "evidence_schema": "validated_hourly_himawari",
        "gfs_service_id": point.service_id,
        "truth_service_coordinates": {
            "latitude": truth_point.latitude, "longitude": truth_point.longitude},
        "truth_offset_km": offset, "expected_hour_count": len(hours),
        "valid_truth_hour_count": valid_count, "valid_fraction": fraction,
        "eligible": (fraction >= minimum_valid_fraction
                     and offset <= maximum_truth_offset_km),
        "truth_period_start": start.isoformat(),
        "truth_period_end": end.isoformat(),
        "policy": {"minimum_valid_fraction": minimum_valid_fraction,
                   "maximum_truth_offset_km": maximum_truth_offset_km}}


@dataclass(frozen=True)
class SpatialContext:
    registry: tuple[ServicePoint, ...]
    truth_evidence: Mapping[str, dict]
    truth_period_start: dt.date
    truth_period_end: dt.date

    @classmethod
    def validated(cls, registry: Iterable[ServicePoint],
                  convergence: Mapping[str, Any],
                  truth_evidence: Mapping[str, dict], *,
                  truth_period_start: dt.date,
                  truth_period_end: dt.date) -> "SpatialContext":
        points = require_frozen_registry(registry, convergence)
        if truth_period_start > truth_period_end:
            raise SpatialContractError("invalid truth period")
        by_id = {point.service_id: point for point in points}
        if not set(truth_evidence).issubset(by_id):
            raise SpatialContractError(
                "truth evidence references a non-registry point")
        hours = 24 * ((truth_period_end - truth_period_start).days + 1)
        for key, evidence in truth_evidence.items():
            policy = evidence.get("policy", {})
            count = evidence.get("valid_truth_hour_count")
            fraction = evidence.get("valid_fraction")
            offset = evidence.get("truth_offset_km")
            if (evidence.get("evidence_schema") != "validated_hourly_himawari"
                    or evidence.get("gfs_service_id") != key
                    or evidence.get("truth_period_start") != truth_period_start.isoformat()
                    or evidence.get("truth_period_end") != truth_period_end.isoformat()
                    or evidence.get("expected_hour_count") != hours
                    or not isinstance(count, int) or not 0 <= count <= hours
                    or not isinstance(fraction, (int, float))
                    or not math.isclose(fraction, count / hours)
                    or not isinstance(offset, (int, float)) or offset < 0
                    or not isinstance(policy.get("minimum_valid_fraction"), (int, float))
                    or not 0 < policy["minimum_valid_fraction"] <= 1
                    or not isinstance(policy.get("maximum_truth_offset_km"), (int, float))
                    or policy["maximum_truth_offset_km"] < 0):
                raise SpatialContractError(
                    "malformed hourly Himawari truth evidence schema")
            eligible = (fraction >= policy["minimum_valid_fraction"]
                        and offset <= policy["maximum_truth_offset_km"])
            if evidence.get("eligible") is not eligible:
                raise SpatialContractError(
                    "truth eligibility does not match measured coverage/offset")
        return cls(points, dict(truth_evidence),
                   truth_period_start, truth_period_end)

    @property
    def by_id(self) -> dict[str, ServicePoint]:
        return {point.service_id: point for point in self.registry}

    @property
    def eligible_ids(self) -> frozenset[str]:
        return frozenset(
            key for key, evidence in self.truth_evidence.items()
            if evidence["eligible"])

    def require_eligible(self, service_ids: Iterable[str], label: str
                         ) -> tuple[ServicePoint, ...]:
        ids = tuple(service_ids)
        if len(set(ids)) != len(ids):
            raise SpatialContractError(f"duplicate {label} service identity")
        if not ids or not set(ids).issubset(self.eligible_ids):
            raise SpatialContractError(
                f"{label} must use Himawari-truth-eligible returned service points")
        return tuple(self.by_id[key] for key in ids)


def _distances(test: Iterable[ServicePoint],
               training: Sequence[ServicePoint]) -> dict[str, float]:
    return {point.service_id: min(
        haversine_km(point, candidate) for candidate in training)
            for point in sorted(test)}


def _plan(level: str, train: Sequence[ServicePoint],
          test: Sequence[ServicePoint], *, period: str) -> dict[str, Any]:
    if not train or not test or set(train) & set(test):
        raise SpatialContractError(
            "spatial train/test must be nonempty and disjoint")
    return {
        "level": level, "period": period,
        "training_service_ids": [point.service_id for point in sorted(train)],
        "test_service_ids": [point.service_id for point in sorted(test)],
        "test_distance_to_nearest_training_km": _distances(test, train),
        "geometry_basis": "API-returned GFS service coordinates only",
        "truth_basis": "validated hourly Himawari truth gate",
        "required_grouping": [
            "lead_time", "region", "season", "weather",
            "distance_to_nearest_training_site"],
        "score_status": "not_run"}


def plan_level1(context: SpatialContext,
                candidate_training_ids: Iterable[str], *,
                regions: Sequence[str], holdout_per_region: int = 1
                ) -> dict[str, Any]:
    candidates = context.require_eligible(
        candidate_training_ids, "Level 1 candidates")
    if (not regions or len(set(regions)) != len(regions)
            or holdout_per_region < 1):
        raise SpatialContractError(
            "Level 1 requires unique regions and positive per-region holdout")
    if set(point.region for point in candidates) != set(regions):
        raise SpatialContractError(
            "Level 1 candidate regions differ from declared regions")
    train, test = [], []
    for region in regions:
        regional = sorted(
            point for point in candidates if point.region == region)
        if len(regional) <= holdout_per_region:
            raise SpatialContractError(
                f"insufficient Level 1 sites in region {region}")
        train.extend(regional[:-holdout_per_region])
        test.extend(regional[-holdout_per_region:])
    result = _plan(
        "level_1_stratified_spatial_holdout", train, test,
        period="development_only")
    result["holdout_count_by_region"] = dict(sorted(
        Counter(point.region for point in test).items()))
    return result


def plan_level2(context: SpatialContext,
                training_ids: Iterable[str]) -> dict[str, Any]:
    train = context.require_eligible(training_ids, "Level 2 training")
    train_ids = {point.service_id for point in train}
    test = tuple(point for point in context.registry
                 if point.service_id in context.eligible_ids - train_ids)
    return _plan(
        "level_2_province_unseen_service_points", train, test,
        period="caller_must_verify_official_protocol_window")


def plan_level3(context: SpatialContext,
                candidate_training_ids: Iterable[str], *,
                held_out_region: str,
                regions: Sequence[str]) -> dict[str, Any]:
    if (not regions or len(set(regions)) != len(regions)
            or held_out_region not in regions):
        raise SpatialContractError(
            "Level 3 needs one held-out declared region")
    candidates = context.require_eligible(
        candidate_training_ids, "Level 3 candidates")
    train = tuple(
        point for point in candidates if point.region != held_out_region)
    if {point.region for point in train} != set(regions) - {held_out_region}:
        raise SpatialContractError(
            "Level 3 training must span every non-held-out region")
    test = tuple(point for point in context.registry
                 if point.region == held_out_region
                 and point.service_id in context.eligible_ids)
    result = _plan(
        "level_3_regional_extrapolation", train, test,
        period="caller_must_verify_official_protocol_window")
    result["held_out_region"] = held_out_region
    return result


def plan_density_marginals(context: SpatialContext,
                           layers: Mapping[int, Iterable[str]], *,
                           regions: Sequence[str],
                           selector_config: Mapping[str, Any],
                           coverage_radius_km: float) -> dict[str, Any]:
    counts, quotas = _density_contract(selector_config, regions)
    if set(layers) != set(counts):
        raise SpatialContractError(
            "density layers differ from configured allowed counts")
    previous_ids: set[str] = set()
    selected, diagnostics, marginals = {}, {}, []
    for count in counts:
        points = context.require_eligible(layers[count], f"Density {count}")
        current_ids = {point.service_id for point in points}
        actual = Counter(point.region for point in points)
        if (len(points) != count or not previous_ids.issubset(current_ids)
                or actual != Counter(quotas[count])):
            raise SpatialContractError(
                "density layers violate configured size, quotas or nesting")
        selected[count] = [point.service_id for point in sorted(points)]
        diagnostics[count] = coverage_diagnostics(
            context.registry, points, coverage_radius_km=coverage_radius_km)
        if previous_ids:
            added = sorted(current_ids - previous_ids)
            marginals.append({
                "transition": f"{len(previous_ids)}->{count}",
                "added_service_ids": added,
                "additional_training_points": len(added),
                "additional_valid_truth_hours": sum(
                    context.truth_evidence[key]["valid_truth_hour_count"]
                    for key in added),
                "marginal_probability_gain": None,
                "incremental_compute_seconds": None,
                "status": "awaiting_official_run"})
        previous_ids = current_ids
    return {
        "layers": selected, "coverage_diagnostics": diagnostics,
        "marginals": marginals,
        "geometry_basis": "API-returned GFS service coordinates only",
        "truth_basis": "validated hourly Himawari truth gate",
        "score_status": "not_run"}


def validate_spatial_preflight(
        registry: Iterable[ServicePoint], audit: Mapping[str, Any],
        training: Iterable[ServicePoint], truth_eligible: Iterable[ServicePoint],
        density_layers: Mapping[int, Iterable[ServicePoint]], *,
        selector_config: Mapping[str, Any], regions: Sequence[str]
) -> dict[str, Any]:
    population = require_frozen_registry(registry, audit)
    population_set, train_set = set(population), set(training)
    truth_set = set(truth_eligible)
    if not train_set or not train_set.issubset(population_set):
        raise SpatialContractError(
            "training points must be returned service points")
    if not truth_set or not truth_set.issubset(population_set):
        raise SpatialContractError(
            "truth-gated points must be returned service points")
    counts, quotas = _density_contract(selector_config, regions)
    if set(density_layers) != set(counts):
        raise SpatialContractError(
            "density levels differ from selector configuration")
    preceding: set[ServicePoint] = set()
    for count in counts:
        current = set(density_layers[count])
        actual = Counter(point.region for point in current)
        if (len(current) != count or not preceding.issubset(current)
                or not current.issubset(population_set)
                or actual != Counter(quotas[count])):
            raise SpatialContractError(
                "density sites violate configured size, quotas or nesting")
        preceding = current
    unseen = truth_set - train_set
    if not unseen:
        raise SpatialContractError(
            "Level 2 needs truth-gated unseen service points")
    available_regions = {point.region for point in truth_set}
    if available_regions != set(regions):
        raise SpatialContractError(
            "Level 3 needs truth-gated points in every declared region")
    return {
        "registry_count": len(population),
        "truth_eligible_count": len(truth_set),
        "level_2_unseen_count": len(unseen),
        "level_3_holdout_region_counts": {
            region: sum(point.region == region for point in truth_set)
            for region in regions},
        "density_counts": list(counts),
        "status": "preflight_only_no_official_results"}


def validate_returned_coordinates(columns: set[str]) -> None:
    required = {"gfs_service_latitude", "gfs_service_longitude"}
    if missing := required - columns:
        raise ValueError(
            f"returned service coordinates missing: {sorted(missing)}")
