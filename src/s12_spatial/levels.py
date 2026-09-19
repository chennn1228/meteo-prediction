"""Deterministic Level 1/2/3 and density preflight plans; no official scores."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from src.s12_spatial.density import coverage_diagnostics, haversine_km
from src.s12_spatial.service_registry import ServicePoint, SpatialContractError
from src.s12_spatial.spatial_context import SpatialContext


def _distances(test: Iterable[ServicePoint], training: Sequence[ServicePoint]) -> dict[str, float]:
    return {point.service_id: min(haversine_km(point, candidate) for candidate in training)
            for point in sorted(test)}


def _plan(level: str, train: Sequence[ServicePoint], test: Sequence[ServicePoint],
          *, period: str) -> dict:
    if not train or not test or set(train) & set(test):
        raise SpatialContractError("Spatial train/test must be nonempty and disjoint")
    return {
        "level": level,
        "period": period,
        "training_service_ids": [point.service_id for point in sorted(train)],
        "test_service_ids": [point.service_id for point in sorted(test)],
        "test_distance_to_nearest_training_km": _distances(test, train),
        "geometry_basis": "API-returned GFS service coordinates only",
        "truth_basis": "validated hourly Himawari truth gate",
        "required_grouping": ["lead_time", "region", "season", "weather",
                              "distance_to_nearest_training_site"],
        "required_evaluation": ["raw_gfs", "corrected_quantiles", "skill",
                                "improvement", "probability_metrics"],
        "score_status": "not_run",
    }


def plan_level1(context: SpatialContext, candidate_training_ids: Iterable[str],
                *, regions: Sequence[str], holdout_per_region: int = 1) -> dict:
    """Development-only deterministic stratified holdout of existing training sites."""
    candidates = context.require_eligible(candidate_training_ids, "Level 1 candidates")
    if len(regions) != 5 or len(set(regions)) != 5 or holdout_per_region < 1:
        raise SpatialContractError("Level 1 requires five regions and positive per-region holdout")
    if set(point.region for point in candidates) != set(regions):
        raise SpatialContractError("Level 1 candidate regions differ from declared five regions")
    train, test = [], []
    for region in regions:
        regional = sorted(point for point in candidates if point.region == region)
        if len(regional) <= holdout_per_region:
            raise SpatialContractError(f"Insufficient Level 1 sites in region {region}")
        train.extend(regional[:-holdout_per_region])
        test.extend(regional[-holdout_per_region:])
    result = _plan("level_1_stratified_spatial_holdout", train, test, period="development_only")
    result["holdout_count_by_region"] = dict(sorted(Counter(point.region for point in test).items()))
    return result


def plan_level2(context: SpatialContext, training_ids: Iterable[str]) -> dict:
    """All truth-eligible province service points unseen in training.

    A requested 0.1-degree grid or annual-GHI CSV cannot enter the context.
    """
    train = context.require_eligible(training_ids, "Level 2 training")
    train_ids = {point.service_id for point in train}
    test = tuple(point for point in context.registry
                 if point.service_id in context.eligible_ids - train_ids)
    return _plan("level_2_province_unseen_service_points", train, test,
                 period="caller_must_verify_official_protocol_window")


def plan_level3(context: SpatialContext, candidate_training_ids: Iterable[str],
                *, held_out_region: str, regions: Sequence[str]) -> dict:
    """Four-region fit, fifth-region truth-gated extrapolation stress test."""
    if len(regions) != 5 or len(set(regions)) != 5 or held_out_region not in regions:
        raise SpatialContractError("Level 3 requires one of exactly five declared regions")
    candidates = context.require_eligible(candidate_training_ids, "Level 3 candidates")
    train = tuple(point for point in candidates if point.region != held_out_region)
    if {point.region for point in train} != set(regions) - {held_out_region}:
        raise SpatialContractError("Level 3 training must span all four non-held-out regions")
    test = tuple(point for point in context.registry
                 if point.region == held_out_region and point.service_id in context.eligible_ids)
    result = _plan("level_3_regional_extrapolation", train, test,
                   period="caller_must_verify_official_protocol_window")
    result["held_out_region"] = held_out_region
    return result


def plan_density_marginals(context: SpatialContext,
                           layers: Mapping[int, Iterable[str]], *,
                           regions: Sequence[str], coverage_radius_km: float) -> dict:
    """Audit nested cost geometry; probability benefit remains unknown until run."""
    if set(layers) != {5, 10, 15, 20} or len(regions) != 5 or len(set(regions)) != 5:
        raise SpatialContractError("Density plan requires five regions and 5/10/15/20 layers")
    selected = {}
    previous_ids: set[str] = set()
    diagnostics = {}
    marginals = []
    for count in (5, 10, 15, 20):
        points = context.require_eligible(layers[count], f"Density {count}")
        current_ids = {point.service_id for point in points}
        quotas = Counter(point.region for point in points)
        if len(points) != count or not previous_ids.issubset(current_ids):
            raise SpatialContractError("Density layers must have exact sizes and strict nesting")
        if quotas != Counter({region: count // 5 for region in regions}):
            raise SpatialContractError("Density region quotas must be 1/2/3/4 per region")
        selected[count] = [point.service_id for point in sorted(points)]
        diagnostics[count] = coverage_diagnostics(context.registry, points,
                                                  coverage_radius_km=coverage_radius_km)
        if previous_ids:
            added = sorted(current_ids - previous_ids)
            marginals.append({
                "transition": f"{count - 5}->{count}",
                "added_service_ids": added,
                "additional_training_points": len(added),
                "additional_valid_truth_hours": sum(
                    context.truth_evidence[key]["valid_truth_hour_count"] for key in added),
                "marginal_probability_gain": None,
                "incremental_compute_seconds": None,
                "status": "awaiting_official_run",
            })
        previous_ids = current_ids
    return {"layers": selected, "coverage_diagnostics": diagnostics,
            "marginals": marginals, "geometry_basis": "API-returned GFS service coordinates only",
            "truth_basis": "validated hourly Himawari truth gate",
            "score_status": "not_run"}
