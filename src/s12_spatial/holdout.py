"""Fail-closed preflight for the three planned formal spatial evaluations.

No result is produced here. A full official run additionally requires frozen
registry, independent Himawari truth and the validated model pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.s12_spatial.service_registry import (
    ServicePoint, SpatialContractError, require_frozen_registry,
)


def validate_spatial_preflight(registry: Iterable[ServicePoint], audit: dict,
                               training: Iterable[ServicePoint],
                               truth_eligible: Iterable[ServicePoint],
                               density_layers: Mapping[int, Iterable[ServicePoint]]) -> dict:
    population = require_frozen_registry(registry, audit)
    population_set = set(population)
    train_set = set(training)
    truth_set = set(truth_eligible)
    if not train_set or not train_set.issubset(population_set):
        raise SpatialContractError("Training points must be returned Jiangsu service points")
    if not truth_set or not truth_set.issubset(population_set):
        raise SpatialContractError("Truth-gated evaluation points must be returned service points")
    if set(density_layers) != {5, 10, 15, 20}:
        raise SpatialContractError("Density levels must be 5/10/15/20")
    preceding: set[ServicePoint] = set()
    for count in (5, 10, 15, 20):
        current = set(density_layers[count])
        if len(current) != count or not preceding.issubset(current) or not current.issubset(population_set):
            raise SpatialContractError("Density sites must be exactly sized, nested, returned service points")
        preceding = current
    unseen = truth_set - train_set
    if not unseen:
        raise SpatialContractError("Level 2 needs truth-gated unseen service points")
    available_regions = {p.region for p in truth_set}
    if len(available_regions) != 5:
        raise SpatialContractError("Level 3 requires truth-gated points in all five regions")
    return {
        "registry_count": len(population), "truth_eligible_count": len(truth_set),
        "level_2_unseen_count": len(unseen),
        "level_3_holdout_region_counts": {
            region: sum(p.region == region for p in truth_set)
            for region in sorted(available_regions)
        },
        "density_counts": [5, 10, 15, 20],
        "status": "preflight_only_no_official_results",
    }
