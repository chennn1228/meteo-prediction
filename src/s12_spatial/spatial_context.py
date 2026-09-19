"""Require a frozen API-service registry and evidence-backed hourly truth."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from src.s12_spatial.service_registry import (
    ServicePoint, SpatialContractError, require_frozen_registry,
)


@dataclass(frozen=True)
class SpatialContext:
    registry: tuple[ServicePoint, ...]
    truth_evidence: Mapping[str, dict]
    truth_period_start: dt.date
    truth_period_end: dt.date

    @classmethod
    def validated(cls, registry: Iterable[ServicePoint], convergence_audit: dict,
                  truth_evidence: Mapping[str, dict], *, truth_period_start: dt.date,
                  truth_period_end: dt.date) -> "SpatialContext":
        points = require_frozen_registry(registry, convergence_audit)
        if truth_period_start > truth_period_end:
            raise SpatialContractError("Invalid truth period")
        by_id = {point.service_id: point for point in points}
        if not set(truth_evidence).issubset(by_id):
            raise SpatialContractError("Truth evidence references a non-registry point")
        for key, evidence in truth_evidence.items():
            if (evidence.get("evidence_schema") != "validated_hourly_himawari_v1"
                    or evidence.get("gfs_service_id") != key
                    or evidence.get("truth_period_start") != truth_period_start.isoformat()
                    or evidence.get("truth_period_end") != truth_period_end.isoformat()):
                raise SpatialContractError("Truth evidence schema, identity, or period mismatch")
            hours = 24 * ((truth_period_end - truth_period_start).days + 1)
            count = evidence.get("valid_truth_hour_count")
            policy = evidence.get("policy", {})
            fraction = evidence.get("valid_fraction")
            offset = evidence.get("truth_offset_km")
            if (evidence.get("expected_hour_count") != hours
                    or not isinstance(count, int) or not 0 <= count <= hours
                    or not isinstance(fraction, (int, float)) or not math.isclose(fraction, count / hours)
                    or not isinstance(offset, (int, float)) or not math.isfinite(offset) or offset < 0
                    or not isinstance(policy.get("minimum_valid_fraction"), (int, float))
                    or not math.isfinite(policy["minimum_valid_fraction"])
                    or not 0 < policy["minimum_valid_fraction"] <= 1
                    or not isinstance(policy.get("maximum_truth_offset_km"), (int, float))
                    or not math.isfinite(policy["maximum_truth_offset_km"])
                    or policy["maximum_truth_offset_km"] < 0):
                raise SpatialContractError("Malformed hourly Himawari truth evidence")
            eligible = (fraction >= policy["minimum_valid_fraction"]
                        and offset <= policy["maximum_truth_offset_km"])
            if evidence.get("eligible") is not eligible:
                raise SpatialContractError("Truth eligibility does not match measured coverage/offset")
        return cls(points, dict(truth_evidence), truth_period_start, truth_period_end)

    @property
    def by_id(self) -> dict[str, ServicePoint]:
        return {point.service_id: point for point in self.registry}

    @property
    def eligible_ids(self) -> frozenset[str]:
        return frozenset(key for key, evidence in self.truth_evidence.items() if evidence["eligible"])

    def require_eligible(self, service_ids: Iterable[str], label: str) -> tuple[ServicePoint, ...]:
        ids = tuple(service_ids)
        if len(set(ids)) != len(ids):
            raise SpatialContractError(f"Duplicate {label} service identity")
        if not ids or not set(ids).issubset(self.eligible_ids):
            raise SpatialContractError(f"{label} must use Himawari-truth-eligible returned service points")
        return tuple(self.by_id[key] for key in ids)
