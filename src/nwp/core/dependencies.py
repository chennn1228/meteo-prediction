"""Minimal dependency boundaries for model-local and aggregate artifacts."""
from __future__ import annotations

from typing import Any, Mapping

from .fingerprints import stable_object_hash


def model_dependency_fingerprint(
    stage: str, model_id: str, *, upstream: Mapping[str, str],
    model_record: Mapping[str, Any], search_space: Any = None,
    stage_contract: Any = None,
) -> str:
    if stage not in {"tuning", "fitting", "prediction", "calibration", "metrics", "interpretation"}:
        raise ValueError(f"not a model-local stage: {stage}")
    return stable_object_hash({
        "stage": stage, "model_id": model_id,
        "upstream": dict(upstream), "model_record": model_record,
        "search_space": search_space if stage == "tuning" else None,
        "stage_contract": stage_contract,
    })


def aggregate_dependency_fingerprint(
    stage: str, model_artifact_hashes: Mapping[str, str], *, contract: Any
) -> str:
    if stage not in {"comparison", "aggregate_metrics", "aggregate_analysis"}:
        raise ValueError(f"not an aggregate stage: {stage}")
    return stable_object_hash({
        "stage": stage, "models": dict(model_artifact_hashes), "contract": contract})
