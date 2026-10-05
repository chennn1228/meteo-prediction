"""Minimal dependency boundaries for model-local and aggregate artifacts."""
from __future__ import annotations

from typing import Any, Mapping

from .fingerprints import stable_object_hash


def data_dependency_fingerprint(data_config: Any,
                                record_source_hashes: Any) -> str:
    return stable_object_hash({
        "data_config": data_config, "records": record_source_hashes})


def feature_dependency_fingerprint(data_dependency: str, feature_config: Any,
                                   output_hashes: Mapping[str, str]) -> str:
    return stable_object_hash({
        "clean": data_dependency, "features": feature_config,
        "outputs": dict(output_hashes)})


def split_dependency_fingerprint(feature_dependency: str, sites: Any,
                                 validation: Any) -> str:
    return stable_object_hash({
        "features": feature_dependency, "sites": sites,
        "validation": validation})


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
