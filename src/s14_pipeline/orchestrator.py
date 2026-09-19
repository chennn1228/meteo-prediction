"""Stage registry; no training or official execution is implicit."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from s01_core.config_loader import ProtocolError
from s01_core.provenance import validate_run_levels
from s15_validation.validate_project import result as validation_result


@dataclass(frozen=True)
class Stage:
    number: int
    name: str
    implementation: str


STAGES = (
    Stage(1, "validate", "structural_preflight"),
    Stage(2, "data", "requires_v2_refetch"),
    Stage(3, "service_points", "requires_convergence_probe"),
    Stage(4, "features", "causal_library_available"),
    Stage(5, "splits", "rolling_library_available"),
    Stage(6, "tuning_cpu", "blocked_until_real_data_and_validated_models"),
    Stage(7, "train_cpu", "blocked_until_real_data_and_validated_models"),
    Stage(8, "calibrate", "contract_library_available"),
    Stage(9, "evaluate", "requires_predictions"),
    Stage(10, "interpret", "development_only"),
    Stage(11, "spatial", "requires_frozen_registry_and_truth"),
    Stage(12, "figures", "requires_official_results"),
    Stage(13, "report", "requires_official_results"),
)


def run_stage(stage: str, *, implementation_level: str = "prototype",
              execution_level: str = "smoke", validation_mode: str = "structural",
              input_path: Path | None = None, output_path: Path | None = None,
              probe_step: float | None = None, max_new_batches: int = 0) -> dict:
    """Run only a registered, currently safe stage; never auto-chain training."""
    validate_run_levels(implementation_level, execution_level)
    selected = next((item for item in STAGES if item.name == stage), None)
    if selected is None:
        raise ProtocolError(f"unknown stage: {stage}")
    if execution_level == "official":
        preflight = validation_result("official_full")
        if preflight["status"] != "pass":
            raise ProtocolError("official run blocked by strong validator")
    if selected.name == "validate":
        return validation_result(validation_mode)
    if selected.name == "service_points":
        if probe_step is None:
            raise ProtocolError("service_points requires explicit --probe-step")
        from s12_spatial.probe import probe_round
        return probe_round(probe_step, max_new_batches=max_new_batches)
    if selected.name == "features":
        if input_path is None or output_path is None:
            raise ProtocolError("features requires explicit --input-path and --output-path")
        from s03_features.build import build_feature_file
        return build_feature_file(input_path, output_path)
    raise ProtocolError(
        f"stage {selected.number:02d} {selected.name} is not auto-runnable: "
        f"{selected.implementation}; pass explicit audited inputs to its library"
    )
