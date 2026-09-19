"""Stage registry; no training or official execution is implicit."""
from __future__ import annotations

from dataclasses import dataclass

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
    Stage(6, "tuning", "trial_library_available"),
    Stage(7, "train", "blocked_until_validated_models"),
    Stage(8, "calibrate", "contract_library_available"),
    Stage(9, "evaluate", "requires_predictions"),
    Stage(10, "interpret", "development_only"),
    Stage(11, "spatial", "requires_frozen_registry_and_truth"),
    Stage(12, "figures", "requires_official_results"),
    Stage(13, "report", "requires_official_results"),
)


def run_stage(stage: str, *, implementation_level: str = "prototype",
              execution_level: str = "smoke") -> dict:
    """Run only a registered, currently safe stage; never auto-chain training."""
    validate_run_levels(implementation_level, execution_level)
    selected = next((item for item in STAGES if item.name == stage), None)
    if selected is None:
        raise ProtocolError(f"unknown stage: {stage}")
    if execution_level == "official":
        preflight = validation_result("official")
        if preflight["status"] != "pass":
            raise ProtocolError("official run blocked by strong validator")
    if selected.name == "validate":
        return validation_result("structural")
    raise ProtocolError(
        f"stage {selected.number:02d} {selected.name} is not auto-runnable: "
        f"{selected.implementation}; pass explicit audited inputs to its library"
    )
