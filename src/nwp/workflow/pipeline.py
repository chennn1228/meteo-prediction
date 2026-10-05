"""Workflow DAG, lifecycle enforcement, reuse validation, and dispatch only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nwp.core.context import RunContext
from nwp.core.fingerprints import environment_fingerprint, sha256_directory, sha256_file
from nwp.core.lifecycle import transition
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError, StageResult
from .stages.prepare import build_features, build_splits, resolve_data, resolve_sites, validate_config


def _modeling_handler(name: str, context: RunContext) -> StageResult:
    from .stages import modeling
    return getattr(modeling, name)(context)


def _analysis_handler(name: str, context: RunContext) -> StageResult:
    from .stages import analysis
    return getattr(analysis, name)(context)

STAGES = ("validate", "selection", "data", "features", "splits", "tuning",
          "fitting", "prediction", "calibration", "evaluation", "analysis",
          "figures", "report")
ALIASES = {"resolve_sites": "selection", "resolve_data": "data",
           "build_features": "features", "build_splits": "splits",
           "evaluate": "evaluation"}
HANDLERS: dict[str, Callable[[RunContext], StageResult]] = {
    "validate": validate_config, "selection": resolve_sites, "data": resolve_data,
    "features": build_features, "splits": build_splits,
    "tuning": lambda context: _modeling_handler("tune_models", context),
    "fitting": lambda context: _modeling_handler("fit_models", context),
    "prediction": lambda context: _modeling_handler("predict", context),
    "calibration": lambda context: _modeling_handler("calibrate", context),
    "evaluation": lambda context: _analysis_handler("evaluate", context),
    "analysis": lambda context: _analysis_handler("analyse", context),
    "figures": lambda context: _analysis_handler("figures", context),
    "report": lambda context: _analysis_handler("report", context)}

_STAGE_PACKAGES = (
    "numpy", "pandas", "pyarrow", "pvlib", "scikit-learn", "lightgbm",
    "xgboost", "torch", "matplotlib")


def _stage_environment_fingerprint() -> str:
    return environment_fingerprint(_STAGE_PACKAGES)


def _outputs(stage: dict[str, Any]) -> dict[str, Any]:
    return {**stage.get("artifact_outputs", {}), **stage.get("metadata_outputs", {})}


def _actual_output_hashes(stage: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, value in stage.get("artifact_outputs", {}).items():
        path = Path(value)
        if not path.exists():
            raise ContractError(f"saved stage output is missing: {path}")
        hashes[key] = sha256_file(path) if path.is_file() else sha256_directory(path)
    return hashes


def _verify_saved_stage(context: RunContext, stage: str,
                        previous: dict[str, Any]) -> None:
    try:
        receipt = read_receipt(context.paths.stage_receipt(stage))
        valid = (
            receipt.get("config_hash") == context.config.config_hash
            and receipt.get("status") == previous.get("status")
            and receipt.get("dependency_fingerprint") == previous.get("dependency_fingerprint")
            and receipt.get("implementation_fingerprint") == previous.get("implementation_fingerprint")
            and receipt.get("environment_fingerprint") == _stage_environment_fingerprint()
            and receipt.get("input_hashes") == previous.get("input_hashes")
            and receipt.get("output_hashes") == previous.get("output_hashes")
            and _actual_output_hashes(previous) == previous.get("output_hashes"))
        if valid:
            return
    except (ContractError, OSError, ValueError):
        pass
    raise ContractError(f"saved stage {stage} is inconsistent; create a child run")


def run_pipeline(context: RunContext, *, from_stage: str = "validate",
                 to_stage: str = "report") -> dict[str, dict[str, Any]]:
    from_stage, to_stage = ALIASES.get(from_stage, from_stage), ALIASES.get(to_stage, to_stage)
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}")
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise ValueError("from-stage must not follow to-stage")
    for stage in STAGES[:start]:
        previous = context.stage_results.get(stage)
        if previous and previous.get("status") in {"success", "reused", "skipped"}:
            _verify_saved_stage(context, stage, previous)
    for stage in STAGES[start:end + 1]:
        previous = context.stage_results.get(stage)
        if previous and previous.get("status") in {"success", "reused", "skipped"}:
            _verify_saved_stage(context, stage, previous)
            continue
        context.active_stage = stage
        try:
            result = HANDLERS[stage](context)
        finally:
            context.active_stage = None
        context.record_stage(stage, result)
        extra = ({key: _outputs(result.as_dict()).get(key)
                  for key in ("trial_budget", "actual_trials", "selection_metric")}
                 if stage == "tuning" else {})
        write_receipt(context.paths.stage_receipt(stage), make_receipt(
            root=context.paths.root, stage=stage, config_hash=context.config.config_hash,
            execution_level=context.execution_level, status=result.status,
            input_hashes=result.input_hashes, output_hashes=result.output_hashes,
            dependency_fingerprint=result.dependency_fingerprint,
            implementation_fingerprint=result.implementation_fingerprint,
            environment_fingerprint=_stage_environment_fingerprint(),
            message=result.message, **extra))
        if result.status in {"blocked", "failed"}:
            target = "BLOCKED" if result.status == "blocked" else "FAILED"
            transition(context.paths.meta_dir / "status.json", target)
            context.lifecycle_state = target
            break
    selected = STAGES[start:end + 1]
    if to_stage == "report" and all(context.stage_results.get(stage, {}).get("status")
                                    in {"success", "reused", "skipped"} for stage in selected):
        transition(context.paths.meta_dir / "status.json", "COMPLETE")
        context.lifecycle_state = "COMPLETE"
    return context.stage_results
