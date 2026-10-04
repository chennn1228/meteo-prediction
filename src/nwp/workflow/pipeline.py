"""Workflow DAG, lifecycle enforcement, reuse validation, and dispatch only."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from nwp.core.context import RunContext
from nwp.core.lifecycle import transition
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError, StageResult
from .stages.analysis import analyse, evaluate, figures, report
from .stages.modeling import calibrate, fit_models, predict, tune_models
from .stages.prepare import build_features, build_splits, resolve_data, resolve_sites, validate_config

STAGES = ("validate", "selection", "data", "features", "splits", "tuning",
          "fitting", "prediction", "calibration", "evaluation", "analysis",
          "figures", "report")
ALIASES = {"resolve_sites": "selection", "resolve_data": "data",
           "build_features": "features", "build_splits": "splits",
           "evaluate": "evaluation"}
HANDLERS: dict[str, Callable[[RunContext], StageResult]] = {
    "validate": validate_config, "selection": resolve_sites, "data": resolve_data,
    "features": build_features, "splits": build_splits, "tuning": tune_models,
    "fitting": fit_models, "prediction": predict, "calibration": calibrate,
    "evaluation": evaluate, "analysis": analyse, "figures": figures, "report": report}


def _outputs(stage: dict[str, Any]) -> dict[str, Any]:
    return {**stage.get("artifact_outputs", {}), **stage.get("metadata_outputs", {})}


def run_pipeline(context: RunContext, *, from_stage: str = "validate",
                 to_stage: str = "report") -> dict[str, dict[str, Any]]:
    from_stage, to_stage = ALIASES.get(from_stage, from_stage), ALIASES.get(to_stage, to_stage)
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}")
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise ValueError("from-stage must not follow to-stage")
    for stage in STAGES[start:end + 1]:
        previous = context.stage_results.get(stage)
        if previous and previous.get("status") in {"success", "reused", "skipped"}:
            try:
                receipt = read_receipt(context.paths.stage_receipt(stage))
                paths = [Path(value) for value in _outputs(previous).values()
                         if isinstance(value, str) and ("/" in value or "\\" in value)]
                if (receipt.get("config_hash") == context.config.config_hash
                        and receipt.get("status") == previous.get("status")
                        and all(path.exists() for path in paths)):
                    continue
            except (ContractError, OSError, ValueError):
                pass
            raise ContractError(f"saved stage {stage} is inconsistent; create a child run")
        result = HANDLERS[stage](context)
        context.record_stage(stage, result)
        extra = ({key: _outputs(result.as_dict()).get(key)
                  for key in ("trial_budget", "actual_trials", "selection_metric")}
                 if stage == "tuning" else {})
        write_receipt(context.paths.stage_receipt(stage), make_receipt(
            root=context.paths.root, stage=stage, config_hash=context.config.config_hash,
            execution_level=context.execution_level, status=result.status,
            input_hashes=result.input_hashes, output_hashes=result.output_hashes,
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
