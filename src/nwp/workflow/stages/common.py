"""Shared workflow-stage helpers; contains no stage handlers."""
from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import base64
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from nwp.core.config import ConfigError, assert_official_ready, to_plain
from nwp.core.artifacts import environment_packages, new_artifact_record
from nwp.core.context import RunContext
from nwp.core.dependencies import model_dependency_fingerprint
from nwp.core.lifecycle import transition
from nwp.core.hashing import content_hash, file_sha256
from nwp.core.fingerprints import (
    environment_fingerprint, model_implementation_sources, sha256_directory,
    source_files_fingerprint, stable_object_hash)
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError, StageResult
from nwp.data.contracts import DatasetRecord
from nwp.features.engineering import formal_daylight_mask
from nwp.splits.rolling import assert_gap, inner_folds, outer_folds, purge_days


STAGES = (
    "validate", "selection", "data", "features", "splits", "tuning",
    "fitting", "prediction", "calibration", "evaluation", "analysis",
    "figures", "report")
ALIASES = {
    "resolve_sites": "selection", "resolve_data": "data",
    "build_features": "features", "build_splits": "splits",
    "evaluate": "evaluation"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _final_test_evaluation_allowed(context: RunContext) -> bool:
    """Allow hold-out access only after the frozen official readiness gate."""
    config = context.config
    if (context.execution_level != "official"
            or not config.locked_config_hash
            or config.locked_config_hash != config.config_hash
            or not config.official_result_set):
        return False
    receipt_path = context.paths.meta_dir / "readiness_receipt.json"
    if not receipt_path.is_file():
        return False
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return bool(
        receipt.get("overall_ready")
        and receipt.get("scientific_run_hash") == config.config_hash)


def _stage_outputs(stage: dict[str, Any]) -> dict[str, Any]:
    return {**stage.get("artifact_outputs", {}), **stage.get("metadata_outputs", {})}


def _result(context: RunContext, status: str, *,
            dependency_fingerprint: str,
            implementation_fingerprint: str,
            inputs: dict[str, Any] | None = None,
            outputs: dict[str, Any] | None = None,
            input_hashes: dict[str, str] | None = None, message: str = "",
            started_at: str | None = None) -> StageResult:
    values = outputs or {}
    artifacts = {key: value for key, value in values.items()
                 if isinstance(value, str) and Path(value).exists()}
    metadata = {key: value for key, value in values.items() if key not in artifacts}
    output_hashes = {
        key: (file_sha256(Path(value)) if Path(value).is_file()
              else sha256_directory(Path(value)))
        for key, value in artifacts.items()
    }
    return StageResult(
        status=status, inputs=inputs or {}, artifact_outputs=artifacts,
        metadata_outputs=metadata,
        dependency_fingerprint=dependency_fingerprint,
        implementation_fingerprint=implementation_fingerprint,
        input_hashes=input_hashes or {}, started_at=started_at or _now(),
        output_hashes=output_hashes,
        finished_at=_now(), message=message)


def _stage_result(context: RunContext, status: str, **values: Any) -> StageResult:
    """Bind one handler result to its explicit stage dependency and source code."""
    stage = context.active_stage
    if stage not in STAGES:
        raise ContractError("stage result requires an active pipeline stage")
    input_hashes = values.get("input_hashes") or {}
    dependency = (next(iter(input_hashes.values())) if len(input_hashes) == 1
                  else stable_object_hash({
                      "stage": stage, "config_hash": context.config.config_hash,
                      "inputs": values.get("inputs") or {},
                      "upstream": input_hashes}))
    module = ("prepare.py" if stage in {"validate", "selection", "data", "features", "splits"}
              else "modeling.py" if stage in {"tuning", "fitting", "prediction", "calibration"}
              else "analysis.py")
    implementation = source_files_fingerprint(
        context.paths.root,
        ("src/nwp/workflow/stages/common.py", f"src/nwp/workflow/stages/{module}"))
    return _result(
        context, status, dependency_fingerprint=dependency,
        implementation_fingerprint=implementation, **values)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ContractError(f"stale temporary artifact exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_artifact_bundle(directory: Path, bundle_path: Path) -> None:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path != bundle_path:
            files[path.relative_to(directory).as_posix()] = base64.b64encode(
                path.read_bytes()).decode("ascii")
    _write_json(bundle_path, {"files": files})


def _restore_artifact_bundle(bundle_path: Path, directory: Path) -> None:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    for relative, encoded in payload.get("files", {}).items():
        target = (directory / relative).resolve()
        try:
            target.relative_to(directory.resolve())
        except ValueError as exc:
            raise ContractError("artifact bundle path escapes destination") from exc
        if target.exists():
            raise ContractError(f"artifact bundle destination exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded, validate=True))


def _model_implementation_fingerprint(context: RunContext, model_id: str) -> str:
    record = context.config.models["registry"][model_id]
    return source_files_fingerprint(context.paths.root, (
        "src/nwp/models/base.py", "src/nwp/models/factory.py",
        "src/nwp/features/preprocessing.py",
        *model_implementation_sources(record)))


def _artifact_scope(context: RunContext) -> dict[str, Any]:
    return {"profile": context.config.profile, "sites": list(context.selected_sites)}


def _find_reusable_model_artifact(
    context: RunContext, *, artifact_type: str, model_id: str,
    fold_id: str, dependency: str, implementation: str, time_scope: str,
) -> tuple[Path, Any] | None:
    return context.artifact_resolver.find_compatible_artifact(
        artifact_type=artifact_type,
        dependency_fingerprint=dependency,
        implementation_fingerprint=implementation,
        data_scope={"sites": list(context.selected_sites)},
        time_scope=time_scope,
        split_scope={"fold": fold_id},
        execution_level=context.execution_level,
        environment_fingerprint=environment_fingerprint(
            environment_packages(artifact_type,
                str(context.config.models["registry"][model_id].get("implementation")))))


def _find_current_model_artifact(
    context: RunContext, *, artifact_type: str, model_id: str,
    fold_id: str, dependency: str, implementation: str, time_scope: str,
) -> tuple[Path, Any] | None:
    return context.artifact_resolver.find_current_artifact(
        artifact_type=artifact_type,
        dependency_fingerprint=dependency,
        implementation_fingerprint=implementation,
        data_scope={"sites": list(context.selected_sites)},
        time_scope=time_scope,
        split_scope={"fold": fold_id},
        execution_level=context.execution_level,
        environment_fingerprint=environment_fingerprint(
            environment_packages(
                artifact_type,
                str(context.config.models["registry"][model_id].get(
                    "implementation")))))


def _record_model_artifact(
    context: RunContext, path: Path, *, artifact_type: str, stage: str,
    model_id: str, fold_id: str, dependency: str, implementation: str,
    time_scope: str, reuse: dict[str, Any] | None = None,
) -> None:
    relative = path.resolve().relative_to(context.paths.run_root).as_posix()
    context.artifact_resolver.append_record(new_artifact_record(
        artifact_id=stable_object_hash({"run": context.run_id, "path": relative}),
        artifact_type=artifact_type, stage=stage, model_id=model_id,
        fold_id=fold_id, scope={**_artifact_scope(context), **(reuse or {})},
        data_scope={"sites": list(context.selected_sites)},
        time_scope=time_scope, split_scope={"fold": fold_id},
        dependency_fingerprint=dependency,
        implementation_fingerprint=implementation,
        environment_fingerprint=environment_fingerprint(
            environment_packages(artifact_type,
                str(context.config.models["registry"][model_id].get("implementation")))),
        sha256=file_sha256(path), path=relative, portable=True,
        execution_level=context.execution_level,
        result_status=("official" if context.execution_level == "official" else "diagnostic"),
        official_reuse_eligible=context.execution_level == "official"))


def _months(protocol: Any) -> tuple[str, ...]:
    first = pd.Timestamp(protocol["development_period"]["start"])
    last = pd.Timestamp(protocol["test_period"]["end"])
    return tuple(period.strftime("%Y-%m") for period in
                 pd.period_range(first, last, freq="M"))


def _time_range(month: str) -> str:
    period = pd.Period(month, freq="M")
    return f"{period.start_time.date().isoformat()}/{period.end_time.date().isoformat()}"


def _verified_record(context: RunContext, record: DatasetRecord) -> DatasetRecord:
    resolved = context.catalog.resolve(
        stage=record.stage, config_hash=record.config_hash,
        sites=record.sites, time_range=record.time_range)
    if resolved is None:
        raise ContractError(f"catalog record is no longer resolvable: {record.dataset_id}")
    return resolved


def _manifest_records(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ContractError(f"dataset manifest has no records: {path}")
    return records


def _load_feature_frame(context: RunContext) -> pd.DataFrame:
    stage = context.stage_results.get("features", {})
    manifest = _stage_outputs(stage).get("feature_manifest")
    if stage.get("status") != "success" or not manifest:
        raise ContractError("feature stage has no successful manifest")
    frames = []
    for item in _manifest_records(manifest):
        record = _verified_record(context, context.catalog.record_from_dict(item))
        frames.append(pd.read_parquet(context.catalog.dataset_path(record)))
    if not frames:
        raise ContractError("feature manifest is empty")
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values(
        ["target_time_utc", "location_id", "lead_time"]).reset_index(drop=True)


def _eligible(context: RunContext, block: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(block, context.config.protocol)
    truth = pd.to_numeric(block.get("ghi_obs_sat"), errors="coerce")
    output = block.loc[mask & np.isfinite(truth)].copy()
    output["y"] = truth.loc[output.index].to_numpy(dtype=float)
    if output.empty:
        raise ContractError("workflow block has no eligible daylight truth rows")
    return output


def _window(frame: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    target = pd.to_datetime(frame.target_time_utc, utc=True)
    start = pd.Timestamp(first)
    stop = pd.Timestamp(last) + pd.Timedelta(days=1)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if stop.tzinfo is None:
        stop = stop.tz_localize("UTC")
    return frame.loc[(target >= start) & (target < stop)].copy()


def _final_blocks(context: RunContext, frame: pd.DataFrame, *,
                  require_test_truth: bool = False,
                  include_final_test: bool = False,
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = context.config.protocol
    schedule = protocol["validation"]["final_fit"]
    raw = [
        _window(frame, protocol["development_period"]["start"][:10],
                schedule["fit_end"]),
        _window(frame, schedule["early_stop_start"],
                schedule["early_stop_end"]),
        _window(frame, schedule["calibration_start"],
                schedule["calibration_end"])]
    gap = purge_days(protocol["validation"])
    assert_gap(raw[0], raw[1], gap)
    assert_gap(raw[1], raw[2], gap)
    fit, early, calibration = (
        _eligible(context, block) for block in raw[:3])
    if not include_final_test:
        return fit, early, calibration, frame.iloc[0:0].copy()
    test_source = _window(
        frame, protocol["test_period"]["start"][:10],
        protocol["test_period"]["end"][:10])
    mask = formal_daylight_mask(test_source, protocol)
    test = test_source.loc[mask].copy()
    test["y"] = pd.to_numeric(test.get("ghi_obs_sat"), errors="coerce")
    if require_test_truth:
        test = test.loc[np.isfinite(test.y)].copy()
    if test.empty:
        raise ContractError("configured final-test block has no daylight rows")
    return fit, early, calibration, test

# End of shared stage helpers.
