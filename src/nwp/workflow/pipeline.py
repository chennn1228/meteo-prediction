"""The sole executable, dependency-hashed research workflow DAG."""
from __future__ import annotations

from dataclasses import asdict
import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from nwp.core.config import ConfigError, assert_official_ready, to_plain
from nwp.core.context import RunContext
from nwp.core.hashing import content_hash, file_sha256
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError, StageResult
from nwp.data.contracts import DatasetRecord
from nwp.evaluation.grouped import evaluate_predictions, write_evaluation_report
from nwp.evaluation.interpretation import group_mechanism_evidence, model_summary
from nwp.experiment.calibration import CausalIssueQuantileCalibrator
from nwp.experiment.fitting import (
    fit_final_quantile_model, fit_model, fit_outer_quantile_model)
from nwp.experiment.prediction import (
    quantile_columns, quantiles, read_predictions, write_predictions)
from nwp.experiment.tuning import IncompleteTrialsError, candidates, run_trials
from nwp.features.build import build_feature_month
from nwp.features.engineering import formal_daylight_mask
from nwp.models.base import BaseModel
from nwp.models.statistical import ridge_fit_predict as _ridge_fit_predict
from nwp.models.trees import tree_fit_predict
from nwp.splits.diagnostics import split_specs
from nwp.splits.rolling import assert_gap, inner_folds, outer_folds
from nwp.visualization.figures import generate_run_figures, write_figure_index


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


def _result(context: RunContext, status: str, *,
            inputs: dict[str, Any] | None = None,
            outputs: dict[str, Any] | None = None,
            input_hashes: dict[str, str] | None = None, message: str = "",
            started_at: str | None = None) -> StageResult:
    return StageResult(
        status=status, inputs=inputs or {}, outputs=outputs or {},
        config_hash=context.config.config_hash,
        input_hashes=input_hashes or {}, started_at=started_at or _now(),
        finished_at=_now(), message=message)


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
    manifest = stage.get("outputs", {}).get("feature_manifest")
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
                  require_test_truth: bool = False
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = context.config.protocol
    schedule = protocol["validation"]["final_fit"]
    raw = (
        _window(frame, protocol["development_period"]["start"][:10],
                schedule["fit_end"]),
        _window(frame, schedule["early_stop_start"],
                schedule["early_stop_end"]),
        _window(frame, schedule["calibration_start"],
                schedule["calibration_end"]),
        _window(frame, protocol["test_period"]["start"][:10],
                protocol["test_period"]["end"][:10]))
    gap = int(protocol["validation"]["gap_days"])
    assert_gap(raw[0], raw[1], gap)
    assert_gap(raw[1], raw[2], gap)
    fit, early, calibration = (
        _eligible(context, block) for block in raw[:3])
    mask = formal_daylight_mask(raw[3], protocol)
    test = raw[3].loc[mask].copy()
    test["y"] = pd.to_numeric(test.get("ghi_obs_sat"), errors="coerce")
    if require_test_truth:
        test = test.loc[np.isfinite(test.y)].copy()
    if test.empty:
        raise ContractError("configured final-test block has no daylight rows")
    return fit, early, calibration, test


def validate_config(context: RunContext) -> StageResult:
    started = _now()
    try:
        assert_official_ready(
            context.config,
            readiness_passed=context.config.execution != "official")
    except ConfigError as exc:
        return _result(context, "blocked", inputs={"profile": context.config.profile},
                       message=str(exc), started_at=started)
    return _result(
        context, "success", inputs={"profile": context.config.profile},
        outputs={"config_hash": context.config.config_hash}, started_at=started)


def resolve_sites(context: RunContext) -> StageResult:
    started = _now()
    path = context.paths.stage_dir("selection") / "sites.json"
    registry = context.config.sites["registry"]
    payload = [{"site_id": site_id, **to_plain(registry[site_id])}
               for site_id in context.selected_sites]
    _write_json(path, payload)
    return _result(
        context, "success", outputs={"sites": str(path), "count": len(payload)},
        input_hashes={"site_selection": content_hash(payload)}, started_at=started)


def resolve_data(context: RunContext) -> StageResult:
    """Resolve one verified clean catalog partition for every site and month."""
    started = _now()
    months = _months(context.config.protocol)
    records = context.catalog.records()
    chosen, missing, ambiguous = [], [], []
    for site_id in context.selected_sites:
        for month in months:
            matches = [record for record in records
                       if record.stage == "clean" and record.status == "ready"
                       and record.sites == (site_id,)
                       and record.time_range == _time_range(month)]
            valid = []
            for record in matches:
                try:
                    record = _verified_record(context, record)
                    receipt = read_receipt(context.catalog.receipt_path(record))
                    if receipt.get("data_version") in {
                            None, context.config.data["version"]}:
                        valid.append(record)
                except (ContractError, OSError, ValueError):
                    continue
            if len(valid) == 1:
                chosen.append(valid[0])
            elif not valid:
                missing.append(f"{site_id}/{month}")
            else:
                ambiguous.append(f"{site_id}/{month}")
    path = context.paths.meta_dir / "clean_manifest.json"
    payload = {
        "stage": "clean", "sites": list(context.selected_sites),
        "months": list(months), "records": [record.as_dict() for record in chosen],
        "missing": missing, "ambiguous": ambiguous}
    _write_json(path, payload)
    dependency = content_hash({
        "data_config": context.config.data,
        "records": [record.source_hashes for record in chosen]})
    if missing or ambiguous:
        return _result(
            context, "blocked", inputs={"missing": missing, "ambiguous": ambiguous},
            outputs={"clean_manifest": str(path)},
            input_hashes={"data": dependency},
            message="clean catalog coverage is incomplete or ambiguous",
            started_at=started)
    return _result(
        context, "success", outputs={"clean_manifest": str(path),
                                     "partitions": len(chosen)},
        input_hashes={"data": dependency}, started_at=started)


def build_features(context: RunContext) -> StageResult:
    """Build or reuse every clean-to-feature monthly catalog partition."""
    started = _now()
    data_stage = context.stage_results.get("data", {})
    manifest = data_stage.get("outputs", {}).get("clean_manifest")
    if data_stage.get("status") != "success" or not manifest:
        return _result(context, "blocked",
                       message="features require complete clean catalog coverage",
                       started_at=started)
    records = []
    for item in _manifest_records(manifest):
        clean = context.catalog.record_from_dict(item)
        month = clean.time_range[:7]
        record = build_feature_month(
            site_id=clean.sites[0], month=month,
            clean_config_hash=clean.config_hash,
            data_config=context.config.data,
            feature_config=context.config.features,
            paths=context.paths, catalog=context.catalog,
            execution_level=context.execution_level)
        records.append(record)
    path = context.paths.meta_dir / "feature_manifest.json"
    _write_json(path, {"stage": "features",
                       "records": [record.as_dict() for record in records]})
    hashes = {record.dataset_id: file_sha256(
        context.catalog.dataset_path(record)) for record in records}
    dependency = content_hash({
        "clean": data_stage["input_hashes"]["data"],
        "features": context.config.features, "outputs": hashes})
    return _result(
        context, "success", outputs={"feature_manifest": str(path),
                                     "partitions": len(records)},
        input_hashes={"features": dependency}, started_at=started)


def build_splits(context: RunContext) -> StageResult:
    """Materialize real outer/inner fold counts for the resolved feature data."""
    started = _now()
    if context.stage_results.get("features", {}).get("status") != "success":
        return _result(context, "blocked",
                       message="splits require materialized feature partitions",
                       started_at=started)
    frame = _load_feature_frame(context)
    validation = context.config.protocol["validation"]
    development = context.config.protocol["development_period"]
    gap = int(validation["gap_days"])
    outer = outer_folds(frame, validation, development, gap_days=gap)
    payload = {"specification": split_specs(validation), "outer_folds": []}
    for fold in outer:
        inner = inner_folds(fold, validation, development, gap_days=gap)
        payload["outer_folds"].append({
            "fold_id": fold.fold_id, "fit_rows": len(fold.fit),
            "score_rows": len(fold.score),
            "inner": [{"fold_id": item.fold_id, "fit_rows": len(item.fit),
                       "early_stop_rows": len(item.early_stop),
                       "score_rows": len(item.score)} for item in inner]})
    path = context.paths.stage_dir("splits") / "splits.json"
    _write_json(path, payload)
    dependency = content_hash({
        "features": context.stage_results["features"]["input_hashes"]["features"],
        "sites": context.selected_sites, "validation": validation})
    return _result(context, "success", outputs={"splits": str(path)},
                   input_hashes={"splits": dependency}, started_at=started)


def tune_models(context: RunContext) -> StageResult:
    """Run the registered equal-budget selector on every outer prefix."""
    started = _now()
    if context.stage_results.get("splits", {}).get("status") != "success":
        return _result(context, "blocked", message="tuning requires real splits",
                       started_at=started)
    if not context.allow_model_execution:
        return _result(
            context, "blocked",
            message="tuning is gated; pass --execute-model-stages explicitly",
            started_at=started)
    frame = _load_feature_frame(context)
    protocol, models = context.config.protocol, context.config.models
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    selected: dict[str, Any] = {}
    try:
        for outer in outer_folds(frame, validation, development, gap_days=gap):
            folds = inner_folds(outer, validation, development, gap_days=gap)
            eligible = [type(fold)(
                fold.fold_id, _eligible(context, fold.fit),
                _eligible(context, fold.early_stop), _eligible(context, fold.score),
                fold.fit_end, fold.early_stop_start, fold.early_stop_end,
                fold.score_start) for fold in folds]
            for model_id in context.selected_models:
                entry = models["registry"][model_id]
                if not entry["tuning"]:
                    continue
                adapter_name = entry.get("quantile_adapter")
                if adapter_name == "ridge":
                    def adapter(parameters, fit, early, score, seed, levels):
                        return _ridge_fit_predict(
                            parameters, fit, early, score, seed, levels,
                            feature_config=context.config.features)
                    tree_receipts = []
                elif adapter_name == "tree":
                    adapter = tree_fit_predict(
                        str(entry["algorithm"]),
                        feature_config=context.config.features)
                    tree_receipts = adapter.receipts
                else:
                    raise ContractError(
                        f"workflow tuning adapter is not registered for {model_id}")
                chosen, ledger = run_trials(
                    model_id, outer.fold_id, eligible, adapter,
                    model_config=models, protocol_config=protocol,
                    device=str(entry["device"]), gap_days=gap)
                selected.setdefault(outer.fold_id, {})[model_id] = {
                    "selected_candidate": chosen,
                    "parameters": candidates(model_id, models)[chosen],
                    "trials": [asdict(trial) for trial in ledger],
                    "tree_round_receipts": list(tree_receipts)}
    except IncompleteTrialsError as exc:
        return _result(
            context, "failed", inputs={"ledger": [asdict(row) for row in exc.ledger]},
            message=str(exc), started_at=started)
    final_selection: dict[str, Any] = {}
    for model_id in context.selected_models:
        if not models["registry"][model_id]["tuning"]:
            continue
        candidate_scores: dict[int, list[float]] = {
            index: [] for index in range(len(candidates(model_id, models)))}
        for outer_models in selected.values():
            for trial in outer_models[model_id]["trials"]:
                if trial["status"] == "ok" and trial["mean_pinball"] is not None:
                    candidate_scores[int(trial["candidate"])].append(
                        float(trial["mean_pinball"]))
        expected = (len(selected)
                    * int(protocol["validation"]["inner_folds"]))
        if any(len(values) != expected for values in candidate_scores.values()):
            raise ContractError(
                f"{model_id}: final selection lacks the common outer-inner score grid")
        means = {candidate: float(np.mean(values))
                 for candidate, values in candidate_scores.items()}
        chosen = min(means, key=lambda candidate: (means[candidate], candidate))
        final_selection[model_id] = {
            "selected_candidate": chosen,
            "parameters": candidates(model_id, models)[chosen],
            "mean_pinball_by_candidate": means,
            "score_count_per_candidate": expected}
    path = context.paths.tuning_dir / "selection.json"
    _write_json(path, {"outer": selected, "final": final_selection})
    dependency = content_hash({
        "features": context.stage_results["features"]["input_hashes"]["features"],
        "splits": context.stage_results["splits"]["input_hashes"]["splits"],
        "models": models, "selected_models": context.selected_models})
    return _result(context, "success", outputs={"selection": str(path)},
                   input_hashes={"tuning": dependency}, started_at=started)


def fit_models(context: RunContext) -> StageResult:
    """Fit and serialize every selected outer-fold model artifact."""
    started = _now()
    tuning = context.stage_results.get("tuning", {})
    if tuning.get("status") != "success":
        return _result(context, "blocked",
                       message="fitting requires successful tuning",
                       started_at=started)
    frame = _load_feature_frame(context)
    protocol = context.config.protocol
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    selection = json.loads(Path(
        tuning["outputs"]["selection"]).read_text(encoding="utf-8"))
    outer_selection = selection["outer"]
    artifacts = []
    for outer in outer_folds(frame, validation, development, gap_days=gap):
        fit = _eligible(context, outer.fit)
        folds = inner_folds(outer, validation, development, gap_days=gap)
        eligible = [type(fold)(
            fold.fold_id, _eligible(context, fold.fit),
            _eligible(context, fold.early_stop), _eligible(context, fold.score),
            fold.fit_end, fold.early_stop_start, fold.early_stop_end,
            fold.score_start) for fold in folds]
        for model_id in context.selected_models:
            entry = context.config.models["registry"][model_id]
            if entry["tuning"]:
                choice = outer_selection[outer.fold_id][model_id]
                model = fit_outer_quantile_model(
                    context, model_id, fit, eligible, choice["parameters"],
                    tree_receipts=choice["tree_round_receipts"])
            else:
                model = fit_model(context, model_id, fit, fit["y"].to_numpy())
            path = context.paths.model_fold(model_id, outer.fold_id) / "model.pkl"
            model.save(path)
            artifacts.append({
                "model_id": model_id, "outer_fold": outer.fold_id,
                "fit_scope": "outer", "path": str(path),
                "sha256": file_sha256(path)})
    final_fit, final_early, final_calibration, _ = _final_blocks(
        context, frame)
    final_history = pd.concat(
        [final_early, final_calibration], ignore_index=True)
    for model_id in context.selected_models:
        entry = context.config.models["registry"][model_id]
        if entry["tuning"]:
            choice = selection["final"][model_id]
            model = fit_final_quantile_model(
                context, model_id, final_fit, final_early,
                choice["parameters"])
        else:
            model = fit_model(
                context, model_id,
                {"fit": final_fit, "history": final_history})
        path = context.paths.model_fold(model_id, "final") / "model.pkl"
        model.save(path)
        artifacts.append({
            "model_id": model_id, "outer_fold": "final",
            "fit_scope": "final", "path": str(path),
            "sha256": file_sha256(path)})
    path = context.paths.models_dir / "model_index.json"
    _write_json(path, {"models": artifacts})
    dependency = content_hash({
        "features": context.stage_results["features"]["input_hashes"]["features"],
        "tuning": tuning["input_hashes"]["tuning"],
        "model_config": context.config.models})
    return _result(context, "success", outputs={"model_index": str(path)},
                   input_hashes={"fitting": dependency}, started_at=started)


def _prediction_rows(context: RunContext, score: pd.DataFrame, model_id: str,
                     fold_id: str, values: np.ndarray,
                     is_quantile: bool, *, inner_id: str = "outer_refit") -> pd.DataFrame:
    columns = quantile_columns(context.config.protocol)
    median_index = quantiles(context.config.protocol).index(0.5)
    output = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": (
            "validated" if context.config.models["registry"][model_id]
            ["implementation_status"] == "validated" else "prototype"),
        "execution_level": context.execution_level,
        "prediction_type": "quantile" if is_quantile else "point",
        "location_id": score["location_id"].to_numpy(),
        "requested_coordinates": list(zip(
            score.requested_latitude, score.requested_longitude)),
        "gfs_service_coordinates": list(zip(
            score.gfs_service_latitude, score.gfs_service_longitude)),
        "truth_service_coordinates": list(zip(
            score.himawari_service_latitude, score.himawari_service_longitude)),
        "target_time_utc": score.target_time_utc.to_numpy(),
        "forecast_issue_time_utc": score.forecast_issue_time_utc.to_numpy(),
        "lead_time": score.lead_time.to_numpy(), "outer_fold": fold_id,
        "inner_fold": inner_id, "seed": int(
            context.config.protocol["seed_policy"]["tuning_seed"]),
        "y": score.y.to_numpy(dtype=float),
        "point_prediction": values[:, median_index] if is_quantile else values,
        "data_version": context.config.data["version"],
        "feature_version": context.config.features["version"],
        "protocol_revision": context.config.protocol_version,
        "experiment_id": context.run_id,
        "result_status": (
            "official" if context.execution_level == "official" else "diagnostic"),
    })
    for index, column in enumerate(columns):
        output[column] = values[:, index] if is_quantile else np.nan
    return output


def predict(context: RunContext) -> StageResult:
    """Load fitted artifacts and write canonical outer-fold predictions."""
    started = _now()
    fitting = context.stage_results.get("fitting", {})
    if fitting.get("status") != "success":
        return _result(context, "blocked",
                       message="prediction requires fitted model artifacts",
                       started_at=started)
    index = json.loads(Path(
        fitting["outputs"]["model_index"]).read_text(encoding="utf-8"))
    by_key = {(item["outer_fold"], item["model_id"]): item
              for item in index["models"]}
    frame = _load_feature_frame(context)
    protocol = context.config.protocol
    validation, development = protocol["validation"], protocol["development_period"]
    gap = int(validation["gap_days"])
    outer_outputs, final_calibration_outputs, final_test_outputs = [], [], []
    levels = quantiles(protocol)
    for outer in outer_folds(frame, validation, development, gap_days=gap):
        score = _eligible(context, outer.score)
        for model_id in context.selected_models:
            item = by_key[(outer.fold_id, model_id)]
            if file_sha256(Path(item["path"])) != item["sha256"]:
                raise ContractError("serialized model hash changed before prediction")
            model = BaseModel.load(Path(item["path"]))
            is_quantile = bool(context.config.models["registry"][model_id]["tuning"])
            values = (np.asarray(model.predict_quantiles(score, levels), dtype=float)
                      if is_quantile else np.asarray(model.predict(score), dtype=float))
            rows = _prediction_rows(
                context, score, model_id, outer.fold_id, values, is_quantile)
            path = context.paths.prediction_file(model_id, outer.fold_id)
            write_predictions(rows, path, protocol)
            outer_outputs.append({
                "model_id": model_id, "outer_fold": outer.fold_id,
                "path": str(path), "sha256": file_sha256(path)})
    _, _, final_calibration, final_test = _final_blocks(context, frame)
    for model_id in context.selected_models:
        item = by_key[("final", model_id)]
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ContractError("serialized final model hash changed before prediction")
        model = BaseModel.load(path)
        is_quantile = bool(
            context.config.models["registry"][model_id]["tuning"])
        if is_quantile:
            calibration_values = np.asarray(
                model.predict_quantiles(final_calibration, levels), dtype=float)
            calibration_rows = _prediction_rows(
                context, final_calibration, model_id, "final_calibration",
                calibration_values, True, inner_id="final_fit")
            calibration_path = context.paths.prediction_file(
                model_id, "final_calibration")
            write_predictions(
                calibration_rows, calibration_path, protocol,
                require_truth=True)
            final_calibration_outputs.append({
                "model_id": model_id, "path": str(calibration_path),
                "sha256": file_sha256(calibration_path)})
            test_values = np.asarray(
                model.predict_quantiles(final_test, levels), dtype=float)
        else:
            test_values = np.asarray(model.predict(final_test), dtype=float)
        test_rows = _prediction_rows(
            context, final_test, model_id, "final_test", test_values,
            is_quantile, inner_id="final_fit")
        test_path = context.paths.prediction_file(
            model_id, "final_test_uncalibrated")
        write_predictions(
            test_rows, test_path, protocol, require_truth=False)
        final_test_outputs.append({
            "model_id": model_id, "path": str(test_path),
            "sha256": file_sha256(test_path)})
    index_path = context.paths.predictions_dir / "prediction_index.json"
    payload = {
        "outer_predictions": outer_outputs,
        "final_calibration_predictions": final_calibration_outputs,
        "final_test_uncalibrated": final_test_outputs}
    _write_json(index_path, payload)
    dependency = content_hash({
        "fitting": fitting["input_hashes"]["fitting"], "artifacts": payload})
    return _result(context, "success",
                   outputs={"prediction_index": str(index_path)},
                   input_hashes={"prediction": dependency}, started_at=started)


def calibrate(context: RunContext) -> StageResult:
    started = _now()
    prediction = context.stage_results.get("prediction", {})
    if prediction.get("status") != "success":
        return _result(context, "blocked",
                       message="calibration requires canonical predictions",
                       started_at=started)
    prediction_index = Path(prediction["outputs"]["prediction_index"])
    source = json.loads(prediction_index.read_text(encoding="utf-8"))
    calibration_by_model = {
        item["model_id"]: item
        for item in source.get("final_calibration_predictions", [])}
    test_by_model = {
        item["model_id"]: item
        for item in source.get("final_test_uncalibrated", [])}
    schedule = context.config.protocol["validation"]["final_fit"]
    fit_end = pd.Timestamp(schedule["fit_end"], tz="UTC") + pd.Timedelta(hours=23)
    early_end = (pd.Timestamp(schedule["early_stop_end"], tz="UTC")
                 + pd.Timedelta(hours=23))
    artifacts = []
    for model_id, calibration_item in sorted(calibration_by_model.items()):
        test_item = test_by_model.get(model_id)
        if test_item is None:
            raise ContractError(
                f"missing final-test predictions for calibration model {model_id}")
        calibration_path, test_path = (
            Path(calibration_item["path"]), Path(test_item["path"]))
        if (file_sha256(calibration_path) != calibration_item["sha256"]
                or file_sha256(test_path) != test_item["sha256"]):
            raise ContractError(
                "final prediction hash changed before calibration")
        calibration_rows = read_predictions(
            calibration_path, context.config.protocol)
        test_rows = read_predictions(
            test_path, context.config.protocol, require_truth=False)
        calibrated = CausalIssueQuantileCalibrator(
            context.config.protocol).fit(
                calibration_rows, fit_end=fit_end,
                early_stop_end=early_end).apply(test_rows)
        path = context.paths.calibration_file(model_id, "final_test")
        write_predictions(
            calibrated, path, context.config.protocol, require_truth=False)
        artifacts.append({
            "model_id": model_id, "path": str(path),
            "sha256": file_sha256(path)})
    path = context.paths.calibration_dir / "calibration_index.json"
    payload = {
        "method": context.config.protocol["probability"]["calibration"]["primary"],
        "chronology": context.config.protocol["probability"]["calibration"]["chronology"],
        "prediction_index": str(prediction_index),
        "calibrated_final_test": artifacts,
        "outer_prediction_policy": "uncalibrated_development_evaluation"}
    _write_json(path, payload)
    dependency = content_hash({
        "prediction": prediction["input_hashes"]["prediction"],
        "calibration": context.config.protocol["probability"]["calibration"]})
    status = "success" if artifacts else "skipped"
    return _result(context, status, outputs={"calibration_index": str(path)},
                   input_hashes={"calibration": dependency}, started_at=started)


def evaluate(context: RunContext) -> StageResult:
    started = _now()
    calibration = context.stage_results.get("calibration", {})
    if calibration.get("status") not in {"success", "skipped", "reused"}:
        return _result(context, "blocked",
                       message="evaluation requires prediction/calibration state",
                       started_at=started)
    prediction = context.stage_results.get("prediction", {})
    index_path = Path(prediction["outputs"]["prediction_index"])
    index = json.loads(index_path.read_text(encoding="utf-8"))
    artifact_hashes = []
    def load_items(items: list[dict[str, Any]], *, require_truth: bool) -> pd.DataFrame:
        frames = []
        for item in items:
            path = Path(item["path"])
            digest = file_sha256(path)
            if digest != item["sha256"]:
                raise ContractError(
                    f"prediction hash changed before evaluation: {path}")
            frames.append(read_predictions(
                path, context.config.protocol, require_truth=require_truth))
            artifact_hashes.append(digest)
        return (pd.concat(frames, ignore_index=True)
                if frames else pd.DataFrame())

    combined = load_items(index.get("outer_predictions", []), require_truth=True)
    if combined.empty:
        raise ContractError("prediction index contains no evaluation artifacts")
    combined["season"] = pd.to_datetime(
        combined.target_time_utc, utc=True).dt.month.map({
            12: "winter", 1: "winter", 2: "winter",
            3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer",
            9: "autumn", 10: "autumn", 11: "autumn"})
    report = evaluate_predictions(
        combined, context.config.protocol, context.config.models,
        formal=context.execution_level == "official")
    metrics_index = write_evaluation_report(report, context.paths.metrics_dir)
    root_metrics = json.loads(metrics_index.read_text(encoding="utf-8"))
    final_indexes = {}
    final_raw = load_items(
        index.get("final_test_uncalibrated", []), require_truth=False)
    if not final_raw.empty:
        final_raw = final_raw.loc[np.isfinite(final_raw.y)].copy()
        if not final_raw.empty:
            final_raw["season"] = pd.to_datetime(
                final_raw.target_time_utc, utc=True).dt.month.map({
                    12: "winter", 1: "winter", 2: "winter",
                    3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer",
                    9: "autumn", 10: "autumn", 11: "autumn"})
            final_report = evaluate_predictions(
                final_raw, context.config.protocol, context.config.models,
                formal=context.execution_level == "official")
            final_indexes["uncalibrated"] = str(write_evaluation_report(
                final_report, context.paths.metrics_dir / "final_uncalibrated"))
    calibration_index = Path(
        calibration["outputs"]["calibration_index"])
    calibrated_payload = json.loads(
        calibration_index.read_text(encoding="utf-8"))
    final_calibrated = load_items(
        calibrated_payload.get("calibrated_final_test", []),
        require_truth=False)
    if not final_calibrated.empty:
        final_calibrated = final_calibrated.loc[
            np.isfinite(final_calibrated.y)].copy()
        if not final_calibrated.empty:
            # Calibration artifacts contain quantile models only.  Carry the
            # unchanged point baselines into this evaluation view so formal
            # RMSE-skill comparisons retain their preregistered references.
            if not final_raw.empty:
                point_references = final_raw.loc[
                    final_raw.prediction_type.eq("point")].copy()
                if not point_references.empty:
                    final_calibrated = pd.concat(
                        [final_calibrated, point_references],
                        ignore_index=True)
            final_calibrated["season"] = pd.to_datetime(
                final_calibrated.target_time_utc, utc=True).dt.month.map({
                    12: "winter", 1: "winter", 2: "winter",
                    3: "spring", 4: "spring", 5: "spring",
                    6: "summer", 7: "summer", 8: "summer",
                    9: "autumn", 10: "autumn", 11: "autumn"})
            calibrated_report = evaluate_predictions(
                final_calibrated, context.config.protocol,
                context.config.models,
                formal=context.execution_level == "official")
            final_indexes["calibrated"] = str(write_evaluation_report(
                calibrated_report,
                context.paths.metrics_dir / "final_calibrated"))
    root_metrics["final"] = final_indexes
    _write_json(metrics_index, root_metrics)
    dependency = content_hash({
        "prediction_hashes": artifact_hashes,
        "evaluation": context.config.protocol["evaluation"]})
    return _result(
        context, "success", inputs={"prediction_index": str(index_path)},
        outputs={"metrics_index": str(metrics_index)},
        input_hashes={"evaluation": dependency}, started_at=started)


def analyse(context: RunContext) -> StageResult:
    started = _now()
    if context.stage_results.get("evaluation", {}).get("status") != "success":
        return _result(context, "blocked",
                       message="analysis requires evaluated predictions",
                       started_at=started)
    metrics_index = Path(
        context.stage_results["evaluation"]["outputs"]["metrics_index"])
    metrics = json.loads(metrics_index.read_text(encoding="utf-8"))
    probability = pd.read_csv(metrics["overview"]["probability_primary"])
    point = pd.read_csv(metrics["overview"]["point_secondary"])
    summary = model_summary(probability, point)
    summary_path = context.paths.analysis_dir / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    index_path = context.paths.analysis_dir / "analysis_index.json"
    tuning_trials_path = None
    ablation_path = permutation_path = None
    tuning = context.stage_results.get("tuning", {})
    if tuning.get("status") == "success":
        selection = json.loads(Path(
            tuning["outputs"]["selection"]).read_text(encoding="utf-8"))
        rows = []
        for outer_id, models in selection["outer"].items():
            for model_id, detail in models.items():
                for trial in detail.get("trials", []):
                    rows.append({"outer_fold": outer_id, "model_id": model_id,
                                 **trial})
        if rows:
            tuning_trials_path = context.paths.analysis_dir / "tuning_trials.csv"
            pd.DataFrame(rows).to_csv(tuning_trials_path, index=False)
        evidence = set(context.config.features["analysis_evidence"]["selection"])
        reference = context.config.features["analysis_evidence"]["reference_model"]
        if ({"group_ablation", "grouped_permutation_importance"} <= evidence
                and reference in context.selected_models):
            ablation, permutation = group_mechanism_evidence(
                _load_feature_frame(context), selection,
                protocol_config=context.config.protocol,
                feature_config=context.config.features,
                model_config=context.config.models,
                eligible=lambda block: _eligible(context, block))
            ablation_path = context.paths.analysis_dir / "group_ablation.csv"
            permutation_path = (
                context.paths.analysis_dir / "grouped_permutation.csv")
            ablation.to_csv(ablation_path, index=False)
            permutation.to_csv(permutation_path, index=False)
    _write_json(index_path, {
        "model_summary": str(summary_path),
        "tuning_trials": (
            None if tuning_trials_path is None else str(tuning_trials_path)),
        "group_ablation": (
            None if ablation_path is None else str(ablation_path)),
        "grouped_permutation": (
            None if permutation_path is None else str(permutation_path)),
        "source_metrics_index": str(metrics_index),
        "selection_rule": context.config.protocol["probability"]["selection_metric"],
        "interpretation_role": "downstream_only_not_model_selection"})
    dependency = content_hash({
        "evaluation": context.stage_results["evaluation"]["input_hashes"]["evaluation"],
        "analysis": "model_summary_v1"})
    return _result(
        context, "success", outputs={"analysis_index": str(index_path)},
        input_hashes={"analysis": dependency}, started_at=started)


def figures(context: RunContext) -> StageResult:
    started = _now()
    if context.stage_results.get("analysis", {}).get("status") != "success":
        return _result(context, "blocked",
                       message="figures require analysis source tables",
                       started_at=started)
    directory = context.paths.figures_dir
    metrics_index = Path(
        context.stage_results["evaluation"]["outputs"]["metrics_index"])
    analysis_index = Path(
        context.stage_results["analysis"]["outputs"]["analysis_index"])
    prediction_index = Path(
        context.stage_results["prediction"]["outputs"]["prediction_index"])
    status = "official" if context.execution_level == "official" else "diagnostic"
    entries = generate_run_figures(
        metrics_index, analysis_index, directory, result_status=status,
        protocol_config=to_plain(context.config.protocol),
        prediction_index=prediction_index)
    index = directory / "figure_index.json"
    write_figure_index(index, entries)
    dependency = content_hash({
        "analysis": context.stage_results["analysis"]["input_hashes"]["analysis"],
        "figure_style": "publication_v1"})
    return _result(
        context, "success",
        outputs={"figure_index": str(index), "figure_count": len(entries)},
        input_hashes={"figures": dependency}, started_at=started)


def report(context: RunContext) -> StageResult:
    started = _now()
    ready = context.stage_results.get("figures", {}).get("status") == "success"
    return _result(context, "success" if ready else "blocked",
                   message="" if ready else "report requires generated figures",
                   started_at=started)


HANDLERS: dict[str, Callable[[RunContext], StageResult]] = {
    "validate": validate_config, "selection": resolve_sites,
    "data": resolve_data, "features": build_features, "splits": build_splits,
    "tuning": tune_models, "fitting": fit_models, "prediction": predict,
    "calibration": calibrate, "evaluation": evaluate, "analysis": analyse,
    "figures": figures, "report": report}


def run_pipeline(context: RunContext, *, from_stage: str = "validate",
                 to_stage: str = "report") -> dict[str, dict[str, Any]]:
    from_stage = ALIASES.get(from_stage, from_stage)
    to_stage = ALIASES.get(to_stage, to_stage)
    if from_stage not in STAGES or to_stage not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}")
    start, end = STAGES.index(from_stage), STAGES.index(to_stage)
    if start > end:
        raise ValueError("from-stage must not follow to-stage")
    for stage in STAGES[start:end + 1]:
        previous = context.stage_results.get(stage)
        if previous and previous.get("status") in {"success", "reused", "skipped"}:
            receipt_path = context.paths.stage_receipt(stage)
            try:
                receipt = read_receipt(receipt_path)
                paths = [Path(value) for value in previous.get("outputs", {}).values()
                         if isinstance(value, str) and ("/" in value or "\\" in value)]
                reusable = (
                    receipt.get("config_hash") == context.config.config_hash
                    and receipt.get("status") == previous.get("status")
                    and all(path.exists() for path in paths))
            except (ContractError, OSError, ValueError):
                reusable = False
            if reusable:
                continue
            raise ContractError(
                f"saved stage {stage} is inconsistent; start a new run instead of "
                "overwriting provenance")
        result = HANDLERS[stage](context)
        context.record_stage(stage, result)
        receipt = make_receipt(
            root=context.paths.root, stage=stage,
            config_hash=context.config.config_hash,
            execution_level=context.execution_level, status=result.status,
            input_hashes=result.input_hashes,
            output_hashes={key: content_hash(value)
                           for key, value in result.outputs.items()},
            message=result.message)
        write_receipt(context.paths.stage_receipt(stage), receipt)
        if result.status in {"blocked", "failed"}:
            break
    return context.stage_results
