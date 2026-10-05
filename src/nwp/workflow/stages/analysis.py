"""Stage handlers split by workflow responsibility."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nwp.core.artifacts import environment_packages, new_artifact_record
from nwp.core.config import to_plain
from nwp.core.context import RunContext
from nwp.core.dependencies import model_dependency_fingerprint
from nwp.core.fingerprints import (
    environment_fingerprint, source_files_fingerprint, stable_object_hash)
from nwp.core.hashing import content_hash, file_sha256
from nwp.core.schema import ContractError, StageResult
from nwp.evaluation.grouped import evaluate_predictions, write_evaluation_report
from nwp.evaluation.interpretation import group_mechanism_evidence, model_summary
from nwp.experiment.prediction import read_predictions
from nwp.visualization.figures import plan_run_figures, write_figure_index
from nwp.visualization.style import apply_publication_style

from .common import (
    _eligible, _find_reusable_model_artifact, _load_feature_frame,
    _model_implementation_fingerprint, _now, _record_model_artifact,
    _stage_result, _stage_outputs, _write_json,
    _restore_artifact_bundle, _write_artifact_bundle,
)

def evaluate(context: RunContext) -> StageResult:
    started = _now()
    calibration = context.stage_results.get("calibration", {})
    if calibration.get("status") not in {"success", "skipped", "reused"}:
        return _stage_result(context, "blocked",
                       message="evaluation requires prediction/calibration state",
                       started_at=started)
    prediction = context.stage_results.get("prediction", {})
    index_path = Path(_stage_outputs(prediction)["prediction_index"])
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
    model_metric_hashes: dict[str, str] = {}
    for model_id, model_rows in combined.groupby("model_id", sort=True):
        model_id = str(model_id)
        local_dir = context.paths.model_metrics_dir(model_id)
        bundle_path = local_dir / "metrics.bundle.json"
        dependency = model_dependency_fingerprint(
            "metrics", model_id,
            upstream={"predictions": stable_object_hash([
                {"sha256": item["sha256"],
                 "dependency_fingerprint": item["dependency_fingerprint"]}
                for item in index.get("outer_predictions", [])
                if item["model_id"] == model_id])},
            model_record=context.config.models["registry"][model_id],
            stage_contract=context.config.protocol["evaluation"])
        implementation = _model_implementation_fingerprint(context, model_id)
        reusable = _find_reusable_model_artifact(
            context, artifact_type="metrics", model_id=model_id,
            fold_id="aggregate_outer", dependency=dependency,
            implementation=implementation, time_scope="outer_validation")
        reuse_meta = None
        if reusable is not None:
            source_path, source_record = reusable
            source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
            reuse_meta = context.artifact_resolver.materialize(
                source_path, bundle_path, source_record.sha256,
                reused_from_run=source_run,
                source_artifact_id=source_record.artifact_id,
                source_dependency_fingerprint=source_record.dependency_fingerprint,
                source_execution_level=source_record.execution_level)
            _restore_artifact_bundle(bundle_path, local_dir)
            local_index = local_dir / "metrics_index.json"
            local_payload = json.loads(local_index.read_text(encoding="utf-8"))
            local_payload["overview"] = {
                name: str(local_dir / f"{name}.csv")
                for name in local_payload["overview"]}
            local_payload["grouped"] = {
                dimension: {
                    name: str(local_dir / "grouped" / dimension / f"{name}.csv")
                    for name in tables}
                for dimension, tables in local_payload["grouped"].items()}
            local_index.write_text(
                json.dumps(local_payload, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n", encoding="utf-8")
        else:
            local_report = evaluate_predictions(
                model_rows.copy(), context.config.protocol, context.config.models,
                formal=context.execution_level == "official")
            write_evaluation_report(local_report, local_dir)
            _write_artifact_bundle(local_dir, bundle_path)
        _record_model_artifact(
            context, bundle_path, artifact_type="metrics", stage="evaluation",
            model_id=model_id, fold_id="aggregate_outer",
            dependency=dependency,
            implementation=implementation,
            time_scope="outer_validation", reuse=reuse_meta)
        model_metric_hashes[model_id] = file_sha256(bundle_path)
    metrics_index = write_evaluation_report(report, context.paths.aggregate_metrics_dir())
    root_metrics = json.loads(metrics_index.read_text(encoding="utf-8"))
    root_metrics["model_local_artifacts"] = model_metric_hashes
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
                final_report, context.paths.aggregate_metrics_dir() / "final_uncalibrated"))
    calibration_index = Path(
        _stage_outputs(calibration)["calibration_index"])
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
                context.paths.aggregate_metrics_dir() / "final_calibrated"))
    root_metrics["final"] = final_indexes
    _write_json(metrics_index, root_metrics)
    dependency = content_hash({
        "model_metric_hashes": model_metric_hashes,
        "evaluation": context.config.protocol["evaluation"]})
    return _stage_result(
        context, "success", inputs={"prediction_index": str(index_path)},
        outputs={"metrics_index": str(metrics_index)},
        input_hashes={"evaluation": dependency}, started_at=started)


def analyse(context: RunContext) -> StageResult:
    started = _now()
    if context.stage_results.get("evaluation", {}).get("status") != "success":
        return _stage_result(context, "blocked",
                       message="analysis requires evaluated predictions",
                       started_at=started)
    metrics_index = Path(
        _stage_outputs(context.stage_results["evaluation"])["metrics_index"])
    metrics = json.loads(metrics_index.read_text(encoding="utf-8"))
    probability = pd.read_csv(metrics["overview"]["probability_primary"])
    point = pd.read_csv(metrics["overview"]["point_secondary"])
    summary = model_summary(probability, point)
    model_analysis_hashes: dict[str, str] = {}
    for model_id, rows in summary.groupby("model_id", sort=True):
        model_id = str(model_id)
        local_dir = context.paths.model_analysis_dir(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        local_summary = local_dir / "summary.csv"
        dependency = model_dependency_fingerprint(
            "interpretation", model_id,
            upstream={"metrics": metrics["model_local_artifacts"][model_id]},
            model_record=context.config.models["registry"][model_id],
            stage_contract=context.config.analysis)
        implementation = _model_implementation_fingerprint(context, model_id)
        reusable = _find_reusable_model_artifact(
            context, artifact_type="analysis", model_id=model_id,
            fold_id="aggregate_outer", dependency=dependency,
            implementation=implementation, time_scope="outer_validation")
        reuse_meta = None
        if reusable is not None:
            source_path, source_record = reusable
            source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
            reuse_meta = context.artifact_resolver.materialize(
                source_path, local_summary, source_record.sha256,
                reused_from_run=source_run,
                source_artifact_id=source_record.artifact_id,
                source_dependency_fingerprint=source_record.dependency_fingerprint,
                source_execution_level=source_record.execution_level)
        else:
            rows.to_csv(local_summary, index=False)
        _record_model_artifact(
            context, local_summary, artifact_type="analysis", stage="analysis",
            model_id=model_id, fold_id="aggregate_outer",
            dependency=dependency,
            implementation=implementation,
            time_scope="outer_validation", reuse=reuse_meta)
        model_analysis_hashes[model_id] = file_sha256(local_summary)
    aggregate_analysis = context.paths.aggregate_analysis_dir()
    aggregate_analysis.mkdir(parents=True, exist_ok=True)
    summary_path = aggregate_analysis / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    index_path = aggregate_analysis / "analysis_index.json"
    tuning_trials_path = None
    ablation_path = permutation_path = None
    tuning = context.stage_results.get("tuning", {})
    if tuning.get("status") == "success":
        selection = json.loads(Path(
            _stage_outputs(tuning)["selection"]).read_text(encoding="utf-8"))
        rows = []
        for outer_id, models in selection["outer"].items():
            for model_id, detail in models.items():
                for trial in detail.get("trials", []):
                    rows.append({"outer_fold": outer_id, "model_id": model_id,
                                 **trial})
        if rows:
            tuning_trials_path = aggregate_analysis / "tuning_trials.csv"
            pd.DataFrame(rows).to_csv(tuning_trials_path, index=False)
        evidence = set(context.config.analysis["evidence"]["selection"])
        reference = context.config.analysis["evidence"]["reference_model"]
        if ({"group_ablation", "grouped_permutation_importance"} <= evidence
                and reference in context.selected_models):
            ablation_path = context.paths.model_analysis_dir(reference) / "group_ablation.csv"
            ablation_path.parent.mkdir(parents=True, exist_ok=True)
            permutation_path = (
                context.paths.model_analysis_dir(reference) / "grouped_permutation.csv")
            reference_dependency = model_dependency_fingerprint(
                "interpretation", reference,
                upstream={
                    "features": context.stage_results["features"]["input_hashes"]["features"],
                    "reference_selection": stable_object_hash({
                        outer_id: details.get(reference)
                        for outer_id, details in selection["outer"].items()})},
                model_record=context.config.models["registry"][reference],
                stage_contract=context.config.analysis)
            reference_implementation = _model_implementation_fingerprint(
                context, reference)
            reusable_by_label = {}
            for label in ("group_ablation", "grouped_permutation"):
                reusable_by_label[label] = _find_reusable_model_artifact(
                    context, artifact_type="analysis", model_id=reference,
                    fold_id=label, dependency=reference_dependency,
                    implementation=reference_implementation,
                    time_scope="outer_validation")
            computed = None
            for label, artifact_path in (
                    ("group_ablation", ablation_path),
                    ("grouped_permutation", permutation_path)):
                reusable = reusable_by_label[label]
                reuse_meta = None
                if reusable is not None:
                    source_path, source_record = reusable
                    source_run = source_path.relative_to(
                        context.paths.outputs_root).parts[1]
                    reuse_meta = context.artifact_resolver.materialize(
                        source_path, artifact_path, source_record.sha256,
                        reused_from_run=source_run,
                        source_artifact_id=source_record.artifact_id,
                        source_dependency_fingerprint=(
                            source_record.dependency_fingerprint),
                        source_execution_level=source_record.execution_level)
                else:
                    if computed is None:
                        computed = group_mechanism_evidence(
                            _load_feature_frame(context), selection,
                            protocol_config=context.config.protocol,
                            feature_config=context.config.features,
                            model_config=context.config.models,
                            eligible=lambda block: _eligible(context, block))
                    table = computed[0] if label == "group_ablation" else computed[1]
                    table.to_csv(artifact_path, index=False)
                _record_model_artifact(
                    context, artifact_path, artifact_type="analysis",
                    stage="analysis", model_id=reference, fold_id=label,
                    dependency=reference_dependency,
                    implementation=reference_implementation,
                    time_scope="outer_validation", reuse=reuse_meta)
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
        "model_analysis_hashes": model_analysis_hashes,
        "analysis": context.config.analysis})
    return _stage_result(
        context, "success", outputs={"analysis_index": str(index_path)},
        input_hashes={"analysis": dependency}, started_at=started)


def figures(context: RunContext) -> StageResult:
    started = _now()
    if context.stage_results.get("analysis", {}).get("status") != "success":
        return _stage_result(context, "blocked",
                       message="figures require analysis source tables",
                       started_at=started)
    directory = context.paths.figures_dir
    metrics_index = Path(
        _stage_outputs(context.stage_results["evaluation"])["metrics_index"])
    analysis_index = Path(
        _stage_outputs(context.stage_results["analysis"])["analysis_index"])
    prediction_index = Path(
        _stage_outputs(context.stage_results["prediction"])["prediction_index"])
    status = "official" if context.execution_level == "official" else "diagnostic"
    plans = plan_run_figures(
        metrics_index, analysis_index, directory, result_status=status,
        protocol_config=to_plain(context.config.protocol),
        prediction_index=prediction_index)
    visualization_implementation = source_files_fingerprint(
        context.paths.root, ("src/nwp/visualization",))
    entries = []
    style_applied = False
    for plan in plans:
        sources = [Path(value) for value in str(plan["source_table"]).split(";")]
        source_hashes = [file_sha256(path) for path in sources]
        dependency = stable_object_hash({
            "sources": source_hashes,
            "generation_function": plan["generation_function"],
            "visualization_implementation": visualization_implementation})
        reusable_by_output = {}
        for output in (Path(value) for value in plan["files"]):
            reusable_by_output[output] = (
                context.artifact_resolver.find_compatible_artifact(
                    artifact_type="figure", dependency_fingerprint=dependency,
                    implementation_fingerprint=visualization_implementation,
                    data_scope={"sites": list(context.selected_sites)},
                    time_scope="outer_validation",
                    split_scope={"figure_id": plan["figure_id"],
                                 "format": output.suffix.lower()},
                    execution_level=context.execution_level,
                    environment_fingerprint=environment_fingerprint(
                        environment_packages("figure"))))
        if all(value is not None for value in reusable_by_output.values()):
            entry = {key: value for key, value in plan.items() if key != "render"}
        else:
            if not style_applied:
                apply_publication_style()
                style_applied = True
            entry = plan["render"]()
        entries.append(entry)
        for output in (Path(value) for value in entry["files"]):
            reuse_meta = None
            reusable = reusable_by_output[output]
            if reusable is not None:
                source_path, source_record = reusable
                source_run = source_path.relative_to(context.paths.outputs_root).parts[1]
                if output.exists():
                    output.unlink()
                reuse_meta = context.artifact_resolver.materialize(
                    source_path, output, source_record.sha256,
                    reused_from_run=source_run,
                    source_artifact_id=source_record.artifact_id,
                    source_dependency_fingerprint=source_record.dependency_fingerprint,
                    source_execution_level=source_record.execution_level)
            relative = output.resolve().relative_to(context.paths.run_root).as_posix()
            context.artifact_resolver.append_record(new_artifact_record(
                artifact_id=stable_object_hash({"run": context.run_id, "path": relative}),
                artifact_type="figure", stage="figures", model_id=None,
                fold_id=None, scope={"figure_id": entry["figure_id"],
                                     **(reuse_meta or {})},
                data_scope={"sites": list(context.selected_sites)},
                time_scope="outer_validation",
                split_scope={"figure_id": entry["figure_id"],
                             "format": output.suffix.lower()},
                dependency_fingerprint=dependency,
                implementation_fingerprint=visualization_implementation,
                environment_fingerprint=environment_fingerprint(
                    environment_packages("figure")),
                sha256=file_sha256(output), path=relative, portable=True,
                execution_level=context.execution_level,
                result_status=status,
                official_reuse_eligible=context.execution_level == "official"))
    index = directory / "figure_index.json"
    write_figure_index(index, entries)
    dependency = content_hash({
        "analysis": context.stage_results["analysis"]["input_hashes"]["analysis"],
        "visualization_implementation": visualization_implementation})
    return _stage_result(
        context, "success",
        outputs={"figure_index": str(index), "figure_count": len(entries)},
        input_hashes={"figures": dependency}, started_at=started)


def report(context: RunContext) -> StageResult:
    started = _now()
    required = {
        "metrics_index": _stage_outputs(context.stage_results.get("evaluation", {})).get("metrics_index"),
        "analysis_index": _stage_outputs(context.stage_results.get("analysis", {})).get("analysis_index"),
        "figure_index": _stage_outputs(context.stage_results.get("figures", {})).get("figure_index"),
    }
    if any(not value or not Path(value).is_file() for value in required.values()):
        return _stage_result(context, "blocked",
                       message="report requires metrics, analysis, and figure indexes",
                       started_at=started)
    readiness_path = context.paths.meta_dir / "readiness_receipt.json"
    readiness = (json.loads(readiness_path.read_text(encoding="utf-8"))
                 if readiness_path.is_file() else None)
    blockers = [stage for stage, value in context.stage_results.items()
                if value.get("status") in {"blocked", "failed"}]
    payload = {
        "run_id": context.run_id, "run_type": context.run_type,
        "execution": context.execution_level,
        "parent_run_id": context.provenance.get("parent_run_id"),
        "scientific_run_hash": context.config.config_hash,
        "selected_site_ids": list(context.selected_sites),
        "selected_models": list(context.selected_models),
        "stage_statuses": {key: value.get("status") for key, value in context.stage_results.items()},
        "main_metrics": required["metrics_index"],
        "aggregate_comparison": required["metrics_index"],
        "analysis_index": required["analysis_index"],
        "figure_index": required["figure_index"],
        "readiness": readiness,
        "result_status": "official" if context.execution_level == "official" else "diagnostic",
        "blockers": blockers, "provenance": context.provenance,
    }
    summary = context.paths.report_dir / "summary.md"
    manifest = context.paths.report_dir / "report_manifest.json"
    if summary.exists() or manifest.exists():
        raise ContractError("report artifacts are write-once")
    summary.write_text("# Run summary\n\n```json\n" +
                       json.dumps(payload, ensure_ascii=False, indent=2) +
                       "\n```\n", encoding="utf-8")
    _write_json(manifest, {**payload, "summary_sha256": file_sha256(summary)})
    return _stage_result(context, "success", outputs={"summary": str(summary),
                   "report_manifest": str(manifest)}, started_at=started)
