from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from nwp.core.config import project_root, resolve_config, to_plain
from nwp.core.context import RunContext
from nwp.core.lifecycle import transition
from nwp.core.provenance import make_receipt, write_receipt
from nwp.core.fingerprints import sha256_file
from nwp.core.fingerprints import stable_object_hash
from nwp.workflow.pipeline import _stage_environment_fingerprint, run_pipeline
import nwp.workflow.stages.modeling as modeling


MODELS = ("ridge_mos", "lgbm", "xgboost")


class FakeModel:
    def __init__(self, model_id: str, calls: dict[str, int]):
        self.model_id = model_id
        self.calls = calls

    def save(self, path: Path) -> None:
        self.calls[self.model_id] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"bounded-model:{self.model_id}".encode())


def _prime(context: RunContext, *, xgb_depth: int) -> None:
    selection = {
        "outer": {"outer_1": {}},
        "final": {},
    }
    for model_id in MODELS:
        parameters = {"depth": xgb_depth} if model_id == "xgboost" else {"fixed": 1}
        detail = {"selected_candidate": 0, "parameters": parameters,
                  "trials": [], "tree_round_receipts": []}
        selection["outer"]["outer_1"][model_id] = detail
        selection["final"][model_id] = detail
    path = context.paths.tuning_dir / "selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    feature_path = context.paths.stage_dir("features") / "feature-index.json"
    feature_path.write_text('{"bounded": true}', encoding="utf-8")
    context.stage_results = {
        "features": {"status": "success", "input_hashes": {"features": "a" * 64},
                     "output_hashes": {"features": sha256_file(feature_path)},
                     "artifact_outputs": {"features": str(feature_path)},
                     "metadata_outputs": {}, "dependency_fingerprint": "c" * 64,
                     "implementation_fingerprint": "d" * 64},
        "tuning": {"status": "success", "input_hashes": {"tuning": "b" * 64},
                   "artifact_outputs": {"selection": str(path)},
                   "output_hashes": {"selection": sha256_file(path)},
                   "metadata_outputs": {}, "dependency_fingerprint": "e" * 64,
                   "implementation_fingerprint": "f" * 64},
    }
    for stage in ("features", "tuning"):
        row = context.stage_results[stage]
        write_receipt(context.paths.stage_receipt(stage), make_receipt(
            root=context.paths.root, stage=stage,
            config_hash=context.config.config_hash,
            execution_level=context.execution_level, status="success",
            input_hashes=row["input_hashes"], output_hashes=row["output_hashes"],
            dependency_fingerprint=row["dependency_fingerprint"],
            implementation_fingerprint=row["implementation_fingerprint"],
            environment_fingerprint=_stage_environment_fingerprint()))


def test_child_workflow_reuses_unchanged_model_local_artifacts(tmp_path, monkeypatch):
    config = resolve_config("nanjing_cpu_diagnostic", models=MODELS)
    calls = {model_id: 0 for model_id in MODELS}
    frame = pd.DataFrame({"placeholder": [1.0]})
    outer = SimpleNamespace(fold_id="outer_1", fit=frame)
    monkeypatch.setattr(modeling, "_load_feature_frame", lambda _context: frame)
    monkeypatch.setattr(modeling, "_eligible", lambda _context, block: block)
    monkeypatch.setattr(modeling, "outer_folds", lambda *_args, **_kwargs: [outer])
    monkeypatch.setattr(modeling, "inner_folds", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        modeling, "_final_blocks", lambda *_args, **_kwargs: (frame, frame, frame, frame))
    monkeypatch.setattr(
        modeling, "fit_outer_quantile_model",
        lambda _context, model_id, *_args, **_kwargs: FakeModel(model_id, calls))
    monkeypatch.setattr(
        modeling, "fit_final_quantile_model",
        lambda _context, model_id, *_args, **_kwargs: FakeModel(model_id, calls))

    first = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="reuse-a",
        allow_model_execution=True)
    _prime(first, xgb_depth=6)
    run_pipeline(first, from_stage="fitting", to_stage="fitting")
    assert calls == {model_id: 2 for model_id in MODELS}
    transition(first.paths.meta_dir / "status.json", "FROZEN")
    first_manifest = json.loads(
        (first.paths.meta_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert len([row for row in first_manifest["artifacts"]
                if row["artifact_type"] == "model"]) == 6

    calls.update({model_id: 0 for model_id in MODELS})
    child = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="reuse-b",
        parent_run_id=first.run_id, change_reason="xgboost search-space change",
        changed_dependencies=["models.search.spaces.xgboost"],
        allow_model_execution=True)
    _prime(child, xgb_depth=9)
    run_pipeline(child, from_stage="fitting", to_stage="fitting")
    assert calls["ridge_mos"] == 0
    assert calls["lgbm"] == 0
    assert calls["xgboost"] == 2
    child_manifest = json.loads(
        (child.paths.meta_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    model_records = [row for row in child_manifest["artifacts"]
                     if row["artifact_type"] == "model"]
    assert len(model_records) == 6
    assert all("reused_from_run" in row["scope"] for row in model_records
               if row["model_id"] in {"ridge_mos", "lgbm"})


def test_child_dependency_changes_reuse_model_artifacts(tmp_path, monkeypatch):
    """Gap, interpretation, and figure-only changes do not refit models."""
    base_config = resolve_config("nanjing_cpu_diagnostic", models=MODELS)
    calls = {model_id: 0 for model_id in MODELS}
    frame = pd.DataFrame({"placeholder": [1.0]})
    outer = SimpleNamespace(fold_id="outer_1", fit=frame)
    monkeypatch.setattr(modeling, "_load_feature_frame", lambda _context: frame)
    monkeypatch.setattr(modeling, "_eligible", lambda _context, block: block)
    monkeypatch.setattr(modeling, "outer_folds", lambda *_args, **_kwargs: [outer])
    monkeypatch.setattr(modeling, "inner_folds", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        modeling, "_final_blocks", lambda *_args, **_kwargs: (frame, frame, frame, frame))
    monkeypatch.setattr(
        modeling, "fit_outer_quantile_model",
        lambda _context, model_id, *_args, **_kwargs: FakeModel(model_id, calls))
    monkeypatch.setattr(
        modeling, "fit_final_quantile_model",
        lambda _context, model_id, *_args, **_kwargs: FakeModel(model_id, calls))

    parent = RunContext.create(
        project_root(), base_config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="boundary-parent",
        allow_model_execution=True)
    _prime(parent, xgb_depth=6)
    run_pipeline(parent, from_stage="fitting", to_stage="fitting")
    transition(parent.paths.meta_dir / "status.json", "FROZEN")

    gap_config = resolve_config(
        "nanjing_cpu_diagnostic", models=MODELS, gap_days=7)
    analysis_payload = to_plain(base_config.analysis)
    analysis_payload["evidence"] = dict(analysis_payload["evidence"])
    analysis_payload["evidence"]["permutation_repeats"] = 11
    permutation_config = replace(
        base_config, analysis=analysis_payload,
        config_hash=stable_object_hash({"variant": "permutation-repeats"}))
    variants = (
        ("boundary-gap", gap_config, "protocol.validation.gap_days"),
        ("boundary-permutation", permutation_config,
         "analysis.evidence.permutation_repeats"),
        ("boundary-figure-style", base_config, "visualization.style"),
    )
    for run_id, config, changed in variants:
        calls.update({model_id: 0 for model_id in MODELS})
        child = RunContext.create(
            project_root(), config, data_root=tmp_path / "data",
            outputs_root=tmp_path / "outputs", run_id=run_id,
            parent_run_id=parent.run_id, change_reason=f"change {changed}",
            changed_dependencies=[changed], allow_model_execution=True)
        _prime(child, xgb_depth=6)
        run_pipeline(child, from_stage="fitting", to_stage="fitting")
        assert calls == {model_id: 0 for model_id in MODELS}
        manifest = json.loads(
            (child.paths.meta_dir / "artifact_manifest.json").read_text(
                encoding="utf-8"))
        records = [row for row in manifest["artifacts"]
                   if row["artifact_type"] == "model"]
        assert len(records) == 6
        assert all(row["scope"].get("reused_from_run") for row in records)
