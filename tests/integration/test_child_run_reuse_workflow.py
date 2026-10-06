from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import yaml

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.core.dependencies import (
    data_dependency_fingerprint, feature_dependency_fingerprint,
    split_dependency_fingerprint)
from nwp.core.fingerprints import sha256_file, source_files_fingerprint
from nwp.core.lifecycle import transition
from nwp.core.provenance import make_receipt, write_receipt
from nwp.experiment.tuning import Trial, candidates
from nwp.splits.rolling import InnerFold, OuterFold
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


def _write_stage(context: RunContext, stage: str, dependency: str,
                 artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps({"stage": stage}), encoding="utf-8")
    implementation = source_files_fingerprint(
        context.paths.root,
        ("src/nwp/workflow/stages/common.py",
         "src/nwp/workflow/stages/prepare.py"))
    result = {
        "status": "success", "input_hashes": {stage: dependency},
        "output_hashes": {stage: sha256_file(artifact)},
        "artifact_outputs": {stage: str(artifact)}, "metadata_outputs": {},
        "dependency_fingerprint": dependency,
        "implementation_fingerprint": implementation,
    }
    context.stage_results[stage] = result
    write_receipt(context.paths.stage_receipt(stage), make_receipt(
        root=context.paths.root, stage=stage,
        config_hash=context.config.config_hash,
        execution_level=context.execution_level, status="success",
        input_hashes=result["input_hashes"], output_hashes=result["output_hashes"],
        dependency_fingerprint=dependency,
        implementation_fingerprint=implementation,
        environment_fingerprint=_stage_environment_fingerprint()))


def _prime_real_upstream_dependencies(context: RunContext) -> None:
    data = data_dependency_fingerprint(
        context.config.data, [{"bounded-source": "1" * 64}])
    features = feature_dependency_fingerprint(
        data, context.config.features, {"bounded-feature": "2" * 64})
    splits = split_dependency_fingerprint(
        features, context.selected_sites,
        context.config.protocol["validation"])
    _write_stage(
        context, "features", features,
        context.paths.stage_dir("features") / "feature-index.json")
    _write_stage(
        context, "splits", splits,
        context.paths.stage_dir("splits") / "splits.json")


def _variant_root(tmp_path: Path) -> Path:
    root = tmp_path / "variant-project"
    root.mkdir()
    shutil.copy2(project_root() / "project_manifest.yaml", root)
    shutil.copytree(project_root() / "config", root / "config")
    models_path = root / "config" / "models.yaml"
    payload = yaml.safe_load(models_path.read_text(encoding="utf-8"))
    payload["search"]["spaces"]["xgboost"][0]["max_depth"] = 9
    models_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return root


def test_xgboost_change_uses_real_tuning_dependencies_and_reuses_other_models(
        tmp_path, monkeypatch):
    base_config = resolve_config("nanjing_cpu_diagnostic", models=MODELS)
    child_config = resolve_config(
        "nanjing_cpu_diagnostic", root=_variant_root(tmp_path), models=MODELS)
    frame = pd.DataFrame({"placeholder": [1.0]})
    stamp = pd.Timestamp("2024-01-01", tz="UTC")
    outer = OuterFold("outer_1", frame, frame, stamp, stamp)
    inner = [InnerFold(
        f"inner_{index}", frame, frame, frame, stamp, stamp, stamp, stamp)
        for index in range(1, 4)]
    fit_calls = {model_id: 0 for model_id in MODELS}
    tuning_calls = {model_id: 0 for model_id in MODELS}

    monkeypatch.setattr(modeling, "_load_feature_frame", lambda _context: frame)
    monkeypatch.setattr(modeling, "_eligible", lambda _context, block: block)
    monkeypatch.setattr(modeling, "outer_folds", lambda *_args, **_kwargs: [outer])
    monkeypatch.setattr(modeling, "inner_folds", lambda *_args, **_kwargs: inner)
    monkeypatch.setattr(
        modeling, "_final_blocks", lambda *_args, **_kwargs:
        (frame, frame, frame, frame))

    def fake_trials(model_id, outer_id, folds, _adapter, *, model_config,
                    protocol_config, device, gap_days):
        tuning_calls[model_id] += 1
        rows = []
        for candidate, parameters in enumerate(candidates(model_id, model_config)):
            for fold in folds:
                rows.append(Trial(
                    model=model_id, candidate=candidate,
                    parameters=dict(parameters), outer=outer_id,
                    inner=fold.fold_id, seed=0, fit_rows=1,
                    early_stop_rows=1, scoring_rows=1,
                    mean_pinball=float(candidate + 1), wall_seconds=0.0,
                    device=device, peak_memory_bytes=0, epoch=1,
                    status="ok", error_reason=None))
        return 0, rows

    monkeypatch.setattr(modeling, "run_trials", fake_trials)
    monkeypatch.setattr(
        modeling, "fit_outer_quantile_model",
        lambda _context, model_id, *_args, **_kwargs:
        FakeModel(model_id, fit_calls))
    monkeypatch.setattr(
        modeling, "fit_final_quantile_model",
        lambda _context, model_id, *_args, **_kwargs:
        FakeModel(model_id, fit_calls))

    parent = RunContext.create(
        project_root(), base_config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="reuse_parent",
        allow_model_execution=True)
    _prime_real_upstream_dependencies(parent)
    run_pipeline(parent, from_stage="tuning", to_stage="fitting")
    assert tuning_calls == {model_id: 1 for model_id in MODELS}
    assert fit_calls == {model_id: 2 for model_id in MODELS}
    transition(parent.paths.meta_dir / "status.json", "FROZEN")

    tuning_calls.update({model_id: 0 for model_id in MODELS})
    fit_calls.update({model_id: 0 for model_id in MODELS})
    child = RunContext.create(
        project_root(), child_config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="reuse_child",
        parent_run_id=parent.run_id, change_reason="xgboost search-space change",
        changed_dependencies=["models.search.spaces.xgboost"],
        allow_model_execution=True)
    _prime_real_upstream_dependencies(child)
    run_pipeline(child, from_stage="tuning", to_stage="fitting")

    assert tuning_calls == {"ridge_mos": 0, "lgbm": 0, "xgboost": 1}
    assert fit_calls == {"ridge_mos": 0, "lgbm": 0, "xgboost": 2}
    manifest = json.loads(
        (child.paths.meta_dir / "artifact_manifest.json").read_text(
            encoding="utf-8"))
    models = [row for row in manifest["artifacts"]
              if row["artifact_type"] == "model"]
    assert len(models) == 6
    assert all(row["scope"].get("reused_from_run") == parent.run_id
               for row in models
               if row["model_id"] in {"ridge_mos", "lgbm"})
    assert all("reused_from_run" not in row["scope"] for row in models
               if row["model_id"] == "xgboost")
