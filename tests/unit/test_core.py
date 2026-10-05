from __future__ import annotations

import json
from dataclasses import replace

import pytest

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.core.hashing import canonical_json, content_hash
from nwp.core.paths import RUN_DIRECTORIES, RunPaths
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError, StageResult


def test_canonical_hashing_is_order_independent_and_fail_closed():
    assert content_hash({"b": [2, 1], "a": {"x", "y"}}) == content_hash({"a": {"y", "x"}, "b": (2, 1)})
    assert canonical_json({"path": project_root() / "data"}).startswith('{"path":')
    with pytest.raises(ValueError, match="non-finite"):
        content_hash({"bad": float("nan")})
    with pytest.raises(TypeError, match="unsupported"):
        content_hash(object())


def test_run_paths_own_all_fixed_locations_and_reject_unsafe_segments(tmp_path):
    paths = RunPaths.create(
        project_root(),
        execution="development",
        run_id="path-test",
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
    )
    paths.initialize_run()
    assert {path.name for path in paths.run_root.iterdir()} == set(RUN_DIRECTORIES)
    assert paths.model_fold("ridge_mos", "outer_01") == paths.models_dir / "ridge_mos" / "outer_01"
    assert paths.prediction_file("xgboost", "outer_02") == paths.predictions_dir / "xgboost" / "outer_02" / "predictions.parquet"
    raw = paths.raw_partition("gfs", "nanjing_1", "2024-02")
    assert raw == paths.data_root / "raw" / "gfs" / "nanjing_1" / "2024-02.parquet"
    with pytest.raises(ContractError, match="unsafe"):
        paths.model_fold("../escape", "outer_01")
    with pytest.raises(ContractError, match="stay under"):
        paths.data_receipt(tmp_path / "outside.parquet")


def test_receipt_roundtrip_validation_and_no_overwrite(tmp_path):
    receipt = make_receipt(
        root=project_root(),
        stage="features",
        config_hash="abcdef12",
        execution_level="development",
        status="success",
        input_hashes={"clean": "12345678"},
        output_hashes={"features": "abcdefabcdef"},
        dependency_fingerprint="1" * 64,
        implementation_fingerprint="2" * 64,
        environment_fingerprint="3" * 64,
    )
    path = tmp_path / "features.receipt.json"
    write_receipt(path, receipt)
    assert read_receipt(path) == receipt
    with pytest.raises(ContractError, match="will not be overwritten"):
        write_receipt(path, receipt)
    damaged = dict(receipt)
    damaged["input_hashes"] = {"clean": "not-a-hash"}
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(damaged), encoding="utf-8")
    with pytest.raises(ContractError, match="lowercase hex"):
        read_receipt(bad_path)


def test_run_context_writes_complete_meta_and_records_immutable_stage(tmp_path):
    config = resolve_config("nanjing_cpu_diagnostic")
    context = RunContext.create(
        project_root(),
        config,
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
        run_id="context-test",
    )
    assert (context.paths.meta_dir / "resolved_config.yaml").is_file()
    provenance = json.loads((context.paths.meta_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["run_id"] == "context-test"
    assert provenance["config_hash"] == config.config_hash
    assert len(provenance["git_commit"]) == 40
    assert not (context.paths.meta_dir / "stage_results.json").exists()
    result = StageResult(
        status="success",
        artifact_outputs={},
        metadata_outputs={"count": 1},
        inputs={"profile": config.profile},
        dependency_fingerprint=config.config_hash,
        implementation_fingerprint="abcdef12",
        input_hashes={"config": config.config_hash},
        output_hashes={"count": "abcdef12"},
        started_at="2026-10-02T00:00:00+00:00",
        finished_at="2026-10-02T00:00:01+00:00",
    )
    context.record_stage("validate", result)
    saved = json.loads((context.paths.meta_dir / "stage_results.json").read_text(encoding="utf-8"))
    assert saved["validate"]["status"] == "success"
    with pytest.raises(TypeError):
        result.inputs["profile"] = "changed"
    with pytest.raises(ContractError, match="already recorded"):
        context.record_stage("validate", result)
    resumed = RunContext.resume(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="context-test")
    assert resumed.stage_results == saved
    registry_only_change = replace(
        config, site_registry_hash="1" * 64, model_registry_hash="2" * 64,
        official_result_set="future-official-result",
        locked_config_hash="f" * 64)
    registry_resumed = RunContext.resume(
        project_root(), registry_only_change, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="context-test")
    assert registry_resumed.config.config_hash == config.config_hash
    changed = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    with pytest.raises(ContractError, match="differs"):
        RunContext.resume(
            project_root(), changed, data_root=tmp_path / "data",
            outputs_root=tmp_path / "outputs", run_id="context-test")


def test_same_run_resume_rejects_runtime_source_drift(tmp_path, monkeypatch):
    config = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="runtime-drift")
    monkeypatch.setattr("nwp.core.context.runtime_source_fingerprint",
                        lambda _root: "0" * 64)
    with pytest.raises(ContractError, match="runtime source fingerprint differs"):
        RunContext.resume(
            project_root(), config, data_root=tmp_path / "data",
            outputs_root=tmp_path / "outputs", run_id="runtime-drift")
