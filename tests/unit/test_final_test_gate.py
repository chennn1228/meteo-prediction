from __future__ import annotations

import json
from dataclasses import replace
from types import MappingProxyType

import pytest

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.core.schema import ContractError
from nwp.workflow.stages.analysis import evaluate
from nwp.workflow.stages.common import _final_test_evaluation_allowed


def test_development_cannot_open_or_evaluate_final_test(tmp_path):
    config = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    context = RunContext.create(
        project_root(), config, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="development_gate")
    assert not _final_test_evaluation_allowed(context)

    index = context.paths.predictions_dir / "prediction_index.json"
    index.write_text(json.dumps({
        "outer_predictions": [],
        "final_test_uncalibrated": [{"path": "forbidden"}],
    }), encoding="utf-8")
    context.stage_results["prediction"] = {
        "status": "success", "artifact_outputs": {
            "prediction_index": str(index)}}
    context.stage_results["calibration"] = {
        "status": "skipped", "artifact_outputs": {}}
    with pytest.raises(ContractError, match="frozen official evaluation gate"):
        evaluate(context)


def test_final_test_gate_requires_frozen_official_config_and_readiness(tmp_path):
    development = resolve_config(
        "nanjing_cpu_diagnostic", models="raw_gfs")
    official = replace(
        development, execution="official", overrides=MappingProxyType({}),
        locked_config_hash=development.config_hash,
        official_result_set="synthetic")
    context = RunContext.create(
        project_root(), official, data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs", run_id="official_gate")
    assert not _final_test_evaluation_allowed(context)
    (context.paths.meta_dir / "readiness_receipt.json").write_text(
        json.dumps({"overall_ready": True,
                    "scientific_run_hash": official.config_hash}),
        encoding="utf-8")
    assert _final_test_evaluation_allowed(context)
