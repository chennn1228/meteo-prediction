from __future__ import annotations

import pytest
import json

from nwp.core.config import project_root, resolve_config
from nwp.core.context import RunContext
from nwp.workflow.pipeline import run_pipeline
from nwp.core.schema import ContractError


def test_development_workflow_stops_at_missing_catalog_dependency(tmp_path):
    context = RunContext.create(
        project_root(),
        resolve_config("nanjing_cpu_diagnostic"),
        data_root=tmp_path / "data",
        outputs_root=tmp_path / "outputs",
    )
    statuses = run_pipeline(context, to_stage="splits")
    assert statuses["validate"]["status"] == "success"
    assert statuses["selection"]["status"] == "success"
    assert statuses["data"]["status"] == "blocked"
    assert "features" not in statuses and "splits" not in statuses
    assert (context.paths.meta_dir / "resolved_config.yaml").exists()
    assert (context.paths.meta_dir / "provenance.json").exists()
    assert (context.paths.stage_dir("selection") / "sites.json").exists()
    assert not (context.paths.stage_dir("splits") / "splits.json").exists()
    with pytest.raises(ValueError, match="BLOCKED is read-only"):
        RunContext.resume(
            project_root(), context.config, run_id=context.run_id,
            data_root=tmp_path / "data", outputs_root=tmp_path / "outputs")


def test_pipeline_resume_rejects_tampered_stage_artifact(tmp_path):
    context = RunContext.create(
        project_root(), resolve_config("nanjing_cpu_diagnostic", models="raw_gfs"),
        data_root=tmp_path / "data", outputs_root=tmp_path / "outputs",
        run_id="tamper-through-pipeline")
    run_pipeline(context, to_stage="selection")
    sites = context.paths.stage_dir("selection") / "sites.json"
    payload = json.loads(sites.read_text(encoding="utf-8"))
    payload["tampered"] = True
    sites.write_text(json.dumps(payload), encoding="utf-8")
    resumed = RunContext.resume(
        project_root(), context.config, run_id=context.run_id,
        data_root=tmp_path / "data", outputs_root=tmp_path / "outputs")
    with pytest.raises(ContractError, match="inconsistent; create a child run"):
        run_pipeline(resumed, from_stage="selection", to_stage="selection")
