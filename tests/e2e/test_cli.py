from __future__ import annotations

from nwp.cli import _child_metadata, main
from nwp.core.config import resolve_config
import json
import yaml
import pytest
from pathlib import Path


def test_single_cli_can_validate_a_resolved_profile_without_creating_a_run():
    assert main(["validate", "--profile", "nanjing_cpu_diagnostic"]) == 0


def test_data_commands_honor_local_data_root_and_boundary(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    local_data = tmp_path / "machine-data"
    (config / "local.yaml").write_text(
        "paths:\n  data_root: machine-data\n  outputs_root: machine-outputs\n",
        encoding="utf-8")
    seen = {}
    monkeypatch.setattr("nwp.cli.project_root", lambda: tmp_path)
    def inventory(path):
        seen["inventory"] = Path(path)
        return {"status": "bounded-test"}
    monkeypatch.setattr("nwp.cli.inventory_data", inventory)
    assert main(["data", "inventory"]) == 0
    assert seen["inventory"] == local_data.resolve()

    monkeypatch.setattr("nwp.cli.load_bundle",
                        lambda _root: {"data": {"forecast": {"model": "gfs"}}})
    def probe(_step, **kwargs):
        seen.update(kwargs)
        return {"status": "bounded-test"}
    monkeypatch.setattr("nwp.cli.probe_service_round", probe)
    assert main(["data", "probe-service", "--step", "0.25"]) == 0
    assert seen["data_root"] == local_data.resolve()
    assert seen["boundary_path"] == (
        local_data / "registry" / "geography" / "jiangsu.geojson").resolve()


def test_child_run_requires_terminal_parent_and_computes_nested_changes(
        tmp_path, monkeypatch):
    config = resolve_config("nanjing_cpu_diagnostic", models="raw_gfs")
    run = tmp_path / "outputs" / "development" / "parent" / "00_meta"
    run.mkdir(parents=True)
    (run / "provenance.json").write_text("{}", encoding="utf-8")
    (run / "artifact_manifest.json").write_text(
        '{"artifacts": []}', encoding="utf-8")
    saved = config.as_dict()
    saved["protocol"]["validation"]["gap_days"] = 2
    (run / "resolved_config.yaml").write_text(
        yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"state": "RUNNING"}), encoding="utf-8")
    monkeypatch.setattr(
        "nwp.cli.load_local_paths",
        lambda _root: {"outputs_root": tmp_path / "outputs",
                       "data_root": tmp_path / "data"})
    with pytest.raises(Exception, match="parent run must be terminal"):
        _child_metadata("parent", config)
    (run / "status.json").write_text(
        json.dumps({"state": "BLOCKED"}), encoding="utf-8")
    assert "protocol.validation.gap_days" in _child_metadata("parent", config)
