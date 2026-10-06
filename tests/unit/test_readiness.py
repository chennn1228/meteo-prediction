"""Readiness modes must report evidence, not promote prototype work."""
from __future__ import annotations

import shutil

import pytest

from nwp.core.config import ConfigError, project_root
from nwp.core.validation import readiness_result


def test_layered_modes_fail_closed_without_real_completion(tmp_path):
    root = tmp_path / "repository_without_local_data"
    shutil.copytree(project_root() / "config", root / "config")
    shutil.copytree(project_root() / "src", root / "src")
    shutil.copy2(project_root() / "project_manifest.yaml", root)

    structural = readiness_result("structural", root)
    assert structural["status"] == "pass"
    data = readiness_result("data_ready", root)
    assert data["status"] == "blocked"
    assert data["total"] > structural["total"]
    for mode in ("cpu_ready", "deep_ready", "official_full"):
        report = readiness_result(mode, root)
        assert report["status"] == "blocked"
        assert report["total"] > data["total"]
    cpu = readiness_result("cpu_ready", root)
    assert any(check["name"] == "complete ready raw source-site-month coverage"
               and not check["passed"] for check in cpu["checks"])
    assert any(check["name"] == "raw provenance is complete and byte-verifiable"
               and check["passed"] for check in cpu["checks"])
    assert not any(check["scope"] == "deep_ready" for check in cpu["checks"])


def test_unknown_readiness_mode_rejected():
    with pytest.raises(ConfigError, match="invalid validation mode"):
        readiness_result("guess_ready")
