"""Readiness modes must report evidence, not promote prototype work."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError
from s15_validation.validate_project import result


def test_layered_modes_fail_closed_without_real_completion():
    structural = result("structural")
    assert structural["status"] == "pass"
    for mode in ("data_ready", "cpu_ready", "deep_ready", "official_full"):
        report = result(mode)
        assert report["status"] == "blocked"
        assert report["total"] > structural["total"]
    cpu = result("cpu_ready")
    assert any(check["name"] == "complete temporary 15-variable monthly raw grid with sidecars"
               and not check["passed"] for check in cpu["checks"])
    assert any(check["name"] == "real pilot GFS satisfies temporary 15-variable contract"
               for check in cpu["checks"])
    assert not any(check["scope"] == "deep_ready" for check in cpu["checks"])


def test_unknown_readiness_mode_rejected():
    try:
        result("guess_ready")
    except ProtocolError:
        return
    raise AssertionError("unregistered readiness mode accepted")
