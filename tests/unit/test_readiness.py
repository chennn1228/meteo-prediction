"""Readiness modes must report evidence, not promote prototype work."""
from __future__ import annotations

import pytest

from nwp.core.config import ConfigError
from nwp.core.validation import readiness_result


def test_layered_modes_fail_closed_without_real_completion():
    structural = readiness_result("structural")
    assert structural["status"] == "pass"
    for mode in ("data_ready", "cpu_ready", "deep_ready", "official_full"):
        report = readiness_result(mode)
        assert report["status"] == "blocked"
        assert report["total"] > structural["total"]
    cpu = readiness_result("cpu_ready")
    assert any(check["name"] == "complete ready raw source-site-month coverage"
               and check["passed"] for check in cpu["checks"])
    assert any(check["name"] == "acquisition-time provenance accepted for official use"
               and not check["passed"] for check in cpu["checks"])
    assert not any(check["scope"] == "deep_ready" for check in cpu["checks"])


def test_unknown_readiness_mode_rejected():
    with pytest.raises(ConfigError, match="invalid validation mode"):
        readiness_result("guess_ready")
