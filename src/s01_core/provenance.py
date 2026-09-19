"""Execution and result-state rules shared by all formal stages."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from .config_loader import ProtocolError, load_manifest


def validate_run_levels(implementation_level: str, execution_level: str) -> None:
    manifest = load_manifest()
    if implementation_level not in manifest["implementation_levels"]:
        raise ProtocolError(f"unknown implementation level: {implementation_level}")
    if execution_level not in manifest["execution_levels"]:
        raise ProtocolError(f"unknown execution level: {execution_level}")
    if execution_level == "official" and implementation_level != "validated":
        raise ProtocolError("official execution requires a validated implementation")


def validate_result_status(
    implementation_level: str,
    execution_level: str,
    result_status: str,
) -> None:
    validate_run_levels(implementation_level, execution_level)
    if result_status not in {"smoke", "development", "provisional", "official", "legacy"}:
        raise ProtocolError(f"unknown result status: {result_status}")
    if result_status == "official" and (implementation_level, execution_level) != (
        "validated", "official"
    ):
        raise ProtocolError("official results require validated + official")


RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                    "xgboost", "pyarrow", "pvlib", "matplotlib", "PyYAML",
                    "pytest", "shap")


def runtime_provenance(root: Path) -> dict:
    """Capture reproducibility facts, without reading credentials or environment secrets."""
    versions = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                         text=True, check=False, timeout=10)
    if git.returncode:
        raise ProtocolError("Git commit SHA unavailable for experiment provenance")
    return {"git_commit_sha": git.stdout.strip(), "python": sys.version.split()[0],
            "packages": versions, "os": platform.platform(),
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "execution_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def write_experiment_provenance(path: Path, *, root: Path, experiment_id: str) -> dict:
    if not experiment_id or path.exists():
        raise ProtocolError("nonempty experiment ID and new receipt path required")
    payload = {"experiment_id": experiment_id, **runtime_provenance(root)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
