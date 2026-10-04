"""One validated provenance and receipt mechanism for every artifact."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .schema import ContractError


REQUIRED_RECEIPT_KEYS = frozenset(
    {
        "stage",
        "config_hash",
        "input_hashes",
        "output_hashes",
        "git_commit",
        "execution_level",
        "status",
        "created_at",
    }
)
RECEIPT_STATUSES = frozenset({
    "success", "blocked", "failed", "reused", "skipped", "ready",
    "quarantined"})
EXECUTION_LEVELS = frozenset({"smoke", "development", "official"})
RUNTIME_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "pyarrow",
    "pvlib",
    "matplotlib",
    "PyYAML",
    "pytest",
    "shap",
)
_HEX = re.compile(r"^[0-9a-f]{8,64}$")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode:
        raise ContractError(f"Git provenance command failed: git {' '.join(args)}")
    return result.stdout.strip()


def git_commit(root: Path) -> str:
    commit = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ContractError("Git commit SHA is invalid")
    return commit


def runtime_provenance(root: Path) -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = _git(root, "status", "--porcelain")
    return {
        "git_commit": git_commit(root),
        "git_branch": _git(root, "branch", "--show-current") or None,
        "git_dirty": bool(status),
        "python": sys.version.split()[0],
        "packages": versions,
        "os": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _hash_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ContractError(f"receipt {label} must be a mapping")
    output: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or not key or not isinstance(digest, str) or not _HEX.fullmatch(digest):
            raise ContractError(f"receipt {label} must contain named lowercase hex hashes")
        output[key] = digest
    return output


def make_receipt(
    *,
    root: Path,
    stage: str,
    config_hash: str,
    execution_level: str,
    status: str,
    input_hashes: Mapping[str, str] | None = None,
    output_hashes: Mapping[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    overlap = REQUIRED_RECEIPT_KEYS & set(extra)
    if overlap:
        raise ContractError(f"receipt extra fields cannot replace required fields: {sorted(overlap)}")
    receipt = {
        "stage": stage,
        "config_hash": config_hash,
        "input_hashes": dict(input_hashes or {}),
        "output_hashes": dict(output_hashes or {}),
        "git_commit": git_commit(root),
        "execution_level": execution_level,
        "status": status,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **extra,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping):
        raise ContractError("receipt must be a mapping")
    missing = REQUIRED_RECEIPT_KEYS - set(receipt)
    if missing:
        raise ContractError(f"receipt is missing required fields: {sorted(missing)}")
    if not isinstance(receipt["stage"], str) or not receipt["stage"]:
        raise ContractError("receipt stage must be nonempty")
    if not isinstance(receipt["config_hash"], str) or not _HEX.fullmatch(receipt["config_hash"]):
        raise ContractError("receipt config_hash must be a lowercase hex hash")
    _hash_mapping(receipt["input_hashes"], "input_hashes")
    _hash_mapping(receipt["output_hashes"], "output_hashes")
    if not isinstance(receipt["git_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", receipt["git_commit"]):
        raise ContractError("receipt git_commit must be a full SHA-1")
    if receipt["execution_level"] not in EXECUTION_LEVELS:
        raise ContractError(f"invalid receipt execution_level: {receipt['execution_level']}")
    if receipt["status"] not in RECEIPT_STATUSES:
        raise ContractError(f"invalid receipt status: {receipt['status']}")
    try:
        created = dt.datetime.fromisoformat(str(receipt["created_at"]))
    except ValueError as exc:
        raise ContractError("receipt created_at must be ISO-8601") from exc
    if created.tzinfo is None:
        raise ContractError("receipt created_at must include a timezone")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ContractError(f"provenance artifact already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise ContractError(f"stale temporary provenance artifact exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    validate_receipt(receipt)
    _write_new_json(path, receipt)


def write_provenance(path: Path, payload: Mapping[str, Any]) -> None:
    """Write any non-receipt provenance record with the same no-overwrite rule."""
    _write_new_json(path, payload)


def read_receipt(path: Path) -> dict[str, Any]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read receipt: {path}") from exc
    validate_receipt(receipt)
    return receipt


def write_run_provenance(path: Path, *, root: Path, run_id: str, config_hash: str,
                         execution_level: str, parent_run_id: str | None = None,
                         change_reason: str | None = None,
                         changed_dependencies: list[str] | None = None) -> dict[str, Any]:
    if not run_id:
        raise ContractError("run_id is required for run provenance")
    payload = {
        "run_id": run_id,
        "config_hash": config_hash,
        "execution_level": execution_level,
        "parent_run_id": parent_run_id,
        "change_reason": change_reason,
        "changed_dependencies": list(changed_dependencies or []),
        **runtime_provenance(root),
    }
    _write_new_json(path, payload)
    return payload
