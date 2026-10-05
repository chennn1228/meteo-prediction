"""Stable byte-, source-, environment-, and dependency fingerprints."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any, Iterable

from .hashing import canonical_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_object_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_directory(path: Path) -> str:
    root = Path(path)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    return stable_object_hash([
        {"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)}
        for item in files
    ])


def source_files_fingerprint(root: Path, relative_paths: Iterable[str]) -> str:
    root = Path(root)
    records = []
    for relative in sorted(set(relative_paths)):
        target = root / relative
        if target.is_dir():
            value = sha256_directory(target)
        elif target.is_file():
            value = sha256_file(target)
        else:
            raise FileNotFoundError(target)
        records.append({"path": Path(relative).as_posix(), "sha256": value})
    return stable_object_hash(records)


def runtime_source_fingerprint(root: Path) -> str:
    """Fingerprint every active Python source file, excluding docs and generated data."""
    source_root = Path(root) / "src" / "nwp"
    return source_files_fingerprint(
        root,
        (path.relative_to(root).as_posix() for path in source_root.rglob("*.py")),
    )


def environment_fingerprint(packages: Iterable[str]) -> str:
    versions = {"python": __import__("platform").python_version()}
    for package in sorted(set(packages)):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return stable_object_hash(versions)


def run_fingerprints(root: Path, *, scientific_run_hash: str, protocol: Any, data: Any, feature_build: Any,
                     analysis: Any, validation: Any, sites: Any,
                     models: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    feature_code = source_files_fingerprint(root, (
        "src/nwp/features/build.py", "src/nwp/features/engineering.py",
        "src/nwp/features/physics.py", "src/nwp/features/preprocessing.py",
        "src/nwp/core/schema.py", "src/nwp/data/contracts.py"))
    split_code = source_files_fingerprint(root, ("src/nwp/splits",))
    analysis_code = source_files_fingerprint(root, ("src/nwp/evaluation",))
    visualization_code = source_files_fingerprint(root, ("src/nwp/visualization",))
    tuning_code = source_files_fingerprint(root, (
        "src/nwp/experiment/tuning.py", "src/nwp/evaluation/metrics.py"))
    evaluation_code = source_files_fingerprint(root, ("src/nwp/evaluation",))
    model_hashes = {}
    for model_id, record in models.items():
        family = record.get("family")
        implementation = ("src/nwp/models/deep" if family in {"deep", "experimental_constrained"}
                          else "src/nwp/models/trees.py" if family == "tree_ml"
                          else "src/nwp/models/statistical.py" if family == "statistical"
                          else "src/nwp/models/baselines.py")
        model_hashes[model_id] = stable_object_hash({
            "config": record,
            "code": source_files_fingerprint(root, (
                "src/nwp/models/base.py", "src/nwp/models/factory.py",
                "src/nwp/features/preprocessing.py", implementation)),
        })
    return {
        "scientific_run_hash": scientific_run_hash,
        "site_selection_hash": stable_object_hash(sites),
        "data_contract_hash": stable_object_hash(data),
        "feature_build_hash": stable_object_hash({"config": feature_build, "code": feature_code}),
        "validation_hash": stable_object_hash({"config": validation, "code": split_code}),
        "analysis_hash": stable_object_hash({"config": analysis, "code": analysis_code}),
        "visualization_hash": stable_object_hash({"code": visualization_code}),
        "tuning_hash": stable_object_hash({"code": tuning_code}),
        "evaluation_hash": stable_object_hash({"code": evaluation_code}),
        "model_hash": model_hashes,
    }
