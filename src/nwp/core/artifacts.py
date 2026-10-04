"""Typed artifact records and strict cross-run resolution."""
from __future__ import annotations

import json
import shutil
import datetime as dt
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .fingerprints import sha256_file
from .lifecycle import require_writable
from .paths import RunPaths
from .schema import ContractError

TIME_SCOPES = frozenset({"development", "outer_validation", "calibration", "final_test", "diagnostic"})
SELECTION_STAGES = frozenset({"tuning", "model_selection", "early_stopping", "feature_selection", "selection_interpretation"})


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    stage: str
    model_id: str | None
    fold_id: str | None
    scope: Mapping[str, Any]
    data_scope: Mapping[str, Any]
    time_scope: str
    split_scope: Mapping[str, Any]
    dependency_fingerprint: str
    implementation_fingerprint: str
    environment_fingerprint: str
    sha256: str
    path: str
    portable: bool
    execution_level: str
    result_status: str
    official_reuse_eligible: bool
    created_at: str

    def __post_init__(self) -> None:
        if self.time_scope not in TIME_SCOPES:
            raise ContractError(f"invalid artifact time_scope: {self.time_scope}")
        if self.stage in SELECTION_STAGES and self.time_scope == "final_test":
            raise ContractError("final-test artifacts cannot enter selection")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArtifactResolver:
    def __init__(self, paths: RunPaths):
        self.paths = paths

    def _records(self):
        for execution in ("development", "official"):
            for manifest in sorted((self.paths.outputs_root / execution).glob("*/00_meta/artifact_manifest.json")):
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                for item in payload.get("artifacts", []):
                    yield manifest, ArtifactRecord(**item)

    def find_compatible_artifact(self, *, artifact_type: str,
                                 dependency_fingerprint: str,
                                 implementation_fingerprint: str,
                                 data_scope: Mapping[str, Any], time_scope: str,
                                 split_scope: Mapping[str, Any], execution_level: str,
                                 environment_fingerprint: str) -> tuple[Path, ArtifactRecord] | None:
        if time_scope not in TIME_SCOPES:
            raise ContractError("invalid requested time scope")
        matches = []
        for manifest, record in self._records():
            checks = (
                record.artifact_type == artifact_type,
                record.dependency_fingerprint == dependency_fingerprint,
                record.implementation_fingerprint == implementation_fingerprint,
                dict(record.data_scope) == dict(data_scope), record.time_scope == time_scope,
                dict(record.split_scope) == dict(split_scope),
                record.environment_fingerprint == environment_fingerprint,
                execution_level != "official" or record.official_reuse_eligible,
                execution_level != "official" or record.execution_level == "official" or record.artifact_type in {"raw", "clean", "features"},
                execution_level != "official" or record.artifact_type not in {"raw", "clean", "features"}
                or (record.official_reuse_eligible
                    and bool(record.scope.get("provenance_complete"))
                    and bool(record.scope.get("receipt_complete"))
                    and bool(record.scope.get("contract_match"))),
            )
            if not all(checks):
                continue
            candidate = (manifest.parents[1] / record.path).resolve()
            if not candidate.is_file() or sha256_file(candidate) != record.sha256:
                continue
            matches.append((candidate, record))
        if len(matches) > 1:
            raise ContractError("ambiguous compatible artifacts")
        return matches[0] if matches else None

    @staticmethod
    def materialize(source: Path, destination: Path, expected_sha256: str, *,
                    reused_from_run: str, source_artifact_id: str,
                    source_dependency_fingerprint: str,
                    source_execution_level: str) -> dict[str, Any]:
        if destination.exists():
            raise ContractError("artifact destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(source) != expected_sha256 or sha256_file(destination) != expected_sha256:
            destination.unlink(missing_ok=True)
            raise ContractError("copied artifact SHA mismatch")
        return {
            "reused_from_run": reused_from_run,
            "source_artifact_id": source_artifact_id,
            "source_sha256": expected_sha256,
            "source_dependency_fingerprint": source_dependency_fingerprint,
            "source_execution_level": source_execution_level,
            "reuse_method": "copy",
        }

    def append_record(self, record: ArtifactRecord) -> None:
        require_writable(self.paths.meta_dir / "status.json")
        manifest = self.paths.meta_dir / "artifact_manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload.get("artifacts")
        if not isinstance(records, list):
            raise ContractError("artifact manifest is malformed")
        existing = [item for item in records if item.get("artifact_id") == record.artifact_id]
        if existing:
            if existing != [record.as_dict()]:
                raise ContractError("artifact ID already has a different committed record")
            return
        target = (self.paths.run_root / record.path).resolve()
        try:
            target.relative_to(self.paths.run_root)
        except ValueError as exc:
            raise ContractError("artifact path escapes its run") from exc
        if not target.is_file() or sha256_file(target) != record.sha256:
            raise ContractError("artifact record does not match persisted bytes")
        records.append(record.as_dict())
        temporary = manifest.with_name(".artifact_manifest.json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest)


def environment_packages(artifact_type: str, implementation: str | None = None) -> tuple[str, ...]:
    base = ("numpy", "pandas", "pyarrow", "pvlib")
    if artifact_type in {"raw", "clean", "features"}:
        return base
    if implementation == "lightgbm":
        return base + ("scikit-learn", "lightgbm")
    if implementation == "xgboost":
        return base + ("scikit-learn", "xgboost")
    if implementation in {"deep", "pinn"}:
        return base + ("torch",)
    if artifact_type == "shap":
        return base + ("shap",)
    return base + ("scikit-learn",)


def new_artifact_record(**values: Any) -> ArtifactRecord:
    values.setdefault("created_at", dt.datetime.now(dt.timezone.utc).isoformat())
    return ArtifactRecord(**values)
