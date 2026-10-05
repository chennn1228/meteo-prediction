"""The single runtime context passed to every workflow stage."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nwp.data.contracts import DataCatalog

from .artifacts import ArtifactResolver
from .config import (RunConfig, load_local_paths, read_resolved_run_config,
                     write_resolved_run_config)
from .lifecycle import read_state, require_writable, transition, write_initial_state
from .fingerprints import runtime_source_fingerprint
from .paths import RunPaths
from .provenance import write_run_provenance
from .schema import ContractError, StageResult


@dataclass
class RunContext:
    config: RunConfig
    paths: RunPaths
    catalog: DataCatalog
    artifact_resolver: ArtifactResolver
    run_id: str
    selected_sites: tuple[str, ...]
    selected_models: tuple[str, ...]
    execution_level: str
    run_type: str = "executable"
    lifecycle_state: str = "RUNNING"
    provenance: dict[str, Any] = field(default_factory=dict)
    allow_model_execution: bool = False
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_stage: str | None = None

    @classmethod
    def create(
        cls,
        root: Path,
        config: RunConfig,
        *,
        allow_model_execution: bool = False,
        data_root: Path | str | None = None,
        outputs_root: Path | str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        change_reason: str | None = None,
        changed_dependencies: list[str] | None = None,
    ) -> "RunContext":
        root = root.resolve()
        local_paths = load_local_paths(root)
        if run_id is None:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            run_id = f"{stamp}_{config.profile}_{config.config_hash}"
        paths = RunPaths.create(
            root,
            execution=config.execution,
            run_id=run_id,
            data_root=data_root or local_paths["data_root"],
            outputs_root=outputs_root or local_paths["outputs_root"],
        )
        paths.initialize_run()
        context = cls(
            config=config,
            paths=paths,
            catalog=DataCatalog(paths), artifact_resolver=ArtifactResolver(paths),
            run_id=run_id,
            selected_sites=config.selected_sites,
            selected_models=config.selected_models,
            execution_level=config.execution,
            allow_model_execution=allow_model_execution,
        )
        write_resolved_run_config(paths.meta_dir / "resolved_config.yaml", config)
        provenance = write_run_provenance(
            paths.meta_dir / "provenance.json",
            root=root,
            run_id=run_id,
            config_hash=config.config_hash,
            execution_level=config.execution,
            parent_run_id=parent_run_id, change_reason=change_reason,
            changed_dependencies=changed_dependencies,
            site_registry_hash=config.site_registry_hash,
            model_registry_hash=config.model_registry_hash,
            runtime_source_fingerprint=runtime_source_fingerprint(root),
        )
        (paths.meta_dir / "artifact_manifest.json").write_text(
            '{"artifacts": []}\n', encoding="utf-8")
        write_initial_state(paths.meta_dir / "status.json")
        if config.execution != "official":
            transition(paths.meta_dir / "status.json", "RUNNING")
            context.lifecycle_state = "RUNNING"
        else:
            context.lifecycle_state = "CREATED"
        context.provenance = provenance
        return context

    @classmethod
    def resume(
        cls,
        root: Path,
        config: RunConfig,
        *,
        run_id: str,
        allow_model_execution: bool = False,
        data_root: Path | str | None = None,
        outputs_root: Path | str | None = None,
    ) -> "RunContext":
        """Resume one exact run after verifying immutable configuration metadata."""
        root = root.resolve()
        local_paths = load_local_paths(root)
        paths = RunPaths.create(
            root, execution=config.execution, run_id=run_id,
            data_root=data_root or local_paths["data_root"],
            outputs_root=outputs_root or local_paths["outputs_root"])
        if not paths.run_root.is_dir():
            raise ContractError(f"run does not exist: {paths.run_root}")
        resolved_path = paths.meta_dir / "resolved_config.yaml"
        provenance_path = paths.meta_dir / "provenance.json"
        status_path = paths.meta_dir / "status.json"
        if not all(path.is_file() for path in (
                resolved_path, provenance_path, status_path)):
            raise ContractError("run metadata is incomplete and cannot be resumed")
        resolved = read_resolved_run_config(resolved_path)
        if resolved != config.as_dict():
            raise ContractError(
                "current resolved configuration differs from the saved run")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("runtime_source_fingerprint") != runtime_source_fingerprint(root):
            raise ContractError(
                "runtime source fingerprint differs from the saved run; create a child run")
        expected = {
            "run_id": run_id, "config_hash": config.config_hash,
            "execution_level": config.execution}
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ContractError("run provenance does not match the requested run")
        state = read_state(status_path)
        if state != "RUNNING":
            raise ContractError(f"only RUNNING runs may resume; {state} is read-only")
        stage_results_path = paths.meta_dir / "stage_results.json"
        stage_results = (json.loads(stage_results_path.read_text(encoding="utf-8"))
                         if stage_results_path.exists() else {})
        return cls(
            config=config, paths=paths, catalog=DataCatalog(paths),
            artifact_resolver=ArtifactResolver(paths), run_id=run_id,
            selected_sites=config.selected_sites,
            selected_models=config.selected_models,
            execution_level=config.execution,
            allow_model_execution=allow_model_execution, lifecycle_state=state,
            provenance=provenance,
            stage_results=stage_results)

    def _write_stage_status(self) -> None:
        require_writable(self.paths.meta_dir / "status.json")
        temporary = self.paths.meta_dir / ".stage_results.json.tmp"
        target = self.paths.meta_dir / "stage_results.json"
        if temporary.exists():
            raise ContractError(f"stale stage-status temporary file exists: {temporary}")
        try:
            temporary.write_text(
                json.dumps(self.stage_results, ensure_ascii=False, indent=2,
                           sort_keys=True) + "\n",
                encoding="utf-8")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_stage(self, stage: str, result: StageResult) -> None:
        require_writable(self.paths.meta_dir / "status.json")
        if stage in self.stage_results:
            raise ContractError(f"stage result already recorded: {stage}")
        self.stage_results[stage] = result.as_dict()
        self._write_stage_status()
