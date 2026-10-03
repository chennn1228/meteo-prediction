"""The single runtime context passed to every workflow stage."""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nwp.data.contracts import DataCatalog

from .config import RunConfig, load_local_paths
from .paths import RunPaths
from .provenance import write_run_provenance
from .schema import ContractError, StageResult


@dataclass
class RunContext:
    config: RunConfig
    paths: RunPaths
    catalog: DataCatalog
    run_id: str
    selected_sites: tuple[str, ...]
    selected_models: tuple[str, ...]
    execution_level: str
    allow_model_execution: bool = False
    stage_results: dict[str, dict[str, Any]] = field(default_factory=dict)

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
            catalog=DataCatalog(paths),
            run_id=run_id,
            selected_sites=config.selected_sites,
            selected_models=config.selected_models,
            execution_level=config.execution,
            allow_model_execution=allow_model_execution,
        )
        (paths.meta_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config.as_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        write_run_provenance(
            paths.meta_dir / "provenance.json",
            root=root,
            run_id=run_id,
            config_hash=config.config_hash,
            execution_level=config.execution,
        )
        (paths.meta_dir / "stage_status.json").write_text("{}\n", encoding="utf-8")
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
        status_path = paths.meta_dir / "stage_status.json"
        if not all(path.is_file() for path in (
                resolved_path, provenance_path, status_path)):
            raise ContractError("run metadata is incomplete and cannot be resumed")
        resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if resolved != config.as_dict():
            raise ContractError(
                "current resolved configuration differs from the saved run")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = {
            "run_id": run_id, "config_hash": config.config_hash,
            "execution_level": config.execution}
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ContractError("run provenance does not match the requested run")
        stage_results = json.loads(status_path.read_text(encoding="utf-8"))
        if not isinstance(stage_results, dict):
            raise ContractError("stage status must be a JSON object")
        return cls(
            config=config, paths=paths, catalog=DataCatalog(paths), run_id=run_id,
            selected_sites=config.selected_sites,
            selected_models=config.selected_models,
            execution_level=config.execution,
            allow_model_execution=allow_model_execution,
            stage_results=stage_results)

    def _write_stage_status(self) -> None:
        temporary = self.paths.meta_dir / ".stage_status.json.tmp"
        target = self.paths.meta_dir / "stage_status.json"
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

    def clear_stages(self, stages: tuple[str, ...]) -> None:
        """Forget explicitly invalidated stage records; artifacts remain recoverable."""
        changed = False
        for stage in stages:
            changed = self.stage_results.pop(stage, None) is not None or changed
        if changed:
            self._write_stage_status()

    def record_stage(self, stage: str, result: StageResult) -> None:
        if stage in self.stage_results:
            raise ContractError(f"stage result already recorded: {stage}")
        if result.config_hash != self.config.config_hash:
            raise ContractError("stage result config hash does not match the run")
        self.stage_results[stage] = result.as_dict()
        self._write_stage_status()
