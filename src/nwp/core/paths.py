"""The only constructors for project data and per-run artifact paths."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .schema import ContractError


RUN_STAGE_DIRECTORIES = {
    "validate": "00_meta",
    "selection": "01_selection",
    "data": "00_meta",
    "features": "00_meta",
    "splits": "01_selection",
    "tuning": "02_tuning",
    "fitting": "03_models",
    "prediction": "04_predictions",
    "calibration": "05_calibration",
    "evaluation": "06_metrics",
    "analysis": "07_analysis",
    "figures": "08_figures",
    "report": "09_report",
}
RUN_DIRECTORIES = tuple(dict.fromkeys(RUN_STAGE_DIRECTORIES.values()))
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _segment(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_SEGMENT.fullmatch(value):
        raise ContractError(f"unsafe {label} path segment: {value!r}")
    return value


def _resolved_path(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


@dataclass(frozen=True)
class RunPaths:
    root: Path
    data_root: Path
    outputs_root: Path
    run_root: Path

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        execution: str,
        run_id: str,
        data_root: Path | str = "data",
        outputs_root: Path | str = "outputs",
    ) -> "RunPaths":
        root = root.resolve()
        if execution not in {"development", "official"}:
            raise ContractError("run outputs support only development or official execution")
        safe_run_id = _segment(run_id, "run_id")
        resolved_data = _resolved_path(root, data_root)
        resolved_outputs = _resolved_path(root, outputs_root)
        return cls(
            root=root,
            data_root=resolved_data,
            outputs_root=resolved_outputs,
            run_root=resolved_outputs / execution / safe_run_id,
        )

    def initialize_run(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=False)
        for directory in RUN_DIRECTORIES:
            (self.run_root / directory).mkdir()

    def stage_dir(self, stage: str) -> Path:
        try:
            directory = RUN_STAGE_DIRECTORIES[stage]
        except KeyError as exc:
            raise ContractError(f"unknown workflow stage: {stage}") from exc
        return self.run_root / directory

    @property
    def meta_dir(self) -> Path:
        return self.run_root / "00_meta"

    @property
    def selection_dir(self) -> Path:
        return self.run_root / "01_selection"

    @property
    def tuning_dir(self) -> Path:
        return self.run_root / "02_tuning"

    @property
    def models_dir(self) -> Path:
        return self.run_root / "03_models"

    @property
    def predictions_dir(self) -> Path:
        return self.run_root / "04_predictions"

    @property
    def calibration_dir(self) -> Path:
        return self.run_root / "05_calibration"

    @property
    def metrics_dir(self) -> Path:
        return self.run_root / "06_metrics"

    @property
    def analysis_dir(self) -> Path:
        return self.run_root / "07_analysis"

    @property
    def figures_dir(self) -> Path:
        return self.run_root / "08_figures"

    @property
    def report_dir(self) -> Path:
        return self.run_root / "09_report"

    @property
    def catalog_path(self) -> Path:
        return self.data_root / "catalog.json"

    @property
    def data_registry_dir(self) -> Path:
        return self.data_root / "registry"

    def geography_path(self) -> Path:
        return self.data_registry_dir / "geography" / "jiangsu.geojson"

    def service_probe_root(self) -> Path:
        return self.data_registry_dir / "service_probes"

    def site_selection_root(self) -> Path:
        return self.data_registry_dir / "site_selections"

    def raw_partition(self, source: str, site_id: str, month: str, *, suffix: str = ".parquet") -> Path:
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise ContractError(f"unsafe partition suffix: {suffix!r}")
        return self.data_root / "raw" / _segment(source, "source") / _segment(site_id, "site_id") / f"{_segment(month, 'month')}{suffix}"

    def processed_partition(self, stage: str, dependency_hash: str, site_id: str, month: str) -> Path:
        if stage not in {"clean", "features"}:
            raise ContractError("processed data stage must be clean or features")
        return self.data_root / stage / _segment(dependency_hash, "dependency_hash") / _segment(site_id, "site_id") / f"{_segment(month, 'month')}.parquet"

    def data_receipt(self, data_path: Path) -> Path:
        resolved = data_path.resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ContractError("data receipt target must stay under data_root") from exc
        return resolved.with_suffix(".receipt.json")

    def tuning_fold(self, model_id: str, fold_id: str) -> Path:
        return self.tuning_dir / _segment(model_id, "model_id") / _segment(fold_id, "fold_id")

    def model_fold(self, model_id: str, fold_id: str) -> Path:
        return self.models_dir / _segment(model_id, "model_id") / _segment(fold_id, "fold_id")

    def prediction_file(self, model_id: str, scope: str) -> Path:
        return (self.predictions_dir / _segment(model_id, "model_id") /
                _segment(scope, "scope") / "predictions.parquet")

    def calibration_file(self, model_id: str, scope: str) -> Path:
        return (self.calibration_dir / _segment(model_id, "model_id") /
                _segment(scope, "scope") / "calibration.parquet")

    def model_metrics_dir(self, model_id: str) -> Path:
        return self.metrics_dir / _segment(model_id, "model_id")

    def aggregate_metrics_dir(self) -> Path:
        return self.metrics_dir / "aggregate"

    def model_analysis_dir(self, model_id: str) -> Path:
        return self.analysis_dir / _segment(model_id, "model_id")

    def aggregate_analysis_dir(self) -> Path:
        return self.analysis_dir / "aggregate"

    def stage_receipt(self, stage: str) -> Path:
        return self.stage_dir(stage) / f"{_segment(stage, 'stage')}.receipt.json"
