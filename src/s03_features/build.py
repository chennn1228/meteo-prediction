"""File-level formal feature build; no model fitting or result promotion."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from s01_core.config_loader import load_manifest
from .pipeline import prepare_formal_features


def build_feature_file(clean_path: Path, output_path: Path) -> dict:
    clean_path = Path(clean_path)
    output_path = Path(output_path)
    frame = pd.read_parquet(clean_path)
    featured, columns = prepare_formal_features(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    featured.to_parquet(temporary, index=False)
    temporary.replace(output_path)
    manifest = load_manifest()
    receipt = {
        "status": "development_feature_build_not_official_result",
        "source_clean_path": str(clean_path.resolve()),
        "output_feature_path": str(output_path.resolve()),
        "rows": len(featured),
        "formal_feature_columns": list(columns),
        "feature_version": manifest["feature_version"],
        "data_version": manifest["data_version"],
        "protocol_version": manifest["protocol_version"],
        "issue_time_field": "forecast_issue_time_utc",
        "physical_coordinate_basis": "gfs_service_coordinates",
    }
    output_path.with_name(output_path.name + ".receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt
