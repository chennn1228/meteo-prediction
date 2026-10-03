"""Receipt-backed clean-to-feature transformation."""
from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

import pandas as pd

from nwp.core.hashing import content_hash, file_sha256
from nwp.core.paths import RunPaths
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError
from nwp.data.contracts import DataCatalog, DatasetRecord

from .engineering import build_forecast_features, formal_feature_columns
from .physics import add_returned_service_physics


def feature_dependency_hash(clean_hash: str, feature_config: Mapping[str, Any]) -> str:
    if not clean_hash:
        raise ContractError("feature data requires a clean content hash")
    return content_hash({"clean_hash": clean_hash, "feature_config": feature_config})


def prepare_formal_features(
    cleaned: pd.DataFrame,
    *,
    data_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if "fcst_issue_time_utc" in cleaned:
        raise ContractError(
            "issue-time alias is unsupported; use forecast_issue_time_utc"
        )
    physical = add_returned_service_physics(cleaned)
    featured = build_forecast_features(
        physical, data_config=data_config, feature_config=feature_config
    )
    columns = tuple(formal_feature_columns(featured, feature_config))
    return featured, columns


def build_features(
    frame: pd.DataFrame,
    *,
    data_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
) -> pd.DataFrame:
    return prepare_formal_features(
        frame, data_config=data_config, feature_config=feature_config
    )[0]


def configured_feature_columns(feature_config: Mapping[str, Any]) -> list[str]:
    return [
        column
        for columns in feature_config["feature_groups"].values()
        for column in columns
    ]


def build_feature_month(
    *,
    site_id: str,
    month: str,
    clean_config_hash: str,
    data_config: Mapping[str, Any],
    feature_config: Mapping[str, Any],
    paths: RunPaths,
    catalog: DataCatalog,
    execution_level: str = "development",
) -> DatasetRecord:
    try:
        first = dt.date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise ContractError(f"invalid month: {month}") from exc
    following = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    last = following - dt.timedelta(days=1)
    time_range = f"{first.isoformat()}/{last.isoformat()}"
    clean_record = catalog.resolve(
        stage="clean",
        config_hash=clean_config_hash,
        sites=(site_id,),
        time_range=time_range,
    )
    if clean_record is None:
        raise ContractError(
            f"catalog has no ready clean partition for {site_id}/{month}/{clean_config_hash}"
        )
    clean_path = catalog.dataset_path(clean_record)
    clean_receipt = read_receipt(catalog.receipt_path(clean_record))
    clean_hash = file_sha256(clean_path)
    if clean_hash not in clean_receipt["output_hashes"].values():
        raise ContractError("clean content hash is absent from its receipt")
    dependency_hash = feature_dependency_hash(clean_hash, feature_config)
    existing = catalog.resolve(
        stage="features",
        config_hash=dependency_hash,
        sites=(site_id,),
        time_range=time_range,
    )
    if existing is not None:
        return existing

    frame = pd.read_parquet(clean_path)
    featured, columns = prepare_formal_features(
        frame, data_config=data_config, feature_config=feature_config
    )
    output_path = paths.processed_partition("features", dependency_hash, site_id, month)
    receipt_path = paths.data_receipt(output_path)
    if output_path.exists() or receipt_path.exists():
        raise ContractError(
            "unregistered feature output already exists and will not be overwritten"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        featured.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_hash = file_sha256(output_path)
    input_hashes = {"clean": clean_hash}
    receipt = make_receipt(
        root=paths.root,
        stage="features",
        config_hash=dependency_hash,
        execution_level=execution_level,
        status="ready",
        input_hashes=input_hashes,
        output_hashes={"data": output_hash},
        site_id=site_id,
        month=month,
        time_range={"start": first.isoformat(), "end": last.isoformat()},
        row_count=len(featured),
        formal_feature_columns=list(columns),
        feature_version=feature_config["version"],
        data_version=data_config["version"],
        issue_time_field="forecast_issue_time_utc",
        physical_coordinate_basis="gfs_service_coordinates",
    )
    write_receipt(receipt_path, receipt)
    record = DatasetRecord(
        dataset_id=f"features-{site_id}-{month}-{dependency_hash}",
        stage="features",
        config_hash=dependency_hash,
        sites=(site_id,),
        time_range=time_range,
        source_hashes=input_hashes,
        status="ready",
        path=output_path.relative_to(paths.data_root).as_posix(),
        receipt_path=receipt_path.relative_to(paths.data_root).as_posix(),
    )
    catalog.register(record)
    return record
