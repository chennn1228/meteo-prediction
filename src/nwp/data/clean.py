"""Receipt-backed monthly raw-to-clean transformations."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from nwp.core.hashing import content_hash, file_sha256
from nwp.core.paths import RunPaths
from nwp.core.provenance import make_receipt, read_receipt, write_receipt
from nwp.core.schema import ContractError

from .contracts import DataCatalog, DatasetRecord, validate_clean_frame
from .fetch import build_requests, raw_dependency_hash


FCST_RENAME = {
    "shortwave_radiation": "ghi_fcst",
    "diffuse_radiation": "dhi_fcst",
    "direct_normal_irradiance": "dni_fcst",
    "global_tilted_irradiance": "gti_fcst",
    "cloud_cover": "cloud_cover_fcst",
    "temperature_2m": "temp_fcst",
    "relative_humidity_2m": "rh_fcst",
    "dewpoint_2m": "dewpoint_fcst",
    "wind_speed_10m": "wind_speed_fcst",
    "wind_direction_10m": "wind_dir_fcst",
    "surface_pressure": "pressure_fcst",
    "precipitation": "precip_fcst",
    "sunshine_duration": "sunshine_fcst",
    "terrestrial_radiation": "terrestrial_fcst",
    "is_day": "is_day_fcst",
}
SATELLITE_RENAME = {
    "shortwave_radiation": "ghi_obs_sat",
    "direct_radiation": "direct_obs_sat",
    "diffuse_radiation": "dhi_obs_sat",
    "direct_normal_irradiance": "dni_obs_sat",
    "global_tilted_irradiance": "gti_obs_sat",
}
ERA5_RENAME = {
    "shortwave_radiation": "ghi_obs_era5",
    "direct_radiation": "direct_obs_era5",
    "diffuse_radiation": "dhi_obs_era5",
    "direct_normal_irradiance": "dni_obs_era5",
    "global_tilted_irradiance": "gti_obs_era5",
    "cloud_cover": "cloud_cover_obs",
    "cloud_cover_low": "cloud_cover_low_obs",
    "cloud_cover_mid": "cloud_cover_mid_obs",
    "cloud_cover_high": "cloud_cover_high_obs",
}
RADIATION_COLUMNS = (
    "ghi_fcst",
    "dhi_fcst",
    "dni_fcst",
    "gti_fcst",
    "ghi_obs_sat",
    "direct_obs_sat",
    "dhi_obs_sat",
    "dni_obs_sat",
    "gti_obs_sat",
    "ghi_obs_era5",
    "direct_obs_era5",
    "dhi_obs_era5",
    "dni_obs_era5",
    "gti_obs_era5",
    "terrestrial_fcst",
    "sunshine_fcst",
)
CLOUD_COLUMNS = (
    "cloud_cover_fcst",
    "cloud_cover_obs",
    "cloud_cover_low_obs",
    "cloud_cover_mid_obs",
    "cloud_cover_high_obs",
)
CLEAN_CONTRACT = {
    "radiation_negative_policy": "clip_to_zero",
    "cloud_range": [0, 100],
    "missing_policy": "preserve_and_report",
    "row_unit": "site_target_time_lead",
}


def clean_dependency_hash(source_hashes: Mapping[str, str], clean_contract: Mapping[str, object] = CLEAN_CONTRACT) -> str:
    if not source_hashes:
        raise ContractError("clean data requires source hashes")
    return content_hash({"source_hashes": source_hashes, "clean_contract": clean_contract})


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"raw JSON is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("hourly"), dict):
        raise ContractError(f"raw JSON has no hourly object: {path}")
    return payload


def read_previous_runs(path: Path, data_config: Mapping[str, Any]) -> pd.DataFrame:
    payload = _json(path)
    hourly = payload["hourly"]
    target_time = pd.to_datetime(hourly["time"], utc=True)
    frames = []
    for lead_hours in data_config["forecast"]["leads"]:
        lead_days = int(lead_hours) // 24
        frame = pd.DataFrame(
            {
                "target_time_utc": target_time,
                "lead_time": int(lead_hours),
                "source_grid_latitude": payload.get("latitude"),
                "source_grid_longitude": payload.get("longitude"),
                "source_grid_elevation": payload.get("elevation"),
                "gfs_service_latitude": payload.get("latitude"),
                "gfs_service_longitude": payload.get("longitude"),
                "gfs_service_elevation": payload.get("elevation"),
            }
        )
        for variable in data_config["forecast"]["variables"]:
            frame[FCST_RENAME[variable]] = hourly[f"{variable}_previous_day{lead_days}"]
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    output["forecast_issue_time_utc"] = output["target_time_utc"] - pd.to_timedelta(output["lead_time"], unit="h")
    return output


def read_truth(path: Path, rename: Mapping[str, str], service_name: str) -> pd.DataFrame:
    payload = _json(path)
    hourly = payload["hourly"]
    frame = pd.DataFrame({"target_time_utc": pd.to_datetime(hourly["time"], utc=True)})
    frame[f"{service_name}_service_latitude"] = payload["latitude"]
    frame[f"{service_name}_service_longitude"] = payload["longitude"]
    frame[f"{service_name}_service_elevation"] = payload.get("elevation")
    for raw_name, clean_name in rename.items():
        frame[clean_name] = hourly[raw_name]
    return frame


def apply_clean_contract(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in CLOUD_COLUMNS:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").clip(0, 100)
    for column in RADIATION_COLUMNS:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce").clip(lower=0)
    return output


def required_clean_columns(data_config: Mapping[str, Any]) -> tuple[str, ...]:
    columns = tuple(FCST_RENAME[name] for name in data_config["forecast"]["variables"])
    columns += tuple(SATELLITE_RENAME.values()) + tuple(ERA5_RENAME.values())
    columns += (
        "target_time_utc",
        "forecast_issue_time_utc",
        "lead_time",
        "requested_latitude",
        "requested_longitude",
        "gfs_service_latitude",
        "gfs_service_longitude",
        "himawari_service_latitude",
        "himawari_service_longitude",
        "era5_service_latitude",
        "era5_service_longitude",
    )
    return columns


def build_clean_month(
    *,
    site_id: str,
    month: str,
    site_registry: Mapping[str, Mapping[str, Any]],
    data_config: Mapping[str, Any],
    paths: RunPaths,
    catalog: DataCatalog,
    execution_level: str = "development",
) -> DatasetRecord:
    if site_id not in site_registry:
        raise ContractError(f"unknown site: {site_id}")
    try:
        first = dt.date.fromisoformat(f"{month}-01")
    except ValueError as exc:
        raise ContractError(f"invalid month: {month}") from exc
    following = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    last = following - dt.timedelta(days=1)
    requests = build_requests(
        data_config,
        site_registry,
        site_ids=(site_id,),
        start=first,
        end=last,
    )
    raw_records: dict[str, DatasetRecord] = {}
    source_hashes: dict[str, str] = {}
    for request in requests:
        dependency_hash = raw_dependency_hash(request, data_config)
        record = catalog.resolve(
            stage="raw",
            config_hash=dependency_hash,
            sites=(site_id,),
            time_range=f"{first.isoformat()}/{last.isoformat()}",
        )
        if record is None:
            raise ContractError(f"catalog has no ready {request.source} raw partition for {site_id}/{month}")
        raw_records[request.source] = record
        receipt = read_receipt(catalog.receipt_path(record))
        source_hashes[request.source] = receipt["data_sha256"]
    dependency_hash = clean_dependency_hash(source_hashes)
    existing = catalog.resolve(
        stage="clean",
        config_hash=dependency_hash,
        sites=(site_id,),
        time_range=f"{first.isoformat()}/{last.isoformat()}",
    )
    if existing is not None:
        return existing

    previous = read_previous_runs(catalog.dataset_path(raw_records["previous_runs"]), data_config)
    satellite = read_truth(catalog.dataset_path(raw_records["satellite"]), SATELLITE_RENAME, "himawari")
    era5 = read_truth(catalog.dataset_path(raw_records["era5"]), ERA5_RENAME, "era5")
    frame = previous.merge(satellite, on="target_time_utc", how="left").merge(era5, on="target_time_utc", how="left")
    frame = apply_clean_contract(frame)
    site = site_registry[site_id]
    frame.insert(0, "station_id", site_id)
    frame.insert(1, "requested_latitude", float(site["lat"]))
    frame.insert(2, "requested_longitude", float(site["lon"]))
    frame.insert(3, "region", site["region"])
    frame = frame.sort_values(["target_time_utc", "lead_time"]).reset_index(drop=True)
    audit = validate_clean_frame(
        frame,
        start=first,
        end=last,
        required_columns=required_clean_columns(data_config),
        leads=data_config["forecast"]["leads"],
    )

    output_path = paths.processed_partition("clean", dependency_hash, site_id, month)
    receipt_path = paths.data_receipt(output_path)
    if output_path.exists() or receipt_path.exists():
        raise ContractError("unregistered clean output already exists and will not be overwritten")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    output_hash = file_sha256(output_path)
    receipt = make_receipt(
        root=paths.root,
        stage="clean",
        config_hash=dependency_hash,
        execution_level=execution_level,
        status="ready",
        input_hashes=source_hashes,
        output_hashes={"data": output_hash},
        site_id=site_id,
        month=month,
        time_range={"start": first.isoformat(), "end": last.isoformat()},
        row_count=audit["row_count"],
        missingness=audit["missingness"],
        service_coordinates=audit["service_coordinates"],
        clean_contract=CLEAN_CONTRACT,
    )
    write_receipt(receipt_path, receipt)
    record = DatasetRecord(
        dataset_id=f"clean-{site_id}-{month}-{dependency_hash}",
        stage="clean",
        config_hash=dependency_hash,
        sites=(site_id,),
        time_range=f"{first.isoformat()}/{last.isoformat()}",
        source_hashes=source_hashes,
        status="ready",
        path=output_path.relative_to(paths.data_root).as_posix(),
        receipt_path=receipt_path.relative_to(paths.data_root).as_posix(),
    )
    catalog.register(record)
    return record
