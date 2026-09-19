"""Independent hourly Himawari truth eligibility for returned GFS service points."""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from src.s12_spatial.density import haversine_km
from src.s12_spatial.service_registry import ServicePoint, SpatialContractError, service_point


def assess_himawari_truth(frame: pd.DataFrame, point: ServicePoint, *,
                          start: dt.date, end: dt.date,
                          minimum_valid_fraction: float,
                          maximum_truth_offset_km: float) -> dict:
    """Check hourly truth against an explicitly supplied coverage/offset policy.

    One GHI truth is expected per UTC target hour even when the clean table has
    three leads. GFS and Himawari returned coordinates are audited separately.
    No nearest-grid truth is imputed here.
    """
    if not 0 < minimum_valid_fraction <= 1 or maximum_truth_offset_km < 0:
        raise SpatialContractError("Explicit valid-fraction and nonnegative offset policy required")
    required = {"target_time_utc", "ghi_obs_sat", "gfs_service_latitude",
                "gfs_service_longitude", "himawari_service_latitude",
                "himawari_service_longitude"}
    if required - set(frame):
        raise SpatialContractError(f"Missing Himawari truth columns: {sorted(required - set(frame))}")
    gfs_coords = set(zip(frame.gfs_service_latitude, frame.gfs_service_longitude))
    if gfs_coords != {(point.latitude, point.longitude)}:
        raise SpatialContractError("Frame is not for the selected API-returned GFS point")
    truth_coords = set(zip(frame.himawari_service_latitude, frame.himawari_service_longitude))
    if len(truth_coords) != 1:
        raise SpatialContractError("Himawari service coordinate changed or is missing")
    truth_lat, truth_lon = next(iter(truth_coords))
    truth_point = service_point(truth_lat, truth_lon, "truth")
    offset = haversine_km(point, truth_point)
    utc_hours = pd.date_range(start, end + dt.timedelta(days=1), freq="h", inclusive="left", tz="UTC")
    time = pd.to_datetime(frame.target_time_utc, utc=True)
    grouped = frame.assign(_target=time).groupby("_target")["ghi_obs_sat"]
    truth = {}
    for target, values in grouped:
        finite = pd.to_numeric(values, errors="coerce").dropna().unique()
        if len(finite) > 1:
            raise SpatialContractError(f"Inconsistent Himawari truth across leads at {target}")
        truth[target] = float(finite[0]) if len(finite) else math.nan
    if set(truth) != set(utc_hours):
        raise SpatialContractError("Himawari truth timeline has missing or extra target hours")
    valid_count = sum(math.isfinite(value) and value >= 0 for value in truth.values())
    fraction = valid_count / len(utc_hours)
    return {
        "evidence_schema": "validated_hourly_himawari_v1",
        "gfs_service_id": point.service_id,
        "truth_service_coordinates": {"latitude": truth_point.latitude, "longitude": truth_point.longitude},
        "truth_offset_km": offset,
        "expected_hour_count": len(utc_hours),
        "valid_truth_hour_count": valid_count,
        "valid_fraction": fraction,
        "eligible": fraction >= minimum_valid_fraction and offset <= maximum_truth_offset_km,
        "truth_period_start": start.isoformat(),
        "truth_period_end": end.isoformat(),
        "policy": {"minimum_valid_fraction": minimum_valid_fraction,
                   "maximum_truth_offset_km": maximum_truth_offset_km},
    }
