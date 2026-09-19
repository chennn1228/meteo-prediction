"""Minimal shared identity and time-semantic constraints."""
from __future__ import annotations

from .config_loader import ProtocolError


IDENTITY_FIELDS = frozenset({
    "station_id", "location_id", "source_grid_id", "requested_location_id",
    "gfs_service_id", "truth_service_id",
})
TRUTH_FIELDS = frozenset({"ghi_obs_sat", "ghi_obs_era5", "cloud_cover_obs", "y"})


def assert_model_features(columns: list[str] | tuple[str, ...]) -> None:
    forbidden = set(columns) & IDENTITY_FIELDS
    forbidden |= {name for name in columns if name.startswith("station_") or name.endswith("_target_encoded")}
    forbidden |= {name for name in columns if name in TRUTH_FIELDS or name.endswith("_obs")}
    if forbidden:
        raise ProtocolError(f"identity or truth fields cannot enter model inputs: {sorted(forbidden)}")
