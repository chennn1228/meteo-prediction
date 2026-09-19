"""Solar geometry at the API-returned GFS service point, never the request."""
from __future__ import annotations

import numpy as np
import pandas as pd

from s01_core.config_loader import ProtocolError


def add_returned_service_physics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"target_time_utc", "gfs_service_latitude", "gfs_service_longitude",
                "gfs_service_elevation"}
    missing = required - set(frame)
    if missing:
        raise ProtocolError(f"returned GFS service coordinates required: {sorted(missing)}")
    data = frame.copy()
    coordinates = data[["gfs_service_latitude", "gfs_service_longitude",
                        "gfs_service_elevation"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ProtocolError("GFS returned service coordinates/elevation must be finite")
    if (coordinates.gfs_service_latitude.abs() > 90).any() or (coordinates.gfs_service_longitude.abs() > 180).any():
        raise ProtocolError("GFS returned service coordinate outside geographic bounds")
    target = pd.to_datetime(data["target_time_utc"], errors="coerce", utc=True)
    if target.isna().any():
        raise ProtocolError("invalid target_time_utc for solar geometry")
    data["source_grid_latitude"] = coordinates.gfs_service_latitude
    data["source_grid_longitude"] = coordinates.gfs_service_longitude
    data["source_grid_elevation"] = coordinates.gfs_service_elevation
    data["location_id"] = [f"om-gfs-land-{lat:.6f}-{lon:.6f}" for lat, lon in
                           zip(coordinates.gfs_service_latitude, coordinates.gfs_service_longitude)]
    for column in ("solar_elevation", "solar_azimuth", "ghi_clear_sky", "dni_clear_sky"):
        data[column] = np.nan
    # pvlib is imported only when physics is requested; no fetch or training occurs.
    import pvlib
    for (latitude, longitude, elevation), positions in coordinates.groupby(
            ["gfs_service_latitude", "gfs_service_longitude", "gfs_service_elevation"],
            sort=False).groups.items():
        times = pd.DatetimeIndex(target.loc[positions])
        site = pvlib.location.Location(latitude=float(latitude), longitude=float(longitude),
                                       altitude=float(elevation), tz="UTC")
        solar = site.get_solarposition(times)
        clear = site.get_clearsky(times, model="ineichen")
        data.loc[positions, "solar_elevation"] = solar["apparent_elevation"].to_numpy()
        data.loc[positions, "solar_azimuth"] = solar["azimuth"].to_numpy()
        data.loc[positions, "ghi_clear_sky"] = clear["ghi"].to_numpy()
        data.loc[positions, "dni_clear_sky"] = clear["dni"].to_numpy()
    return data
