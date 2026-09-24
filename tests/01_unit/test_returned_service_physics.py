"""Real pvlib calculation must ignore the requested probe coordinate."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s03_features.physics import add_returned_service_physics  # noqa: E402
from s03_features.pipeline import prepare_formal_features  # noqa: E402
from s03_features.physics import preceding_hour_solar_geometry  # noqa: E402


def _clean_sample() -> pd.DataFrame:
    target = pd.date_range("2024-06-01T04:00:00Z", periods=3, freq="h")
    frame = pd.DataFrame({
        "target_time_utc": target,
        "forecast_issue_time_utc": target - pd.Timedelta(hours=24),
        "lead_time": [24] * 3,
        "requested_latitude": [31.9] * 3,
        "requested_longitude": [118.1] * 3,
        "gfs_service_latitude": [32.0] * 3,
        "gfs_service_longitude": [118.0] * 3,
        "gfs_service_elevation": [20.0] * 3,
    })
    forecast = {
        "ghi_fcst": 700., "dhi_fcst": 100., "dni_fcst": 800., "gti_fcst": 650.,
        "cloud_cover_fcst": 20.,
        "temp_fcst": 25., "rh_fcst": 60., "dewpoint_fcst": 16.,
        "wind_speed_fcst": 3., "wind_dir_fcst": 350., "pressure_fcst": 1010.,
        "precip_fcst": 0., "sunshine_fcst": 3600., "terrestrial_fcst": 300.,
    }
    for column, value in forecast.items():
        frame[column] = value
    return frame


class ReturnedServicePhysicsTests(unittest.TestCase):
    def test_preceding_hour_geometry_uses_midpoint_not_interval_end(self):
        import pvlib
        times = pd.DatetimeIndex(["2026-04-12T22:00:00Z"])
        site = pvlib.location.Location(31.923203, 118.59375, altitude=16, tz="UTC")
        result = preceding_hour_solar_geometry(times, site)
        endpoint = site.get_solarposition(times)["apparent_elevation"].iloc[0]
        midpoint = site.get_solarposition(times - pd.Timedelta(minutes=30))["apparent_elevation"].iloc[0]
        self.assertAlmostEqual(result["solar_elevation"][0], midpoint)
        self.assertNotAlmostEqual(result["solar_elevation"][0], endpoint)
        self.assertGreaterEqual(result["ghi_clear_sky"][0], 0)
    def test_request_change_does_not_change_formal_physics_or_features(self):
        first = _clean_sample()
        second = first.copy()
        second["requested_latitude"] = 33.5
        second["requested_longitude"] = 119.5
        try:
            left, names = prepare_formal_features(first)
            right, same_names = prepare_formal_features(second)
        except ModuleNotFoundError as exc:
            if exc.name == "pvlib":
                self.skipTest("pvlib unavailable in this Python environment")
            raise
        self.assertEqual(names, same_names)
        pd.testing.assert_frame_equal(left.loc[:, names], right.loc[:, names])
        self.assertEqual(left.location_id.tolist(), right.location_id.tolist())
        self.assertNotIn("requested_latitude", names)
        self.assertNotIn("requested_longitude", names)

    def test_no_requested_coordinate_fallback(self):
        frame = _clean_sample().drop(columns=["gfs_service_latitude"])
        with self.assertRaisesRegex(ValueError, "returned GFS"):
            add_returned_service_physics(frame)


if __name__ == "__main__":
    unittest.main()
