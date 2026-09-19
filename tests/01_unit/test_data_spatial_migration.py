"""Synthetic, offline regression tests for data and spatial protocol gates."""

import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT))

from src.s01_data.fetch import fetch_data  # noqa: E402
from src.s01_data.fetch.fetch_client import fetch_json  # noqa: E402
from src.s02_data.cache_contract import (  # noqa: E402
    DataContractError, cache_metadata, validate_raw_payload,
)
from src.s02_data.clean_contract import audit_clean_frame, validate_month_paths  # noqa: E402
from src.s12_spatial.density import coverage_diagnostics, select_nested_density  # noqa: E402
from src.s12_spatial.holdout import validate_spatial_preflight  # noqa: E402
from src.s12_spatial.truth_gate import assess_himawari_truth  # noqa: E402
from src.s12_spatial.service_registry import (  # noqa: E402
    SpatialContractError, build_registry, convergence_audit, require_frozen_registry,
    service_point,
)


CFG = yaml.safe_load((ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))
DAY = dt.date(2024, 2, 1)


def payload(source="previous_runs"):
    names = (f"{v}_previous_day{lead}" for v in CFG["forecast_variables"] for lead in (1, 2, 3)) \
        if source == "previous_runs" else CFG[f"{source}_variables"]
    hours = [(dt.datetime(2024, 2, 1) + dt.timedelta(hours=i)).isoformat() for i in range(24)]
    return {"latitude": 32.125, "longitude": 119.375,
            "hourly": {"time": hours, **{name: [1.0] * 24 for name in names}}}


class DataContractMigrationTests(unittest.TestCase):
    def test_request_explicitly_selects_land(self):
        fetch_data.DATA_ROOT = Path("unused")
        site = {"id": "test", "lat": 32.0, "lon": 119.0}
        for source in ("previous_runs", "satellite", "era5"):
            _, params, _ = fetch_data.source_request(CFG, source, site, DAY, DAY)
            self.assertEqual(params["cell_selection"], "land")

    def test_rejects_legacy_15_variable_and_bad_lead_time_length(self):
        good = payload()
        validate_raw_payload(good, source="previous_runs", cfg=CFG, start=DAY, end=DAY)
        del good["hourly"]["cloud_cover_low_previous_day1"]
        with self.assertRaisesRegex(DataContractError, "cloud_cover_low"):
            validate_raw_payload(good, source="previous_runs", cfg=CFG, start=DAY, end=DAY)
        good = payload()
        good["hourly"]["cloud_cover_low_previous_day1"] = [None] * 24
        with self.assertRaisesRegex(DataContractError, "All values are null.*cloud_cover_low"):
            validate_raw_payload(good, source="previous_runs", cfg=CFG, start=DAY, end=DAY)
        good = payload()
        good["hourly"]["time"][4] = good["hourly"]["time"][3]
        with self.assertRaisesRegex(DataContractError, "timestamps"):
            validate_raw_payload(good, source="previous_runs", cfg=CFG, start=DAY, end=DAY)

    def test_old_cache_cannot_silently_skip_or_claim_current_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(payload()), encoding="utf-8")
            params = {"latitude": 32.0, "longitude": 119.0,
                      "cell_selection": "land", "start_date": DAY.isoformat(),
                      "end_date": DAY.isoformat(), "timezone": "UTC"}
            meta = cache_metadata(source="previous_runs", cfg=CFG, start=DAY, end=DAY,
                                  params=params, data_version="v2")
            class Response:
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    return False
                def read(self):
                    return json.dumps(payload()).encode("utf-8")
            with patch("urllib.request.urlopen", return_value=Response()) as open_url:
                result = fetch_json("https://invalid.example", params, path, retries=1,
                                    validator=lambda item: validate_raw_payload(
                                        item, source="previous_runs", cfg=CFG, start=DAY, end=DAY),
                                    cache_metadata=meta)
                self.assertEqual(result["latitude"], 32.125)
                open_url.assert_called_once()
            self.assertEqual(json.loads(Path(str(path) + ".meta.json").read_text()), meta)
            with patch("urllib.request.urlopen") as open_url:
                fetch_json("https://invalid.example", params, path, retries=1,
                           validator=lambda item: validate_raw_payload(
                               item, source="previous_runs", cfg=CFG, start=DAY, end=DAY),
                           cache_metadata=meta)
                open_url.assert_not_called()

    def test_missing_month_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataContractError, "Missing"):
                validate_month_paths(Path(directory), "01_gfs", "test", DAY, dt.date(2024, 3, 1))

    def test_clean_audit_checks_issue_lead_duplicates_and_service_coords(self):
        target = pd.date_range("2024-02-01", periods=24, freq="h", tz="UTC")
        frame = pd.DataFrame({"target_time_utc": list(target) * 3,
                              "lead_time": [24] * 24 + [48] * 24 + [72] * 24})
        frame["forecast_issue_time_utc"] = frame["target_time_utc"] - pd.to_timedelta(frame["lead_time"], unit="h")
        for source, lat in (("gfs", 32.1), ("himawari", 32.2), ("era5", 32.3)):
            frame[f"{source}_service_latitude"] = lat
            frame[f"{source}_service_longitude"] = 119.0
        frame["requested_latitude"] = 32.0
        frame["requested_longitude"] = 119.0
        audit = audit_clean_frame(frame, DAY, DAY, tuple(frame.columns))
        self.assertEqual(audit["service_coordinates"]["himawari"], [(32.2, 119.0)])
        bad = frame.copy()
        bad.loc[0, "forecast_issue_time_utc"] = bad.loc[0, "target_time_utc"]
        with self.assertRaisesRegex(DataContractError, "issue"):
            audit_clean_frame(bad, DAY, DAY, tuple(bad.columns))


class SpatialMigrationTests(unittest.TestCase):
    @staticmethod
    def points():
        regions = ("a", "b", "c", "d", "e")
        return tuple(service_point(31 + i * .5 + j * .02, 119 + j * .1, region)
                     for i, region in enumerate(regions) for j in range(5))

    def test_returned_not_requested_point_controls_boundary_and_region(self):
        probes = [{"requested_latitude": 32.0, "requested_longitude": 119.0,
                   "service_latitude": 40.0, "service_longitude": 120.0},
                  {"requested_latitude": 50.0, "requested_longitude": 130.0,
                   "service_latitude": 32.0, "service_longitude": 119.0}]
        registry = build_registry(probes, province_covers=lambda lon, lat: lat < 35,
                                  region_for_service=lambda lon, lat: "jiangsu")
        self.assertEqual(len(registry), 1)
        self.assertEqual(registry[0].latitude, 32.0)

    def test_707_experience_is_not_a_final_count_without_convergence(self):
        points = self.points()
        audit = convergence_audit([(0.05, points)])
        self.assertFalse(audit["converged"])
        self.assertIsNone(audit["final_service_count"])
        with self.assertRaises(SpatialContractError):
            require_frozen_registry(points, audit)
        frozen = convergence_audit([(0.05, points), (0.025, points), (0.0125, points)],
                                   boundary_sensitivity_passed=True)
        self.assertTrue(frozen["converged"])
        self.assertEqual(len(require_frozen_registry(points, frozen)), 25)

    def test_nested_deterministic_farthest_design_and_preflight(self):
        points = self.points()
        layers = select_nested_density(points, ("a", "b", "c", "d", "e"))
        self.assertEqual({n: len(v) for n, v in layers.items()}, {5: 5, 10: 10, 15: 15, 20: 20})
        for lower, upper in ((5, 10), (10, 15), (15, 20)):
            self.assertLess(set(layers[lower]), set(layers[upper]))
        self.assertEqual(layers, select_nested_density(reversed(points), ("a", "b", "c", "d", "e")))
        self.assertEqual(coverage_diagnostics(points, layers[5], coverage_radius_km=1000)["covered_service_count"], 25)
        audit = convergence_audit([(0.05, points), (0.025, points), (0.0125, points)],
                                  boundary_sensitivity_passed=True)
        preflight = validate_spatial_preflight(points, audit, layers[20], points, layers)
        self.assertEqual(preflight["status"], "preflight_only_no_official_results")

    def test_himawari_truth_gate_uses_own_returned_coordinates(self):
        point = service_point(32.0, 119.0, "a")
        frame = pd.DataFrame({
            "target_time_utc": pd.date_range("2024-02-01", periods=24, freq="h", tz="UTC"),
            "ghi_obs_sat": [1.0] * 23 + [float("nan")],
            "gfs_service_latitude": 32.0, "gfs_service_longitude": 119.0,
            "himawari_service_latitude": 32.1, "himawari_service_longitude": 119.1,
        })
        accepted = assess_himawari_truth(frame, point, start=DAY, end=DAY,
                                         minimum_valid_fraction=.95, maximum_truth_offset_km=30)
        self.assertTrue(accepted["eligible"])
        self.assertGreater(accepted["truth_offset_km"], 0)
        rejected = assess_himawari_truth(frame, point, start=DAY, end=DAY,
                                         minimum_valid_fraction=1, maximum_truth_offset_km=30)
        self.assertFalse(rejected["eligible"])


if __name__ == "__main__":
    unittest.main()
