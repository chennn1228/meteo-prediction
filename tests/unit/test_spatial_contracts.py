"""Synthetic, offline regression tests for data and spatial protocol gates."""

import datetime as dt
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import load_bundle, project_root, to_plain  # noqa: E402
from nwp.core.paths import RunPaths  # noqa: E402
from nwp.core.schema import ContractError as DataContractError  # noqa: E402
from nwp.data.client import DataFetchError, fetch_json  # noqa: E402
from nwp.data.contracts import DataCatalog, expected_hourly_fields, validate_clean_frame, validate_raw_payload  # noqa: E402
from nwp.data.fetch import FetchRequest, request_parameters  # noqa: E402
from nwp.evaluation.spatial import (  # noqa: E402
    SpatialContractError, assess_himawari_truth, build_registry,
    convergence_audit, coverage_diagnostics, require_frozen_registry,
    select_nested_density, service_point, validate_spatial_preflight,
)


_BUNDLE = to_plain(load_bundle())
CFG = _BUNDLE["data"]
SELECTOR = _BUNDLE["sites"]["selectors"]["density"]
DAY = dt.date(2024, 2, 1)


def payload(source="previous_runs"):
    names = expected_hourly_fields(source, CFG)
    hours = [(dt.datetime(2024, 2, 1) + dt.timedelta(hours=i)).isoformat() for i in range(24)]
    return {"latitude": 32.125, "longitude": 119.375,
            "hourly": {"time": hours, **{name: [1.0] * 24 for name in names}}}


class DataContractMigrationTests(unittest.TestCase):
    def test_request_explicitly_selects_land(self):
        site = {"id": "test", "lat": 32.0, "lon": 119.0}
        for source in ("previous_runs", "satellite", "era5"):
            request = FetchRequest(
                source=source,
                site_id=site["id"],
                requested_latitude=site["lat"],
                requested_longitude=site["lon"],
                start=DAY,
                end=DAY,
                variables=tuple(
                    CFG["forecast"]["variables"]
                    if source == "previous_runs"
                    else CFG["truth"]["primary" if source == "satellite" else "supplementary"]["variables"]
                ),
                leads=tuple(CFG["forecast"]["leads"]) if source == "previous_runs" else (),
                model=(
                    CFG["forecast"]["model"]
                    if source == "previous_runs"
                    else CFG["truth"]["primary"]["model"] if source == "satellite" else None
                ),
            )
            params = request_parameters(request, CFG)
            self.assertEqual(params["cell_selection"], "land")

    def test_temporary_15_variable_payload_rejects_missing_and_null_total_cloud(self):
        good = payload()
        validate_raw_payload(good, source="previous_runs", data_config=CFG, start=DAY, end=DAY)
        del good["hourly"]["cloud_cover_previous_day1"]
        with self.assertRaisesRegex(DataContractError, "cloud_cover_previous_day1"):
            validate_raw_payload(good, source="previous_runs", data_config=CFG, start=DAY, end=DAY)
        good = payload()
        good["hourly"]["cloud_cover_previous_day1"] = [None] * 24
        with self.assertRaisesRegex(DataContractError, "all values are null.*cloud_cover"):
            validate_raw_payload(good, source="previous_runs", data_config=CFG, start=DAY, end=DAY)
        good = payload()
        good["hourly"]["time"][4] = good["hourly"]["time"][3]
        with self.assertRaisesRegex(DataContractError, "timestamps"):
            validate_raw_payload(good, source="previous_runs", data_config=CFG, start=DAY, end=DAY)

    def test_orphaned_raw_file_cannot_silently_claim_current_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text("{}", encoding="utf-8")
            params = {"latitude": 32.0, "longitude": 119.0,
                      "cell_selection": "land", "start_date": DAY.isoformat(),
                      "end_date": DAY.isoformat(), "timezone": "UTC"}
            opened = []
            with self.assertRaisesRegex(DataFetchError, "both exist or both be absent"):
                fetch_json("https://invalid.example", params, retries=1,
                           data_path=path,
                           receipt_path=path.with_suffix(".receipt.json"),
                           validator=lambda item: validate_raw_payload(
                               item, source="previous_runs", data_config=CFG, start=DAY, end=DAY),
                           receipt_builder=lambda *_: {},
                           opener=lambda *_args, **_kwargs: opened.append(True))
            self.assertEqual(opened, [])

    def test_missing_month_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RunPaths.create(
                project_root(), execution="development", run_id="missing_month",
                data_root=root / "data", outputs_root=root / "outputs",
            )
            catalog = DataCatalog(paths)
            self.assertIsNone(
                catalog.resolve(
                    stage="raw", config_hash="abcdef12", sites=("test",),
                    time_range="2024-02-01/2024-02-29",
                )
            )

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
        audit = validate_clean_frame(
            frame, start=DAY, end=DAY, required_columns=tuple(frame.columns),
            leads=CFG["forecast"]["leads"],
        )
        self.assertEqual(audit["service_coordinates"]["himawari"], [(32.2, 119.0)])
        bad = frame.copy()
        bad.loc[0, "forecast_issue_time_utc"] = bad.loc[0, "target_time_utc"]
        with self.assertRaisesRegex(DataContractError, "issue"):
            validate_clean_frame(
                bad, start=DAY, end=DAY, required_columns=tuple(bad.columns),
                leads=CFG["forecast"]["leads"],
            )


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
        regions = ("a", "b", "c", "d", "e")
        layers = select_nested_density(
            points, regions, selector_config=SELECTOR)
        self.assertEqual({n: len(v) for n, v in layers.items()}, {5: 5, 10: 10, 15: 15, 20: 20})
        for lower, upper in ((5, 10), (10, 15), (15, 20)):
            self.assertLess(set(layers[lower]), set(layers[upper]))
        self.assertEqual(layers, select_nested_density(
            reversed(points), regions, selector_config=SELECTOR))
        self.assertEqual(coverage_diagnostics(points, layers[5], coverage_radius_km=1000)["covered_service_count"], 25)
        audit = convergence_audit([(0.05, points), (0.025, points), (0.0125, points)],
                                  boundary_sensitivity_passed=True)
        preflight = validate_spatial_preflight(
            points, audit, layers[20], points, layers,
            selector_config=SELECTOR, regions=regions)
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
