"""Protocol, causal features and fold-preprocessing regression checks."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError, load_manifest
from s01_core.provenance import validate_result_status
from s01_core.schemas import assert_model_features
from s03_features.engineering import (
    assert_forecast_issue_semantics, build_forecast_features,
    formal_daylight_mask,
)
from s03_features.preprocessing import FoldPreprocessor


class CoreFeatureMigrationTests(unittest.TestCase):
    def test_manifest_and_result_gate(self):
        config = load_manifest()
        self.assertIsNone(config["official_result_set"])
        self.assertEqual(config["selection_metric"], "mean_pinball")
        with self.assertRaises(ProtocolError):
            validate_result_status("prototype", "official", "official")
        validate_result_status("validated", "official", "official")

    def test_identity_and_truth_never_features(self):
        for name in ("station_id", "location_id", "station_nanjing", "ghi_obs_sat"):
            with self.assertRaises(ProtocolError):
                assert_model_features(["ghi_fcst", name])

    def test_issue_time_and_circular_and_sunrise_stability(self):
        targets = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z",
                                  "2025-01-01T02:00:00Z"])
        frame = pd.DataFrame({
            "location_id": ["service-a"] * 3,
            "target_time_utc": targets,
            "forecast_issue_time_utc": targets - pd.Timedelta(hours=24),
            "lead_time": [24] * 3,
            "ghi_fcst": [10.0, 100.0, 200.0],
            "dhi_fcst": [5.0, 50.0, 100.0],
            "dni_fcst": [4.0, 90.0, 180.0],
            "ghi_clear_sky": [1.0, 200.0, 300.0],
            "dni_clear_sky": [1.0, 180.0, 270.0],
            "wind_dir_fcst": [359.0, 0.0, 1.0],
            "solar_azimuth": [359.0, 0.0, 1.0],
            "cloud_cover_fcst": [20.0, 30.0, 40.0],
            "solar_elevation": [-1.0, 0.0, 5.0],
            "ghi_obs_sat": [999.0] * 3,
        })
        assert_forecast_issue_semantics(frame)
        result = build_forecast_features(frame)
        self.assertTrue(np.isnan(result.loc[0, "kt_raw"]))
        self.assertAlmostEqual(result.loc[1, "kt_raw"], 0.5)
        self.assertAlmostEqual(result.loc[1, "kt_model"], 0.5)
        frame.loc[1, "ghi_fcst"] = 400.0
        unbounded = build_forecast_features(frame)
        self.assertAlmostEqual(unbounded.loc[1, "kt_raw"], 2.0)
        self.assertAlmostEqual(unbounded.loc[1, "kt_model"], 2.0)
        self.assertAlmostEqual(result.loc[2, "ghi_fcst_lag1"], 100.0)
        self.assertEqual(formal_daylight_mask(result).tolist(), [False, False, True])
        self.assertAlmostEqual(result.loc[0, "wind_dir_cos"], result.loc[2, "wind_dir_cos"], places=3)
        frame.loc[2, "forecast_issue_time_utc"] = targets[2]
        with self.assertRaises(ProtocolError):
            assert_forecast_issue_semantics(frame)

    def test_fold_preprocessor_never_fits_on_later_rows(self):
        train = pd.DataFrame({
            "forecast_issue_time_utc": pd.to_datetime(["2024-03-01T00:00:00Z", "2024-03-02T00:00:00Z"]),
            "ghi_fcst": [1.0, np.nan],
        })
        later = pd.DataFrame({
            "forecast_issue_time_utc": pd.to_datetime(["2024-04-01T00:00:00Z"]),
            "ghi_fcst": [10000.0],
        })
        prep = FoldPreprocessor(("ghi_fcst",)).fit(train, fit_end_utc="2024-03-02")
        self.assertEqual(prep.train_rows, 2)
        self.assertAlmostEqual(prep.medians["ghi_fcst"], 1.0)
        self.assertGreater(prep.transform(later).iloc[0, 0], 1000)
        with self.assertRaises(ProtocolError):
            FoldPreprocessor(("ghi_fcst",)).fit(pd.concat([train, later]), fit_end_utc="2024-03-02")


if __name__ == "__main__":
    unittest.main()
