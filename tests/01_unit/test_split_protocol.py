from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src" / "s02_experiment"))
import split_protocol as sp


def sample_frame():
    return pd.DataFrame({"target_time_utc": pd.date_range(
        "2024-01-01", "2026-10-01", freq="6h", tz="UTC"), "value": 1})


class SplitProtocolTests(unittest.TestCase):
    def test_final_test_is_exactly_twelve_months_and_bounded(self):
        development, test = sp.split_test(sample_frame())
        self.assertGreaterEqual(development.target_time_utc.min(), sp.DEV_START)
        self.assertLess(development.target_time_utc.max(), sp.TEST_START)
        self.assertEqual(test.target_time_utc.min(), sp.TEST_START)
        self.assertLess(test.target_time_utc.max(), sp.TEST_END)

    def test_final_order_is_strictly_causal(self):
        training, calibration, test = sp.default_split(sample_frame())
        fit, early_stop = sp.split_early_stop(training)
        self.assertLess(fit.target_time_utc.max() + sp.PURGE, early_stop.target_time_utc.min())
        self.assertLess(early_stop.target_time_utc.max() + sp.PURGE, calibration.target_time_utc.min())
        self.assertLess(calibration.target_time_utc.max(), test.target_time_utc.min())

    def test_outer_and_inner_folds_are_causal_and_purged(self):
        development, _ = sp.split_test(sample_frame())
        for train, valid, _ in sp.nested_rolling_folds(development):
            self.assertLess(train.target_time_utc.max() + sp.PURGE, valid.target_time_utc.min())
            for inner_train, inner_valid in sp.inner_rolling_folds(train):
                self.assertLess(inner_train.target_time_utc.max() + sp.PURGE,
                                inner_valid.target_time_utc.min())


if __name__ == "__main__":
    unittest.main()
