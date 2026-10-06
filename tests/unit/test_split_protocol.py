from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "project_manifest.yaml").exists()
)
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import load_bundle, resolve_config, to_plain  # noqa: E402
from nwp.splits.rolling import (  # noqa: E402
    assert_gap,
    development_and_test,
    final_fit_early_stop,
    final_training_calibration_test,
    inner_folds,
    outer_folds,
    purge_days,
)


PROTOCOL = to_plain(load_bundle())["protocol"]
VALIDATION = PROTOCOL["validation"]
GAP_DAYS = purge_days(VALIDATION)


def sample_frame():
    return pd.DataFrame(
        {
            "target_time_utc": pd.date_range(
                "2024-01-01", "2026-10-01", freq="6h", tz="UTC"
            ),
            "value": 1,
        }
    )


class SplitProtocolTests(unittest.TestCase):
    def test_final_test_is_exactly_twelve_months_and_bounded(self):
        development, testing = development_and_test(
            sample_frame(), PROTOCOL["development_period"], PROTOCOL["test_period"]
        )
        self.assertGreaterEqual(
            development.target_time_utc.min(),
            pd.Timestamp(PROTOCOL["development_period"]["start"]),
        )
        self.assertLess(
            development.target_time_utc.max(),
            pd.Timestamp(PROTOCOL["test_period"]["start"]),
        )
        self.assertEqual(
            testing.target_time_utc.min(),
            pd.Timestamp(PROTOCOL["test_period"]["start"]),
        )
        self.assertLessEqual(
            testing.target_time_utc.max(),
            pd.Timestamp(PROTOCOL["test_period"]["end"]),
        )

    def test_final_order_is_strictly_causal(self):
        training, calibration, testing = final_training_calibration_test(
            sample_frame(), PROTOCOL
        )
        fit, early_stop = final_fit_early_stop(training, PROTOCOL)
        assert_gap(fit, early_stop, GAP_DAYS)
        assert_gap(early_stop, calibration, GAP_DAYS)
        self.assertLess(
            calibration.target_time_utc.max(), testing.target_time_utc.min()
        )

    def test_outer_and_inner_folds_are_causal_and_purged(self):
        development, _ = development_and_test(
            sample_frame(), PROTOCOL["development_period"], PROTOCOL["test_period"]
        )
        for outer in outer_folds(
            development, VALIDATION, PROTOCOL["development_period"]
        ):
            assert_gap(outer.fit, outer.score, GAP_DAYS)
            for inner in inner_folds(
                outer, VALIDATION, PROTOCOL["development_period"]
            ):
                assert_gap(inner.fit, inner.early_stop, GAP_DAYS)
                assert_gap(inner.early_stop, inner.score, GAP_DAYS)

    def test_gap_sensitivity_changes_rolling_only_not_final_blocks(self):
        frame = sample_frame()
        final_boundaries = []
        rolling_boundaries = []
        for gap in (7, 10, 14):
            protocol = resolve_config(
                "nanjing_cpu_diagnostic", gap_days=gap).protocol
            training, calibration, testing = final_training_calibration_test(
                frame, protocol)
            fit, early = final_fit_early_stop(training, protocol)
            self.assertTrue(all(not block.empty for block in (
                fit, early, calibration, testing)))
            final_boundaries.append((
                fit.target_time_utc.max(), early.target_time_utc.min(),
                calibration.target_time_utc.min(), testing.target_time_utc.min()))
            outer = outer_folds(
                frame, protocol["validation"], protocol["development_period"],
                gap_days=gap)[0]
            rolling_boundaries.append(outer.fit.target_time_utc.max())
        self.assertEqual(len(set(final_boundaries)), 1)
        self.assertEqual(len(set(rolling_boundaries)), 3)


if __name__ == "__main__":
    unittest.main()
