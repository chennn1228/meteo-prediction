"""Cross-module contracts on tiny synthetic data, with no model training."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from synthetic_protocol_fixture import (  # noqa: E402
    raw_point_companion, synthetic_feature_and_prediction_rows,
)
from s07_prediction.schema import QUANTILE_COLUMNS, validate_predictions  # noqa: E402
from s08_calibration.quantile import AdditiveQuantileCalibrator  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402


class SyntheticProtocolIntegrationTests(unittest.TestCase):
    def test_features_fold_preprocessing_calibration_and_evaluation(self):
        features, processor, calibration_rows, test_rows = synthetic_feature_and_prediction_rows()
        transformed = processor.transform(features.iloc[2:])
        self.assertEqual(transformed.shape, (6, 3))
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertEqual(processor.train_rows, 2)
        self.assertEqual(features.location_id.nunique(), 1)

        before = validate_predictions(test_rows)
        calibrator = AdditiveQuantileCalibrator().fit(
            calibration_rows, fit_end="2025-01-02T00:00:00Z",
            early_stop_end="2025-01-05T00:00:00Z")
        after = calibrator.apply(test_rows)
        self.assertEqual(calibrator.calibration_end_.isoformat(), "2025-01-14T00:00:00+00:00")
        self.assertTrue((after["q0.50"] > before["q0.50"]).all())
        self.assertTrue((np.diff(after.loc[:, QUANTILE_COLUMNS].to_numpy(), axis=1) >= 0).all())

        evaluation = evaluate_predictions(pd.concat([after, raw_point_companion(test_rows)],
                                                    ignore_index=True))
        probability = evaluation.overview.probability_primary
        auxiliary = evaluation.overview.point_secondary
        self.assertEqual(probability.model_id.tolist(), ["synthetic_quantile"])
        self.assertEqual(set(auxiliary.model_id), {"synthetic_quantile", "raw_gfs"})
        self.assertIn("mean_pinball", probability.columns)
        self.assertEqual(set(auxiliary.metric_role), {"auxiliary_point"})
        self.assertIn("lead_time", evaluation.grouped)

    def test_prototype_smoke_cannot_be_upgraded_to_formal(self):
        _, _, _, test_rows = synthetic_feature_and_prediction_rows()
        diagnostic = evaluate_predictions(test_rows, formal=False)
        self.assertEqual(diagnostic.overview.probability_primary.result_status.tolist(),
                         ["diagnostic"])
        with self.assertRaisesRegex(ValueError, "formal evaluation rejects"):
            evaluate_predictions(test_rows, formal=True)
        forged = test_rows.copy()
        forged["result_status"] = "official"
        forged["execution_level"] = "official"
        with self.assertRaisesRegex(ValueError, "validated implementation"):
            validate_predictions(forged)


if __name__ == "__main__":
    unittest.main()
