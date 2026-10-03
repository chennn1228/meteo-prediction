"""End-to-end CSV handoff without any training, network call or official output."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from tests.integration.synthetic_protocol_fixture import (  # noqa: E402
    synthetic_feature_and_prediction_rows)
from nwp.core.config import load_bundle, to_plain  # noqa: E402
from nwp.experiment.calibration import AdditiveQuantileCalibrator  # noqa: E402
from nwp.experiment.prediction import read_predictions, write_predictions  # noqa: E402
from nwp.evaluation.grouped import evaluate_predictions  # noqa: E402


BUNDLE = to_plain(load_bundle())
CONFIG = BUNDLE["protocol"]


class SyntheticPredictionE2ETests(unittest.TestCase):
    def test_csv_contract_calibration_evaluation_and_official_gate(self):
        _, processor, calibration, testing = synthetic_feature_and_prediction_rows()
        self.assertEqual(processor.train_rows, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = write_predictions(
                calibration, root / "calibration.csv", CONFIG)
            test_path = write_predictions(testing, root / "test.csv", CONFIG)
            loaded_calibration = read_predictions(calibration_path, CONFIG)
            loaded_test = read_predictions(test_path, CONFIG)
            calibrator = AdditiveQuantileCalibrator(CONFIG).fit(
                loaded_calibration, fit_end="2025-01-02T00:00:00Z",
                early_stop_end="2025-01-05T00:00:00Z")
            with self.assertRaisesRegex(ValueError, "overlaps or follows"):
                calibrator.apply(loaded_calibration)
            calibrated = calibrator.apply(loaded_test)
            calibrated_path = write_predictions(
                calibrated, root / "calibrated.csv", CONFIG)
            report = evaluate_predictions(
                calibrated_path, CONFIG, BUNDLE["models"], formal=False)
            self.assertEqual(len(report.overview.probability_primary), 1)
            self.assertEqual(report.overview.probability_primary.iloc[0].n, 2)
            self.assertEqual(report.overview.probability_primary.iloc[0].result_status, "diagnostic")
            with self.assertRaisesRegex(ValueError, "formal evaluation rejects"):
                evaluate_predictions(
                    calibrated_path, CONFIG, BUNDLE["models"], formal=True)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["calibrated.csv", "calibration.csv", "test.csv"])


if __name__ == "__main__":
    unittest.main()
