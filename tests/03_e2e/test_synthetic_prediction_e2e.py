"""End-to-end CSV handoff without any training, network call or official output."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

from synthetic_protocol_fixture import synthetic_feature_and_prediction_rows  # noqa: E402
from s07_prediction.reader import read_predictions  # noqa: E402
from s07_prediction.writer import write_predictions  # noqa: E402
from s08_calibration.quantile import AdditiveQuantileCalibrator  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402


class SyntheticPredictionE2ETests(unittest.TestCase):
    def test_csv_contract_calibration_evaluation_and_official_gate(self):
        _, processor, calibration, testing = synthetic_feature_and_prediction_rows()
        self.assertEqual(processor.train_rows, 2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = write_predictions(calibration, root / "calibration.csv")
            test_path = write_predictions(testing, root / "test.csv")
            loaded_calibration = read_predictions(calibration_path)
            loaded_test = read_predictions(test_path)
            calibrator = AdditiveQuantileCalibrator().fit(
                loaded_calibration, fit_end="2025-01-02T00:00:00Z",
                early_stop_end="2025-01-05T00:00:00Z")
            with self.assertRaisesRegex(ValueError, "overlaps or follows"):
                calibrator.apply(loaded_calibration)
            calibrated = calibrator.apply(loaded_test)
            calibrated_path = write_predictions(calibrated, root / "calibrated.csv")
            report = evaluate_predictions(calibrated_path, formal=False)
            self.assertEqual(len(report.overview.probability_primary), 1)
            self.assertEqual(report.overview.probability_primary.iloc[0].n, 2)
            self.assertEqual(report.overview.probability_primary.iloc[0].result_status, "diagnostic")
            with self.assertRaisesRegex(ValueError, "formal evaluation rejects"):
                evaluate_predictions(calibrated_path, formal=True)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["calibrated.csv", "calibration.csv", "test.csv"])


if __name__ == "__main__":
    unittest.main()
