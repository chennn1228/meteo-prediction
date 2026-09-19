from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src" / "s04_evaluation" / "calibration"))
from calibrate_quantiles import TAUS, calibrate


def frame(y, offset):
    data = {"y": np.asarray(y, dtype=float)}
    for index, tau in enumerate(TAUS):
        data[f"q{tau:g}"] = np.asarray(y, dtype=float) - offset + index
    return pd.DataFrame(data)


class CalibrationTests(unittest.TestCase):
    def test_additive_residual_uses_y_minus_qhat_sign(self):
        validation = frame([10, 20, 30, 40], offset=2.0)
        test = frame([100, 110], offset=0.0)
        calibrated, deltas = calibrate(validation, test)
        self.assertEqual(deltas["q0.5"], 2.0 - TAUS.index(0.5))
        self.assertTrue(np.allclose(calibrated["q0.5"],
                                    test["q0.5"] + deltas["q0.5"]))

    def test_calibration_output_is_monotonic(self):
        validation = frame([10, 20, 30], offset=1.0)
        test = frame([100, 110], offset=0.0)
        calibrated, _ = calibrate(validation, test)
        matrix = calibrated[[f"q{tau:g}" for tau in TAUS]].to_numpy()
        self.assertTrue(np.all(np.diff(matrix, axis=1) >= 0))


if __name__ == "__main__":
    unittest.main()
