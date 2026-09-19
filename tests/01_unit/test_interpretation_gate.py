from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError
from s11_interpretation.s03_shap.explain import sample_aligned


class InterpretationGateTests(unittest.TestCase):
    def test_one_sample_index_for_features_and_region(self):
        features = pd.DataFrame({"ghi_fcst": list(range(30))})
        metadata = pd.DataFrame({"target_time_utc": ["2024-06-01T00:00:00Z"] * 30,
                                 "region": [f"region_{i}" for i in range(30)]})
        sampled_x, sampled_meta = sample_aligned(features, metadata, n=10, seed=0)
        self.assertEqual(sampled_x.index.tolist(), sampled_meta.index.tolist())
        self.assertEqual([f"region_{i}" for i in sampled_x["ghi_fcst"]],
                         sampled_meta["region"].tolist())

    def test_final_test_cannot_drive_feature_analysis(self):
        features = pd.DataFrame({"ghi_fcst": [1]})
        metadata = pd.DataFrame({"target_time_utc": ["2025-09-01T00:00:00Z"]})
        with self.assertRaises(ProtocolError):
            sample_aligned(features, metadata, n=1)


if __name__ == "__main__":
    unittest.main()
