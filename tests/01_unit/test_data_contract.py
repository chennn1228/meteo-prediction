from pathlib import Path
import sys
import unittest

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src" / "s01_data" / "clean"))
from clean_data import FCST_RENAME


class DataContractTests(unittest.TestCase):
    def test_forecast_contract_has_cloud_levels_and_correct_count(self):
        cfg = yaml.safe_load((ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(cfg["forecast_variables"]), 18)
        for field in ("cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
            self.assertIn(field, cfg["forecast_variables"])
            self.assertIn(field, FCST_RENAME)


if __name__ == "__main__":
    unittest.main()
