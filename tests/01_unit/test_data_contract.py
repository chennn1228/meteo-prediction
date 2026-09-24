from pathlib import Path
import sys
import unittest

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src" / "s01_data" / "clean"))
from clean_data import FCST_RENAME


class DataContractTests(unittest.TestCase):
    def test_archive_forecast_contract_keeps_total_cloud_only(self):
        cfg = yaml.safe_load((ROOT / "config" / "02_variables.yaml").read_text(encoding="utf-8"))
        self.assertEqual(len(cfg["forecast_variables"]), 15)
        self.assertIn("cloud_cover", cfg["forecast_variables"])
        for field in ("cloud_cover_low", "cloud_cover_mid", "cloud_cover_high"):
            self.assertNotIn(field, cfg["forecast_variables"])


if __name__ == "__main__":
    unittest.main()
