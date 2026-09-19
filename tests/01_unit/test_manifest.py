from pathlib import Path
import sys
import unittest

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src" / "s03_models"))
from model_registry import deep_model_vnum, validate_counts


class ManifestTests(unittest.TestCase):
    def test_counts_and_versions(self):
        self.assertTrue(validate_counts())
        self.assertEqual(len(deep_model_vnum()), 14)
        self.assertEqual(deep_model_vnum()["tcn"], 4)
        self.assertEqual(deep_model_vnum()["pinn"], 15)

    def test_primary_target_and_protocol(self):
        manifest = yaml.safe_load((ROOT / "project_manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["primary_target"], "ghi")
        self.assertEqual(manifest["validation_protocol"]["name"], "nested_purged_rolling_origin")
        self.assertIsNone(manifest["official_result_set"])


if __name__ == "__main__":
    unittest.main()
