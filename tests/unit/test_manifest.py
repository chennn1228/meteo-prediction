from pathlib import Path
import sys
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))
from nwp.core.config import load_bundle, to_plain


class ManifestTests(unittest.TestCase):
    def test_counts_and_versions(self):
        registry = to_plain(load_bundle(str(ROOT)))["models"]["registry"]
        versions = {model_id: int(entry["internal_version"].removeprefix("v"))
                    for model_id, entry in registry.items()
                    if "internal_version" in entry}
        self.assertEqual(len(versions), 14)
        self.assertEqual(versions["tcn"], 4)
        self.assertEqual(versions["pinn"], 15)

    def test_primary_target_and_protocol(self):
        bundle = load_bundle(str(ROOT))
        self.assertEqual(bundle["protocol"]["target"], "ghi")
        self.assertEqual(bundle["protocol"]["validation"]["name"], "nested_purged_rolling_origin")
        self.assertIsNone(bundle["manifest"]["official_result_set"])


if __name__ == "__main__":
    unittest.main()
