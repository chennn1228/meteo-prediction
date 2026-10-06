from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))
from nwp.core.config import load_bundle, to_plain


class ManifestTests(unittest.TestCase):
    def test_canonical_model_ids_have_no_artificial_versions(self):
        registry = to_plain(load_bundle(str(ROOT)))["models"]["registry"]
        expected = {
            "raw_gfs", "climatology", "persistence", "smart_persistence",
            "optimal_convex", "bias_correction", "linear_mos", "ridge_mos",
            "lgbm", "xgboost", "mlp", "cnn", "tcn", "lstm", "transformer",
            "autoformer", "informer", "fedformer", "itransformer", "patchtst",
            "dlinear", "timesnet", "tsmixer", "pinn",
        }
        self.assertEqual(set(registry), expected)
        self.assertTrue(all("internal_version" not in entry
                            for entry in registry.values()))

    def test_primary_target_and_protocol(self):
        bundle = load_bundle(str(ROOT))
        self.assertEqual(bundle["manifest"]["protocol_version"], "2.1.0")
        self.assertEqual(bundle["manifest"]["protocol_status"], "provisional")
        self.assertEqual(bundle["protocol"]["target"], "ghi")
        self.assertEqual(bundle["protocol"]["validation"]["name"], "nested_purged_rolling_origin")
        self.assertIsNone(bundle["manifest"]["official_result_set"])

    def test_active_machine_ids_are_canonical_snake_case(self):
        bundle = to_plain(load_bundle(str(ROOT)))
        identifiers = [
            bundle["manifest"]["project_id"],
            bundle["data"]["version"],
            bundle["features"]["build"]["version"],
            *bundle["models"]["registry"],
            *bundle["models"]["groups"],
            *bundle["models"]["search"]["spaces"],
            *bundle["sites"]["registry"],
            *bundle["sites"]["sets"],
            *bundle["sites"]["selectors"],
            *bundle["experiments"]["profiles"],
        ]
        pattern = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
        self.assertTrue(all(pattern.fullmatch(value) for value in identifiers))
        self.assertTrue(all(not re.search(r"(?:^|_)v\d+(?:_|$)", value)
                            for value in identifiers))

    def test_import_manifests_use_original_tag_paths_and_exact_bytes(self):
        verified = 0
        for name in ("raw_data_audit.json", "nanjing_diagnostic.json"):
            manifest = json.loads((
                ROOT / "migration" / "import_manifests" / name
            ).read_text(encoding="utf-8"))
            for row in manifest["files"]:
                source = row["source_path"]
                self.assertFalse(source.startswith("migration/recovery_staging/"))
                if not source.startswith(("figs/", "reports/")):
                    continue
                data = subprocess.check_output(
                    ["git", "show", f"{manifest['source_tag']}:{source}"], cwd=ROOT)
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
                verified += 1
        self.assertEqual(verified, 94)


if __name__ == "__main__":
    unittest.main()
