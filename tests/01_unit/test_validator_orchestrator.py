from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError
from s14_pipeline.orchestrator import STAGES, run_stage
from s15_validation.validate_project import result


class ValidatorOrchestratorTests(unittest.TestCase):
    def test_ordered_stages_and_safe_default(self):
        self.assertEqual([stage.number for stage in STAGES], list(range(1, 14)))
        self.assertEqual(run_stage("validate")["status"], "pass")
        with self.assertRaises(ProtocolError):
            run_stage("train_cpu")

    def test_official_fail_closed_until_evidence_exists(self):
        report = result("official")
        self.assertEqual(report["status"], "blocked")
        with self.assertRaises(ProtocolError):
            run_stage("train_cpu", implementation_level="validated", execution_level="official")


if __name__ == "__main__":
    unittest.main()
