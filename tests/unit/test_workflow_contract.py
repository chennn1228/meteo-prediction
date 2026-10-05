from __future__ import annotations

import sys
import unittest
import inspect
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import resolve_config
from nwp.core.validation import readiness_result
from nwp.workflow.pipeline import HANDLERS, STAGES
from nwp.workflow.stages.common import _result


class ValidatorOrchestratorTests(unittest.TestCase):
    def test_real_ordered_stages_have_callable_handlers(self):
        self.assertEqual(len(STAGES), 13)
        self.assertEqual(tuple(HANDLERS), STAGES)
        self.assertTrue(all(callable(HANDLERS[stage]) for stage in STAGES))
        self.assertEqual(resolve_config("nanjing_cpu_diagnostic").execution,
                         "development")

    def test_official_fail_closed_until_evidence_exists(self):
        report = readiness_result("official")
        self.assertEqual(report["status"], "blocked")

    def test_stage_result_requires_explicit_dependency_and_implementation(self):
        signature = inspect.signature(_result)
        for name in ("dependency_fingerprint", "implementation_fingerprint"):
            self.assertIs(signature.parameters[name].default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
