"""Cross-module manifest and pre-official-run gates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import ProtocolError, load_data_config, load_manifest  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402
from s15_validation.validate_protocol import (  # noqa: E402
    check_contract_alignment, chronological_boundaries, require_official_chronology,
)


class ProtocolConsistencyTests(unittest.TestCase):
    def test_live_interfaces_match_manifest(self):
        alignment = check_contract_alignment(load_manifest(), load_data_config("02_variables.yaml"))
        self.assertTrue(all(alignment.values()), alignment)
        self.assertEqual(candidates("ridge_mos")[0], load_manifest()["tuning_search_spaces"]["ridge_mos"][0])

    def test_first_outer_retained_and_mathematically_feasible(self):
        evidence = chronological_boundaries(load_manifest())
        self.assertEqual(evidence["first_outer_prefix_days"], 121)
        self.assertEqual(evidence["minimum_days_before_fit"], 76)
        self.assertTrue(evidence["first_outer_has_nonempty_fit"])
        self.assertEqual(load_manifest()["validation_protocol"]["inner_scoring_days"], 14)
        require_official_chronology(load_manifest())

    def test_data_config_contains_no_weather_selection_protocol(self):
        self.assertNotIn("kt_weather_bins", load_data_config("02_variables.yaml"))
        self.assertEqual(load_manifest()["weather_diagnostics"]["role"],
                         "diagnostic_only_not_model_selection")


if __name__ == "__main__":
    unittest.main()
