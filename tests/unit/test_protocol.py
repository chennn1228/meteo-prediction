"""Cross-module manifest and pre-official-run gates."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import load_bundle, to_plain  # noqa: E402
from nwp.experiment.tuning import candidates  # noqa: E402
from nwp.core.validation import (  # noqa: E402
    contract_alignment, chronological_boundaries, require_official_chronology,
)


class ProtocolConsistencyTests(unittest.TestCase):
    def test_live_interfaces_match_manifest(self):
        bundle = to_plain(load_bundle())
        model_config = bundle["models"]
        alignment = contract_alignment(bundle)
        self.assertTrue(all(alignment.values()), alignment)
        self.assertEqual(candidates("ridge_mos", model_config)[0],
                         model_config["search"]["spaces"]["ridge_mos"][0])

    def test_first_outer_retained_and_mathematically_feasible(self):
        protocol = to_plain(load_bundle())["protocol"]
        evidence = chronological_boundaries(protocol)
        self.assertEqual(evidence["first_outer_prefix_days"], 121)
        self.assertEqual(evidence["minimum_days_before_fit"], 76)
        self.assertTrue(evidence["first_outer_has_nonempty_fit"])
        self.assertEqual(protocol["validation"]["scoring_days"], 14)
        require_official_chronology(protocol)

    def test_data_config_contains_no_weather_selection_protocol(self):
        bundle = to_plain(load_bundle())
        self.assertNotIn("kt_weather_bins", bundle["data"])
        self.assertEqual(bundle["features"]["diagnostics"]["role"],
                         "diagnostic_only_not_model_selection")


if __name__ == "__main__":
    unittest.main()
