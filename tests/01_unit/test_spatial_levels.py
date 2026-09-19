"""Offline three-level spatial preflight and density marginal tests."""

import datetime as dt
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT))

from src.s12_spatial.density import select_nested_density  # noqa: E402
from src.s12_spatial.levels import (  # noqa: E402
    plan_density_marginals, plan_level1, plan_level2, plan_level3,
)
from src.s12_spatial.service_registry import (  # noqa: E402
    SpatialContractError, convergence_audit, service_point,
)
from src.s12_spatial.spatial_context import SpatialContext  # noqa: E402
from src.s12_spatial.truth_gate import assess_himawari_truth  # noqa: E402


REGIONS = ("a", "b", "c", "d", "e")
DAY = dt.date(2024, 2, 1)


def make_context():
    points = tuple(service_point(31 + i * .5 + j * .02, 119 + j * .1, region)
                   for i, region in enumerate(REGIONS) for j in range(5))
    audit = convergence_audit([(0.05, points), (0.025, points), (0.0125, points)])
    hours = pd.date_range("2024-02-01", periods=24, freq="h", tz="UTC")
    evidence = {}
    for point in points:
        frame = pd.DataFrame({
            "target_time_utc": hours, "ghi_obs_sat": [1.0] * 24,
            "gfs_service_latitude": point.latitude,
            "gfs_service_longitude": point.longitude,
            "himawari_service_latitude": point.latitude,
            "himawari_service_longitude": point.longitude,
        })
        evidence[point.service_id] = assess_himawari_truth(
            frame, point, start=DAY, end=DAY,
            minimum_valid_fraction=.95, maximum_truth_offset_km=10)
    return points, audit, evidence, SpatialContext.validated(
        points, audit, evidence, truth_period_start=DAY, truth_period_end=DAY)


class SpatialLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.audit, cls.evidence, cls.context = make_context()
        cls.layers = select_nested_density(cls.points, REGIONS)
        cls.layer_ids = {count: [point.service_id for point in points]
                         for count, points in cls.layers.items()}

    def test_frozen_registry_requires_exact_final_identities(self):
        fake = dict(self.audit)
        fake["final_service_ids"] = ["outside"] * 25
        with self.assertRaisesRegex(SpatialContractError, "identities"):
            SpatialContext.validated(self.points, fake, self.evidence,
                                     truth_period_start=DAY, truth_period_end=DAY)

    def test_level1_development_stratified_disjoint(self):
        plan = plan_level1(self.context, self.layer_ids[20], regions=REGIONS)
        self.assertEqual(plan["period"], "development_only")
        self.assertEqual(len(plan["training_service_ids"]), 15)
        self.assertEqual(len(plan["test_service_ids"]), 5)
        self.assertEqual(set(plan["holdout_count_by_region"].values()), {1})
        self.assertFalse(set(plan["training_service_ids"]) & set(plan["test_service_ids"]))
        self.assertEqual(plan["score_status"], "not_run")

    def test_level2_uses_truth_gated_unseen_returned_service_points(self):
        plan = plan_level2(self.context, self.layer_ids[20])
        self.assertEqual(len(plan["test_service_ids"]), 5)
        self.assertTrue(set(plan["test_service_ids"]).issubset(
            {point.service_id for point in self.points}))
        self.assertTrue(all(distance > 0 for distance in
                            plan["test_distance_to_nearest_training_km"].values()))
        with self.assertRaisesRegex(SpatialContractError, "truth-eligible"):
            plan_level2(self.context, ["synthetic-0.1-degree-grid"])

    def test_level3_four_region_training_and_fifth_region_truth_only(self):
        plan = plan_level3(self.context, self.layer_ids[20], held_out_region="e", regions=REGIONS)
        self.assertEqual(len(plan["training_service_ids"]), 16)
        self.assertEqual(len(plan["test_service_ids"]), 5)
        self.assertEqual({self.context.by_id[key].region for key in plan["training_service_ids"]},
                         set(REGIONS) - {"e"})
        self.assertEqual({self.context.by_id[key].region for key in plan["test_service_ids"]}, {"e"})

    def test_density_nested_quotas_and_costs_without_claimed_gain(self):
        plan = plan_density_marginals(self.context, self.layer_ids,
                                      regions=REGIONS, coverage_radius_km=100)
        self.assertEqual(len(plan["marginals"]), 3)
        self.assertEqual([item["additional_training_points"] for item in plan["marginals"]], [5, 5, 5])
        self.assertEqual([item["additional_valid_truth_hours"] for item in plan["marginals"]], [120, 120, 120])
        self.assertTrue(all(item["marginal_probability_gain"] is None for item in plan["marginals"]))
        self.assertEqual(plan["coverage_diagnostics"][20]["selected_count"], 20)
        nonnested = dict(self.layer_ids)
        nonnested[10] = nonnested[10][:-1] + [self.layer_ids[20][-1]]
        with self.assertRaisesRegex(SpatialContractError, "nesting"):
            plan_density_marginals(self.context, nonnested, regions=REGIONS, coverage_radius_km=100)

    def test_truth_evidence_is_required_not_boolean_flag(self):
        incomplete = dict(self.evidence)
        incomplete[self.points[0].service_id] = {"eligible": True}
        with self.assertRaisesRegex(SpatialContractError, "schema"):
            SpatialContext.validated(self.points, self.audit, incomplete,
                                     truth_period_start=DAY, truth_period_end=DAY)
        missing = dict(self.evidence)
        del missing[self.points[0].service_id]
        context = SpatialContext.validated(self.points, self.audit, missing,
                                           truth_period_start=DAY, truth_period_end=DAY)
        with self.assertRaisesRegex(SpatialContractError, "truth-eligible"):
            plan_level2(context, [self.points[0].service_id])


if __name__ == "__main__":
    unittest.main()
