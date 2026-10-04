from __future__ import annotations

import pandas as pd
import pytest

from nwp.core.config import load_bundle
from nwp.core.schema import ContractError, assert_model_features
from nwp.models.factory import model_factory
from nwp.models.baselines import FixedBaselineModel
from nwp.models.deep import DeepModel
from nwp.models.statistical import StatisticalModel
from nwp.models.trees import TreeModel


def test_factory_creates_raw_model_from_registry_and_predicts():
    bundle = load_bundle()
    model = model_factory(
        "raw_gfs", bundle["models"], bundle["features"], bundle["protocol"])
    frame = pd.DataFrame({"ghi_fcst": [12.0, 34.0]})
    assert model.fit(frame).predict(frame).tolist() == [12.0, 34.0]


def test_feature_policy_blocks_identity_and_truth_leakage():
    with pytest.raises(ContractError):
        assert_model_features(["ghi_fcst", "station_id", "y"], load_bundle()["features"]["build"]["policy"])


def test_factory_covers_every_registered_model_family():
    bundle = load_bundle()
    expected = {
        "raw": FixedBaselineModel,
        "baseline": FixedBaselineModel,
        "statistical": StatisticalModel,
        "tree_ml": TreeModel,
        "deep": DeepModel,
        "experimental_constrained": DeepModel,
    }
    for model_id, entry in bundle["models"]["registry"].items():
        model = model_factory(
            model_id, bundle["models"], bundle["features"], bundle["protocol"])
        target = (FixedBaselineModel if model_id in {
            "climatology", "persistence", "smart_persistence",
            "optimal_convex", "raw_gfs", "bias_correction", "linear_mos"
        } else expected[entry["family"]])
        assert isinstance(model, target), model_id
