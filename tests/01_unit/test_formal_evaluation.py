"""Evaluation must consume canonical rows and keep probability first."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s07_prediction.schema import QUANTILE_COLUMNS
from s10_evaluation.grouped import evaluate_groups
from s10_evaluation.runner import evaluate_predictions
from s10_evaluation.summaries import rank_probability_models


def predictions() -> pd.DataFrame:
    entries = []
    for i in range(4):
        lead = 24 if i < 2 else 48
        target = pd.Timestamp("2025-08-20T12:00:00Z") + pd.Timedelta(days=i)
        truth = 10.0 + i
        # Lead 24 has central intervals around truth; lead 48 is low-biased.
        centre = truth if lead == 24 else truth - 4
        quantiles = centre + np.array([-3, -2, -1, 0, 1, 2, 3], dtype=float)
        if i == 3:
            quantiles[0] = quantiles[1] + 1  # One crossing in lead 48 only.
        entries.append({
            "model_id": "ridge_mos", "implementation_level": "validated",
            "execution_level": "development", "prediction_type": "quantile",
            "location_id": "site_a" if lead == 24 else "site_b",
            "requested_coordinates": [32.1, 118.7],
            "gfs_service_coordinates": [32.0, 118.8],
            "truth_service_coordinates": [32.05, 118.75],
            "target_time_utc": target.isoformat(),
            "forecast_issue_time_utc": (target - pd.Timedelta(hours=lead)).isoformat(),
            "lead_time": lead, "outer_fold": "outer_1", "inner_fold": "inner_1",
            "seed": 0, "y": truth, "point_prediction": np.nan,
            "data_version": "data-v1", "feature_version": "features-v1",
            "protocol_revision": "p-v2", "experiment_id": "exp-1",
            "result_status": "provisional",
            **dict(zip(QUANTILE_COLUMNS, quantiles)),
        })
    for i in (0, 2):
        point = entries[i].copy()
        point.update(model_id="raw_gfs", prediction_type="point",
                     point_prediction=point["y"] - 2)
        point.update({column: np.nan for column in QUANTILE_COLUMNS})
        entries.append(point)
    return pd.DataFrame(entries)


def test_probability_first_and_raw_gfs_point_only():
    report = evaluate_predictions(predictions())
    assert report.overview.probability_primary.model_id.tolist() == ["ridge_mos"]
    assert "mean_pinball" in report.overview.probability_primary
    assert "mae" not in report.overview.probability_primary
    point = report.overview.point_secondary.set_index("model_id")
    assert set(point.index) == {"ridge_mos", "raw_gfs"}
    assert point.loc["ridge_mos", "point_source"] == "q0.50"
    assert point.loc["raw_gfs", "point_source"] == "point_prediction"
    assert point.loc["raw_gfs", "metric_role"] == "auxiliary_point"
    with pytest.raises(ValueError, match="point-only baselines"):
        rank_probability_models(report.overview.point_secondary.assign(mean_pinball=0))


def test_group_crossing_and_reliability_are_recomputed_not_copied():
    report = evaluate_predictions(predictions())
    lead = report.grouped["lead_time"].probability_primary.set_index("lead_time")
    assert lead.loc[24, "crossing_rate"] == 0
    assert lead.loc[48, "crossing_rate"] == pytest.approx(.5)
    assert report.overview.probability_primary.crossing_rate.iloc[0] == pytest.approx(.25)
    reliability = report.grouped["lead_time"].reliability
    median = reliability[reliability["quantile"] == .5].set_index("lead_time")
    assert median.loc[24, "empirical_cdf"] != median.loc[48, "empirical_cdf"]
    assert median.loc[24, "pit_status"] == "interior_only_tails_unknown"
    assert median.loc[48, "pit_status"] == "unavailable_crossing"
    location = report.grouped["location_id"].probability_primary.set_index("location_id")
    assert location.loc["site_a", "crossing_rate"] == 0
    assert location.loc["site_b", "crossing_rate"] == pytest.approx(.5)


def test_formal_gate_rejects_prototype_and_nonofficial_without_upgrading():
    frame = predictions()
    with pytest.raises(ValueError, match="formal evaluation rejects"):
        evaluate_predictions(frame, formal=True)
    frame["implementation_level"] = "prototype"
    with pytest.raises(ValueError, match="formal evaluation rejects"):
        evaluate_predictions(frame, formal=True)
    frame["implementation_level"] = "validated"
    frame["execution_level"] = "official"
    frame["result_status"] = "official"
    report = evaluate_predictions(frame, formal=True)
    assert set(report.overview.probability_primary.execution_level) == {"official"}


def test_group_api_validates_contract_and_provenance_is_not_merged():
    frame = predictions()
    frame.loc[0, "experiment_id"] = "exp-2"
    output = evaluate_groups(frame).probability_primary
    assert set(output.experiment_id) == {"exp-1", "exp-2"}
    with pytest.raises(ValueError, match="grouping dimensions"):
        evaluate_groups(frame, ("nonexistent",))
    frame = predictions().drop(columns=["protocol_revision"])
    with pytest.raises(ValueError, match="prediction contract missing"):
        evaluate_predictions(frame)
