"""Protocol-to-code checks for the formal prediction, metric and calibration API."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from nwp.core.config import load_bundle, to_plain
from nwp.experiment.calibration import (
    AdditiveQuantileCalibrator, block_bootstrap_indices, bootstrap_metric)
from nwp.experiment.prediction import (
    quantile_columns, read_predictions, validate_predictions, write_predictions)
from nwp.evaluation.grouped import grouped_metrics
from nwp.evaluation.metrics import (
    point_metrics, probability_metrics, truncated_quantile_crps,
    pit_central, quantile_reliability)


CONFIG = to_plain(load_bundle())["protocol"]
QUANTILE_COLUMNS = quantile_columns(CONFIG)
LEVELS = tuple(CONFIG["probability"]["quantiles"])


def rows(*, times=("2025-08-20T12:00:00Z", "2025-08-21T12:00:00Z"),
         model="ridge_mos", kind="quantile", y=(10.0, 20.0)) -> pd.DataFrame:
    n = len(times)
    forecast = pd.to_datetime(times, utc=True) - pd.Timedelta(days=3)
    frame = pd.DataFrame({
        "model_id": [model] * n, "implementation_level": ["validated"] * n,
        "execution_level": ["development"] * n, "prediction_type": [kind] * n,
        "location_id": ["site_a"] * n,
        "requested_coordinates": [[32.1, 118.7] for _ in times],
        "gfs_service_coordinates": [[32.0, 118.8] for _ in times],
        "truth_service_coordinates": [[32.05, 118.75] for _ in times],
        "target_time_utc": list(times), "forecast_issue_time_utc": forecast,
        "lead_time": [72] * n, "outer_fold": ["outer_1"] * n,
        "inner_fold": ["inner_1"] * n, "seed": [0] * n,
        "y": list(y), "point_prediction": [np.nan] * n,
        "data_version": ["synthetic_data"] * n, "feature_version": ["synthetic_features"] * n,
        "protocol_version": ["synthetic_protocol"] * n, "experiment_id": ["experiment_1"] * n,
        "result_status": ["provisional"] * n,
    })
    for i, column in enumerate(QUANTILE_COLUMNS):
        frame[column] = np.asarray(y) - 3 + i
    if kind == "point":
        frame["point_prediction"] = np.asarray(y) - 1
        frame.loc[:, QUANTILE_COLUMNS] = np.nan
    return frame


def test_contract_roundtrip_and_raw_gfs_point_only(tmp_path):
    valid = validate_predictions(rows(), CONFIG)
    assert valid.loc[0, "gfs_service_coordinates"] == "[32.0,118.8]"
    assert valid.target_time_utc.dt.tz is not None
    assert list(valid.loc[:, QUANTILE_COLUMNS]) == list(QUANTILE_COLUMNS)
    path = tmp_path / "predictions.csv"
    write_predictions(valid, path, CONFIG)
    assert len(read_predictions(path, CONFIG)) == 2
    raw = validate_predictions(rows(model="raw_gfs", kind="point"), CONFIG)
    assert raw.loc[:, QUANTILE_COLUMNS].isna().all().all()
    with pytest.raises(ValueError, match="raw_gfs is a point"):
        validate_predictions(rows(model="raw_gfs"), CONFIG)


def test_official_gate_and_explicit_utc():
    bad = rows()
    bad["result_status"] = "official"
    with pytest.raises(ValueError, match="validated implementation and official execution"):
        validate_predictions(bad, CONFIG)
    bad["execution_level"] = "official"
    bad["model_id"] = "pinn"
    with pytest.raises(ValueError, match="PINN remains experimental"):
        validate_predictions(bad, CONFIG)
    bad = rows()
    bad["forecast_issue_time_utc"] = ["2025-08-17 12:00", "2025-08-18 12:00"]
    with pytest.raises(ValueError, match="explicit UTC"):
        validate_predictions(bad, CONFIG)


def test_metrics_truncated_integral_point_secondary_and_group_crossing():
    frame = rows()
    y = frame.y.to_numpy()
    q = frame.loc[:, QUANTILE_COLUMNS].to_numpy()
    metrics = probability_metrics(y, q, levels=LEVELS)
    assert metrics["mean_pinball"] == pytest.approx(np.mean([
        np.maximum(t * (y - q[:, i]), (t - 1) * (y - q[:, i])).mean()
        for i, t in enumerate((.05, .10, .25, .50, .75, .90, .95))
    ]))
    assert metrics["crps_q7_trunc"] == pytest.approx(
        truncated_quantile_crps(y, q, LEVELS))
    assert metrics["mae"] == pytest.approx(point_metrics(y, q[:, 3])["mae"])
    frame.loc[1, "q0.05"] = frame.loc[1, "q0.10"] + 2
    frame.loc[0, "location_id"] = "good"
    frame.loc[1, "location_id"] = "crossed"
    summary = grouped_metrics(frame, "location_id", CONFIG).set_index("location_id")
    assert summary.loc["good", "crossing_rate"] == 0
    assert summary.loc["crossed", "crossing_rate"] == 1


def test_reliability_and_pit_do_not_invent_probability_tails():
    y = np.array([0., 5., 11.])
    q = np.tile(np.array([1., 2., 3., 5., 7., 8., 9.]), (3, 1))
    pit = pit_central(y, q, levels=LEVELS)
    assert np.isnan(pit[0]) and pit[1] == pytest.approx(.5) and np.isnan(pit[2])
    rel = quantile_reliability(y, q, levels=LEVELS)
    assert list(rel.columns) == ["quantile", "empirical_cdf", "calibration_error", "n"]


def test_time_ordered_calibration_signed_residual_and_no_issue_leakage():
    calibration = rows()
    calibrator = AdditiveQuantileCalibrator(CONFIG, lower_bound=None).fit(
        calibration, fit_end="2025-07-01T00:00:00Z",
        early_stop_end="2025-07-31T23:00:00Z")
    assert calibrator.deltas_["q0.50"] == pytest.approx(0.0)
    test = rows(times=("2025-09-05T12:00:00Z", "2025-09-06T12:00:00Z"))
    calibrated = calibrator.apply(test)
    assert np.all(np.diff(calibrated.loc[:, QUANTILE_COLUMNS].to_numpy(), axis=1) >= 0)
    too_early = rows(times=("2025-08-23T12:00:00Z", "2025-08-24T12:00:00Z"))
    with pytest.raises(ValueError, match="overlaps or follows a forecast issue"):
        calibrator.apply(too_early)
    with pytest.raises(ValueError, match="calibration truth must follow"):
        AdditiveQuantileCalibrator(CONFIG).fit(calibration, fit_end="2025-08-01T00:00:00Z",
                                        early_stop_end="2025-08-20T12:00:00Z")


def test_bootstrap_samples_whole_calendar_days_not_five_rows_as_five_days():
    times = ["2025-01-01T00:00:00Z"] * 5 + ["2025-01-02T00:00:00Z"] * 2 + [
        "2025-01-05T00:00:00Z"] * 8
    index = block_bootstrap_indices(times, block_days=2, rng=np.random.default_rng(3))
    # Within any selected day each original observation must be copied together.
    counts = np.bincount(index, minlength=len(times))
    assert len(set(counts[:5])) == len(set(counts[5:7])) == len(set(counts[7:])) == 1
    frame = pd.DataFrame({"target_time_utc": times, "value": np.arange(len(times))})
    result = bootstrap_metric(frame, lambda sample: float(sample.value.mean()),
                              block_days=2, replicates=12, seed=3)
    assert result["time_basis"] == "UTC calendar days"
    assert result["ci_low"] <= result["ci_high"]
