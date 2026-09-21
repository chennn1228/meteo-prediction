"""D+3 test issues inside the calibration month use only known truth."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402
from s08_calibration.quantile import CausalIssueQuantileCalibrator  # noqa: E402


def rows(targets, leads, *, observed=True):
    records = []
    for index, (target, lead) in enumerate(zip(targets, leads)):
        target = pd.Timestamp(target)
        centre = 100.0 + index
        records.append({
            "model_id": "ridge_mos", "implementation_level": "validated",
            "execution_level": "development", "prediction_type": "quantile",
            "location_id": "nanjing_1", "requested_coordinates": [32.1, 118.7],
            "gfs_service_coordinates": [32.0, 118.8],
            "truth_service_coordinates": [32.05, 118.75],
            "target_time_utc": target.isoformat(),
            "forecast_issue_time_utc": (target - pd.Timedelta(hours=lead)).isoformat(),
            "lead_time": lead, "outer_fold": "final", "inner_fold": "none",
            "seed": 0, "y": centre + 10 if observed else np.nan,
            "point_prediction": centre,
            "data_version": "data-v1", "feature_version": "features-v1",
            "protocol_revision": "p-v2", "experiment_id": "exp-1",
            "result_status": "provisional",
            **dict.fromkeys(QUANTILE_COLUMNS, centre),
        })
    return pd.DataFrame(records)


def test_pre_september_d3_issue_uses_only_known_august_truth():
    calibration = rows(["2025-08-27T00:00:00Z", "2025-08-28T00:00:00Z",
                        "2025-08-29T00:00:00Z", "2025-08-31T00:00:00Z"],
                       [24, 24, 24, 24])
    test = rows(["2025-09-01T00:00:00Z", "2025-09-02T00:00:00Z"],
                [72, 24], observed=False)
    calibrator = CausalIssueQuantileCalibrator().fit(
        calibration, fit_end="2025-06-20T00:00:00Z",
        early_stop_end="2025-07-31T23:59:59Z")
    output = calibrator.apply(test)
    assert output.calibration_available_rows.tolist() == [2, 4]
    assert output.calibration_latest_truth_utc.tolist() == [
        "2025-08-28T00:00:00+00:00", "2025-08-31T00:00:00+00:00"]
    np.testing.assert_allclose(output.point_prediction, [110., 111.])
    assert output.y.isna().all()


def test_causal_calibrator_refuses_unavailable_or_overlapping_truth():
    calibration = rows(["2025-08-11T00:00:00Z"], [24])
    calibrator = CausalIssueQuantileCalibrator().fit(
        calibration, fit_end="2025-06-20T00:00:00Z",
        early_stop_end="2025-07-31T23:59:59Z")
    too_early = rows(["2025-08-11T00:00:00Z"], [24], observed=False)
    with pytest.raises(ValueError, match="overlaps the calibration block"):
        calibrator.apply(too_early)
    # Later target but issue precedes the first calibration truth.
    later = rows(["2025-09-01T00:00:00Z"], [24 * 22], observed=False)
    with pytest.raises(ValueError, match="no calibration truth known"):
        calibrator.apply(later)
