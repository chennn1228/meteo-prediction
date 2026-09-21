"""Fixed CPU value-ladder definitions and strict issue-time persistence."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))

from s06_models.cpu_fixed import FIXED_MODEL_IDS, predict_fixed_cpu  # noqa: E402
from s01_core.config_loader import load_manifest  # noqa: E402


def rows() -> pd.DataFrame:
    target = pd.date_range("2024-02-01T04:00:00Z", periods=12, freq="D")
    return pd.DataFrame({
        "location_id": ["service-1"] * len(target),
        "target_time_utc": target,
        "forecast_issue_time_utc": target - pd.Timedelta(days=1),
        "lead_time": [24] * len(target),
        "y": np.arange(len(target), dtype=float) + 100,
        "ghi_fcst": np.arange(len(target), dtype=float) + 95,
        "ghi_clear_sky": [200.] * len(target),
    })


def test_fixed_status_does_not_require_six_candidates():
    registry = {item["id"]: item for item in load_manifest()["models"]}
    assert all(not registry[name]["tuning_required"] for name in FIXED_MODEL_IDS)
    fit, early, score = rows().iloc[:8], rows().iloc[8:10], rows().iloc[10:]
    for name in FIXED_MODEL_IDS - {"linear_mos"}:
        prediction, receipt = predict_fixed_cpu(name, fit, early, score)
        assert prediction.shape == (2,)
        assert np.isfinite(prediction).all()
        assert receipt["definition"]


def test_persistence_ignores_truth_not_known_at_issue():
    frame = rows()
    fit, early, score = frame.iloc[:8], frame.iloc[8:10], frame.iloc[10:].copy()
    original, _ = predict_fixed_cpu("persistence", fit, early, score)
    changed = score.copy()
    changed["y"] = [1e9, -1e9]
    candidate, _ = predict_fixed_cpu("persistence", fit, early, changed)
    np.testing.assert_allclose(original, candidate)
    assert original[0] == pytest.approx(frame.y.iloc[8])


def test_raw_gfs_is_point_only_and_bias_fits_fit_rows():
    frame = rows()
    fit, early, score = frame.iloc[:8], frame.iloc[8:10], frame.iloc[10:]
    raw, _ = predict_fixed_cpu("raw_gfs", fit, early, score)
    bias, _ = predict_fixed_cpu("bias_correction", fit, early, score)
    np.testing.assert_allclose(raw, score.ghi_fcst)
    np.testing.assert_allclose(bias, score.ghi_fcst + 5)


def test_missing_latest_truth_does_not_hide_earlier_valid_persistence():
    frame = rows()
    fit, early, score = frame.iloc[:8], frame.iloc[8:10].copy(), frame.iloc[10:]
    early.loc[early.index[0], "y"] = np.nan
    prediction, _ = predict_fixed_cpu("persistence", fit, early, score)
    assert prediction[0] == pytest.approx(frame.y.iloc[7])
