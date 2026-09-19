"""Targeted architecture and identity-gate regression checks."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "project_manifest.yaml").exists())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s03_models" / "train"))

from s06_models.model_status import require_deep_execution  # noqa: E402
from cloud_impute import FoldCloudImputer, impute_cloud_forecast  # noqa: E402
import train_v1 as tv  # noqa: E402


def test_topk_lags_use_independent_shifts():
    torch = pytest.importorskip("torch")
    from formal_architectures import aggregate_lags
    values = torch.tensor([0., 1., 2., 3.]).view(1, 1, 4, 1)
    lags = torch.tensor([[1, 2]])
    weights = torch.tensor([[.25, .75]])
    result = aggregate_lags(values, lags, weights).flatten()
    expected = .25 * torch.roll(values.flatten(), 1) + .75 * torch.roll(values.flatten(), 2)
    assert torch.allclose(result, expected)
    assert not torch.allclose(result, torch.roll(values.flatten(), 2))


def test_identity_never_in_tree_or_linear_features():
    frame = pd.DataFrame({name: [1., 2.] for name in tv.FEATURES_NUM})
    frame["station_id"] = ["A", "B"]
    frame["location_id"] = ["x", "y"]
    frame["source_grid_id"] = ["g1", "g2"]
    frame["season"] = ["spring", "winter"]
    columns = tv.tree_matrix(frame).columns.tolist()
    assert not any("station" in c or "location" in c or "grid_id" in c for c in columns)
    assert tv.CAT_COLS == ["season"]
    assert "station" not in tv.target_maps(frame.assign(y=[1., 2.]), "y")


def test_whole_dataset_cloud_imputer_blocked_and_fit_boundary():
    with pytest.raises(RuntimeError, match="retired"):
        impute_cloud_forecast(pd.DataFrame())
    fit = pd.DataFrame({"target_time_utc": pd.date_range("2024-01-01", periods=4, tz="UTC"),
                        "forecast_issue_time_utc": pd.date_range("2023-12-31", periods=4, tz="UTC"),
                        "cloud_cover_fcst": [10., 20., 30., np.nan], "temp_fcst": [1., 2., 3., 4.]})
    later = pd.DataFrame({"target_time_utc": [pd.Timestamp("2025-01-01", tz="UTC")],
                          "cloud_cover_fcst": [np.nan], "temp_fcst": [999.]})
    imputer = FoldCloudImputer(("temp_fcst",)).fit(fit)
    transformed = imputer.transform(later)
    assert transformed.cloud_cover_fcst.iloc[0] == 20.
    assert transformed.attrs["cloud_imputation"]["fit_end_utc"].startswith("2024")
    assert transformed.cloud_fcst_imputed.iloc[0]
    with pytest.raises(ValueError, match="identity"):
        FoldCloudImputer(("station_id",)).fit(fit)


def test_prototype_and_pinn_cannot_be_official():
    require_deep_execution("mlp", "prototype", "smoke")
    with pytest.raises(PermissionError, match="validated"):
        require_deep_execution("mlp", "prototype", "official")
    with pytest.raises(PermissionError, match="PINN"):
        require_deep_execution("pinn", "validated", "official")
