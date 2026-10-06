from __future__ import annotations

import numpy as np
import pandas as pd

import nwp.evaluation.interpretation as interpretation
from nwp.evaluation.metrics import probability_metrics
from nwp.splits.rolling import InnerFold, OuterFold


def test_bounded_group_mechanism_evidence_executes_lightgbm(monkeypatch):
    random = np.random.default_rng(7)

    def block(size: int, start: str) -> pd.DataFrame:
        signal = random.normal(size=size)
        return pd.DataFrame({
            "a": signal,
            "b": random.normal(size=size),
            "cloud_cover_fcst": np.full(size, 50.0),
            "lead_time": np.resize([24, 48, 72], size),
            "y": 10.0 + 3.0 * signal + random.normal(scale=0.1, size=size),
            "target_time_utc": pd.date_range(start, periods=size, freq="h", tz="UTC"),
            "forecast_issue_time_utc": pd.date_range(
                start, periods=size, freq="h", tz="UTC") - pd.Timedelta(hours=24),
        })

    fit = block(48, "2024-02-01")
    early = block(24, "2024-03-01")
    score = block(24, "2024-04-01")
    stamp = pd.Timestamp("2024-01-01", tz="UTC")
    outer = OuterFold("outer_1", fit, score, stamp, stamp)
    inner = InnerFold(
        "inner_1", fit, early, score, stamp, stamp, stamp, stamp)
    monkeypatch.setattr(
        interpretation, "outer_folds", lambda *_args, **_kwargs: [outer])
    monkeypatch.setattr(
        interpretation, "inner_folds", lambda *_args, **_kwargs: [inner])
    monkeypatch.setattr(
        interpretation, "fit_fold_preprocessing",
        lambda fit_rows, early_rows, score_rows, **_kwargs: (
            (fit_rows[["a", "b"]], early_rows[["a", "b"]],
             score_rows[["a", "b"]]), {}))

    levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    parameters = {
        "learning_rate": 0.1, "num_leaves": 7,
        "min_child_samples": 4, "max_depth": 4,
    }
    estimators = interpretation._fit_quantile_estimators(
        "lightgbm", parameters, fit[["a", "b"]], early[["a", "b"]],
        fit.y.to_numpy(), early.y.to_numpy(), levels, 0)
    baseline = probability_metrics(
        score.y.to_numpy(),
        interpretation._tree_predictions(estimators, score[["a", "b"]]),
        levels=levels)["mean_pinball"]
    selection = {"outer": {"outer_1": {"lgbm": {
        "parameters": parameters,
        "selected_candidate": 0,
        "trials": [{"inner": "inner_1", "candidate": 0,
                    "mean_pinball": baseline}],
    }}}}
    protocol = {
        "probability": {"quantiles": levels},
        "validation": {"gap_days": 10},
        "development_period": {
            "start": "2024-02-01T00:00:00Z",
            "end": "2025-08-31T23:59:59Z"},
        "seed_policy": {"tuning_seed": 0},
    }
    feature_config = {"feature_groups": {"signal": ["a"]}}
    analysis_config = {"evidence": {
        "reference_model": "lgbm", "permutation_repeats": 1}}
    model_config = {"registry": {"lgbm": {
        "quantile_adapter": "tree", "implementation": "lightgbm"}}}

    ablation, permutation = interpretation.group_mechanism_evidence(
        pd.concat([fit, early, score]), selection,
        protocol_config=protocol, feature_config=feature_config,
        analysis_config=analysis_config, model_config=model_config,
        eligible=lambda rows: rows)

    assert ablation.group.tolist() == ["signal"]
    assert permutation.group.tolist() == ["signal"]
    assert np.isfinite(ablation.mean_pinball).all()
    assert np.isfinite(permutation.mean_pinball).all()
