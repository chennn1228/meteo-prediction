"""Group-local scoring; no overall crossing/reliability values are copied."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from s07_prediction.schema import QUANTILE_COLUMNS, validate_predictions
from s09_metrics.deterministic import point_metrics
from s09_metrics.probabilistic import probability_metrics
from s09_metrics.reliability import pit_central, quantile_reliability

from .summaries import EvaluationTables, rank_probability_models

# Prevent model/experiment/seed/version mixtures from being reported as one
# estimate without traceable provenance. Outer folds are pooled by default;
# request outer_fold as a grouping dimension when fold-local scores are needed.
PROVENANCE_KEYS = (
    "experiment_id", "protocol_revision", "data_version", "feature_version",
    "model_id", "implementation_level", "execution_level", "result_status",
    "prediction_type", "seed",
)


def _group_keys(frame: pd.DataFrame, dimensions: Sequence[str]) -> list[str]:
    if any(name not in frame.columns for name in dimensions):
        raise ValueError("all grouping dimensions must be prediction columns")
    return list(dict.fromkeys((*PROVENANCE_KEYS, *dimensions)))


def evaluate_groups(frame: pd.DataFrame,
                    dimensions: Sequence[str] = ()) -> EvaluationTables:
    """Return separate probability, auxiliary point, and reliability tables."""
    clean = validate_predictions(frame)
    keys = _group_keys(clean, dimensions)
    probability_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for group_key, subset in clean.groupby(keys, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(keys, key_values))
        y = subset.y.to_numpy(dtype=float)
        is_quantile = identity["prediction_type"] == "quantile"
        if is_quantile:
            q = subset.loc[:, QUANTILE_COLUMNS].to_numpy(dtype=float)
            scores = probability_metrics(y, q)
            # Only probability scores enter the primary table. The q0.50
            # deterministic metrics are emitted in point_secondary below.
            primary_scores = {name: value for name, value in scores.items()
                              if name not in {"mae", "rmse", "bias", "rmse_skill"}}
            probability_rows.append({**identity, "n": len(subset), **primary_scores})
            point_prediction = q[:, 3]
            if scores["crossing_rate"] == 0:
                central_pit = pit_central(y, q)
                pit_status = "interior_only_tails_unknown"
                pit_available_fraction = float(np.isfinite(central_pit).mean())
                pit_central_mean = (float(np.nanmean(central_pit))
                                    if np.isfinite(central_pit).any() else np.nan)
            else:
                pit_status = "unavailable_crossing"
                pit_available_fraction = np.nan
                pit_central_mean = np.nan
            for reliability_row in quantile_reliability(y, q).to_dict("records"):
                reliability_rows.append({
                    **identity, "n": len(subset), **reliability_row,
                    "pit_status": pit_status,
                    "pit_available_fraction": pit_available_fraction,
                    "pit_central_mean": pit_central_mean,
                })
        else:
            point_prediction = subset.point_prediction.to_numpy(dtype=float)
        point_rows.append({
            **identity, "n": len(subset), "metric_role": "auxiliary_point",
            "point_source": "q0.50" if is_quantile else "point_prediction",
            **point_metrics(y, point_prediction),
        })
    probability = pd.DataFrame(probability_rows)
    if not probability.empty:
        probability = rank_probability_models(probability)
    return EvaluationTables(
        probability_primary=probability,
        point_secondary=pd.DataFrame(point_rows),
        reliability=pd.DataFrame(reliability_rows),
    )
