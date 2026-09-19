"""Group-local scoring; no overall crossing/reliability values are copied."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from s01_core.config_loader import load_manifest
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
SAMPLE_KEYS = (
    "experiment_id", "protocol_revision", "data_version", "feature_version",
    "execution_level", "result_status", "seed", "location_id", "target_time_utc",
    "forecast_issue_time_utc", "lead_time", "outer_fold", "inner_fold",
)


def _reference_id(model_id: str) -> str | None:
    config = load_manifest()
    references = config["point_metric_references"]
    if model_id == "climatology":
        return None
    if model_id == "raw_gfs":
        return references["raw_gfs"]
    family = next((entry["family"] for entry in config["models"]
                   if entry["id"] == model_id), "corrected")
    return references["other_fixed" if family == "baseline" else "corrected"]


def _reference_vector(all_rows: pd.DataFrame, subset: pd.DataFrame,
                      reference_id: str, *, required: bool) -> np.ndarray | None:
    source = all_rows.loc[all_rows.model_id == reference_id]
    if source.empty:
        if required:
            raise ValueError(f"missing preregistered RMSE skill reference: {reference_id}")
        return None
    if source.duplicated(list(SAMPLE_KEYS)).any():
        raise ValueError(f"duplicate reference sample keys for {reference_id}")
    left = subset.loc[:, [*SAMPLE_KEYS, "y"]].copy()
    left["_order"] = np.arange(len(left))
    matched = left.merge(source.loc[:, [*SAMPLE_KEYS, "y", "point_prediction"]],
                         on=list(SAMPLE_KEYS), how="left", sort=False,
                         suffixes=("_target", "_reference"))
    if len(matched) != len(subset) or matched.point_prediction.isna().any():
        if required:
            raise ValueError(f"incomplete preregistered RMSE skill reference: {reference_id}")
        return None
    matched = matched.sort_values("_order")
    if not np.allclose(matched.y_target, matched.y_reference, rtol=0, atol=1e-9):
        raise ValueError("reference and model observations disagree on matched samples")
    return matched.point_prediction.to_numpy(dtype=float)


def _group_keys(frame: pd.DataFrame, dimensions: Sequence[str]) -> list[str]:
    if any(name not in frame.columns for name in dimensions):
        raise ValueError("all grouping dimensions must be prediction columns")
    return list(dict.fromkeys((*PROVENANCE_KEYS, *dimensions)))


def evaluate_groups(frame: pd.DataFrame,
                    dimensions: Sequence[str] = (),
                    *, require_references: bool = False) -> EvaluationTables:
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
                              if name not in {"mae", "rmse", "bias", "r2", "rmse_skill"}}
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
        reference_id = _reference_id(str(identity["model_id"]))
        reference = (_reference_vector(clean, subset, reference_id, required=require_references)
                     if reference_id is not None else None)
        point_rows.append({
            **identity, "n": len(subset), "metric_role": "auxiliary_point",
            "point_source": "q0.50" if is_quantile else "point_prediction",
            "rmse_skill_reference": reference_id,
            "rmse_skill_status": ("not_applicable" if reference_id is None else
                                  "matched" if reference is not None else "reference_unavailable"),
            **point_metrics(y, point_prediction, reference_prediction=reference),
        })
    probability = pd.DataFrame(probability_rows)
    if not probability.empty:
        probability = rank_probability_models(probability)
    return EvaluationTables(
        probability_primary=probability,
        point_secondary=pd.DataFrame(point_rows),
        reliability=pd.DataFrame(reliability_rows),
    )
