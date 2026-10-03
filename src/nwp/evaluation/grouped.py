"""Provenance-safe overall and group-local evaluation tables."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from nwp.experiment.prediction import (
    quantile_columns, quantiles, read_predictions, validate_predictions)
from .metrics import (
    deterministic, pit_central, probability_metrics, quantile_reliability)


PROVENANCE_KEYS = (
    "experiment_id", "protocol_revision", "data_version", "feature_version",
    "model_id", "implementation_level", "execution_level", "result_status",
    "prediction_type", "seed")
SAMPLE_KEYS = (
    "experiment_id", "protocol_revision", "data_version", "feature_version",
    "execution_level", "result_status", "seed", "location_id",
    "target_time_utc", "forecast_issue_time_utc", "lead_time", "outer_fold")


@dataclass(frozen=True)
class EvaluationTables:
    probability_primary: pd.DataFrame
    point_secondary: pd.DataFrame
    reliability: pd.DataFrame


@dataclass(frozen=True)
class EvaluationReport:
    overview: EvaluationTables
    grouped: dict[str, EvaluationTables]


_SAFE_GROUP = re.compile(r"^[A-Za-z0-9_.-]+$")


def write_evaluation_report(report: EvaluationReport, directory: Path) -> Path:
    """Persist each metric table once and return a provenance index path."""
    directory.mkdir(parents=True, exist_ok=True)

    def write_tables(tables: EvaluationTables, target: Path) -> dict[str, str]:
        target.mkdir(parents=True, exist_ok=True)
        paths = {}
        for name in ("probability_primary", "point_secondary", "reliability"):
            path = target / f"{name}.csv"
            getattr(tables, name).to_csv(path, index=False)
            paths[name] = str(path)
        return paths

    payload: dict[str, Any] = {
        "overview": write_tables(report.overview, directory), "grouped": {}}
    for name, tables in sorted(report.grouped.items()):
        if not _SAFE_GROUP.fullmatch(name):
            raise ValueError(f"unsafe evaluation grouping name: {name!r}")
        payload["grouped"][name] = write_tables(
            tables, directory / "grouped" / name)
    index = directory / "metrics_index.json"
    index.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return index


def rank_probability_models(table: pd.DataFrame) -> pd.DataFrame:
    required = {"prediction_type", "mean_pinball", "model_id"}
    if not required.issubset(table.columns):
        raise ValueError(
            f"probability table missing {sorted(required - set(table.columns))}")
    if (table.prediction_type != "quantile").any():
        raise ValueError("point-only baselines cannot enter probability model ranking")
    return table.sort_values(
        ["mean_pinball", "model_id"], kind="stable").reset_index(drop=True)


def grouped_metrics(frame: pd.DataFrame, by: str | Sequence[str],
                    protocol_config: Mapping[str, Any]) -> pd.DataFrame:
    """Recompute probability metrics independently inside every group."""
    clean = validate_predictions(frame, protocol_config)
    if (clean.prediction_type != "quantile").any():
        raise ValueError("grouped probability metrics cannot consume point-only rows")
    names = [by] if isinstance(by, str) else list(by)
    if not names or any(name not in clean.columns for name in names):
        raise ValueError("by must name existing grouping columns")
    columns, levels = quantile_columns(protocol_config), quantiles(protocol_config)
    rows = []
    for key, subset in clean.groupby(names, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        rows.append({
            **dict(zip(names, key_values)), "n": len(subset),
            **probability_metrics(
                subset.y.to_numpy(), subset.loc[:, columns].to_numpy(),
                levels=levels)})
    return pd.DataFrame(rows)


def _reference_id(model_id: str, protocol_config: Mapping[str, Any],
                  model_config: Mapping[str, Any]) -> str | None:
    references = protocol_config["evaluation"]["point_metric_references"]
    if model_id == "climatology":
        return None
    if model_id == "raw_gfs":
        return references["raw_gfs"]
    family = model_config["registry"].get(model_id, {}).get("family", "corrected")
    return references["other_fixed" if family == "baseline" else "corrected"]


def _reference_vector(all_rows: pd.DataFrame, subset: pd.DataFrame,
                      reference_id: str, *, required: bool) -> np.ndarray | None:
    source = all_rows.loc[all_rows.model_id == reference_id]
    if source.empty:
        if required:
            raise ValueError(
                f"missing preregistered RMSE skill reference: {reference_id}")
        return None
    if source.duplicated(list(SAMPLE_KEYS)).any():
        raise ValueError(f"duplicate reference sample keys for {reference_id}")
    left = subset.loc[:, [*SAMPLE_KEYS, "y"]].copy()
    left["_order"] = np.arange(len(left))
    matched = left.merge(
        source.loc[:, [*SAMPLE_KEYS, "y", "point_prediction"]],
        on=list(SAMPLE_KEYS), how="left", sort=False,
        suffixes=("_target", "_reference"))
    if len(matched) != len(subset) or matched.point_prediction.isna().any():
        if required:
            raise ValueError(
                f"incomplete preregistered RMSE skill reference: {reference_id}")
        return None
    matched = matched.sort_values("_order")
    if not np.allclose(
            matched.y_target, matched.y_reference, rtol=0, atol=1e-9):
        raise ValueError(
            "reference and model observations disagree on matched samples")
    return matched.point_prediction.to_numpy(dtype=float)


def evaluate_groups(frame: pd.DataFrame, protocol_config: Mapping[str, Any],
                    model_config: Mapping[str, Any],
                    dimensions: Sequence[str] = (), *,
                    require_references: bool = False) -> EvaluationTables:
    clean = validate_predictions(frame, protocol_config)
    if clean.duplicated(["model_id", *SAMPLE_KEYS]).any():
        raise ValueError("duplicate model predictions for one forecast sample")
    if any(name not in clean.columns for name in dimensions):
        raise ValueError("all grouping dimensions must be prediction columns")
    keys = list(dict.fromkeys((*PROVENANCE_KEYS, *dimensions)))
    q_columns = quantile_columns(protocol_config)
    levels = quantiles(protocol_config)
    median = levels.index(0.5)
    probability_rows, point_rows, reliability_rows = [], [], []
    for group_key, subset in clean.groupby(keys, dropna=False, sort=True):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(keys, key_values))
        y = subset.y.to_numpy(dtype=float)
        is_quantile = identity["prediction_type"] == "quantile"
        if is_quantile:
            q = subset.loc[:, q_columns].to_numpy(dtype=float)
            scores = probability_metrics(y, q, levels=levels)
            probability_rows.append({
                **identity, "n": len(subset),
                **{name: value for name, value in scores.items()
                   if name not in {"mae", "rmse", "bias", "r2", "rmse_skill"}}})
            point_prediction = q[:, median]
            if scores["crossing_rate"] == 0:
                pit = pit_central(y, q, levels=levels)
                pit_status = "interior_only_tails_unknown"
                pit_fraction = float(np.isfinite(pit).mean())
                pit_mean = (float(np.nanmean(pit))
                            if np.isfinite(pit).any() else np.nan)
            else:
                pit_status, pit_fraction, pit_mean = (
                    "unavailable_crossing", np.nan, np.nan)
            for row in quantile_reliability(
                    y, q, levels=levels).to_dict("records"):
                reliability_rows.append({
                    **identity, "n": len(subset), **row,
                    "pit_status": pit_status,
                    "pit_available_fraction": pit_fraction,
                    "pit_central_mean": pit_mean})
        else:
            point_prediction = subset.point_prediction.to_numpy(dtype=float)
        reference_id = _reference_id(
            str(identity["model_id"]), protocol_config, model_config)
        reference = (_reference_vector(
            clean, subset, reference_id, required=require_references)
            if reference_id is not None else None)
        point_rows.append({
            **identity, "n": len(subset), "metric_role": "auxiliary_point",
            "point_source": "q0.50" if is_quantile else "point_prediction",
            "rmse_skill_reference": reference_id,
            "rmse_skill_status": (
                "not_applicable" if reference_id is None else
                "matched" if reference is not None else "reference_unavailable"),
            **deterministic(y, point_prediction, reference_prediction=reference)})
    probability = pd.DataFrame(probability_rows)
    if not probability.empty:
        probability = rank_probability_models(probability)
    return EvaluationTables(
        probability_primary=probability,
        point_secondary=pd.DataFrame(point_rows),
        reliability=pd.DataFrame(reliability_rows))


def evaluate_predictions(
        predictions: pd.DataFrame | str | Path,
        protocol_config: Mapping[str, Any], model_config: Mapping[str, Any], *,
        formal: bool = False,
        groupings: tuple[str, ...] | None = None) -> EvaluationReport:
    clean = (read_predictions(predictions, protocol_config)
             if isinstance(predictions, (str, Path))
             else validate_predictions(predictions, protocol_config))
    if formal and not (
            (clean.implementation_level == "validated")
            & (clean.execution_level == "official")
            & (clean.result_status == "official")).all():
        raise ValueError(
            "formal evaluation rejects prototype, development or nonofficial rows")
    dimensions = groupings if groupings is not None else (
        "lead_time", "location_id", "outer_fold", "season")
    overview = evaluate_groups(
        clean, protocol_config, model_config, require_references=formal)
    grouped = {
        name: evaluate_groups(
            clean, protocol_config, model_config, (name,),
            require_references=formal)
        for name in dimensions if name in clean.columns}
    return EvaluationReport(overview=overview, grouped=grouped)
