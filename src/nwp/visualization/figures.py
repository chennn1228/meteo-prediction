"""Config-neutral run figure generation from metric and analysis source tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from nwp.experiment.prediction import quantile_columns
from .style import (
    FigureContract, PALETTE, apply_publication_style, relationship, save_figure)


def write_figure_index(path: Path, figures: Iterable[dict[str, Any]]) -> None:
    payload = list(figures)
    for item in payload:
        if not {"figure_id", "source_table", "generation_function", "caption", "status"} <= set(item):
            raise ValueError("figure index entry lacks required provenance")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bar_figure(table: pd.DataFrame, *, value: str, ylabel: str, title: str,
                source: Path, output: Path, figure_id: str,
                result_status: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    data = table.dropna(subset=[value]).sort_values(value, kind="stable")
    if data.empty:
        raise ValueError(f"figure source has no finite {value}")
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, .32 * len(data))))
    ax.barh(data.model_id.astype(str), data[value], color=PALETTE["method"])
    ax.invert_yaxis()
    ax.set(xlabel=ylabel, ylabel="Model", title=title)
    contract = FigureContract(
        conclusion=title, evidence=(value,), source_data=str(source),
        result_status=result_status,
        n_definition="n is reported in the source metric table",
        statistical_definition=f"Models ordered by ascending {value}")
    paths = save_figure(fig, output / figure_id, contract)
    return {
        "figure_id": figure_id, "source_table": str(source),
        "generation_function": "nwp.visualization.figures._bar_figure",
        "caption": title, "status": result_status,
        "files": [str(path) for path in paths],
    }


def _reliability_figure(table: pd.DataFrame, *, source: Path, output: Path,
                        result_status: str) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("reliability source table is empty")
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    for model_id, subset in table.groupby("model_id", sort=True):
        subset = subset.sort_values("nominal_probability")
        ax.plot(subset.nominal_probability, subset.empirical_probability,
                marker="o", ms=3, lw=1, label=str(model_id))
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], ls="--", lw=.8)
    ax.set(xlabel="Nominal quantile probability",
           ylabel="Empirical proportion below quantile",
           title="Quantile reliability")
    ax.legend(fontsize=6, ncol=2)
    contract = FigureContract(
        conclusion="Calibration is assessed against nominal quantile levels",
        evidence=("nominal_probability", "empirical_probability"),
        source_data=str(source), result_status=result_status,
        n_definition="one point per model and registered quantile",
        statistical_definition="empirical fraction y <= predicted quantile")
    paths = save_figure(fig, output / "reliability", contract)
    return {
        "figure_id": "reliability", "source_table": str(source),
        "generation_function": "nwp.visualization.figures._reliability_figure",
        "caption": "Quantile reliability by model", "status": result_status,
        "files": [str(path) for path in paths],
    }


def grouped_metric_figure(
        table: pd.DataFrame, *, dimension: str, metric: str, source: Path,
        output: Path, result_status: str, figure_id: str | None = None,
) -> dict[str, Any]:
    """Plot any run-owned grouped metric without site/model hard-coding."""
    import matplotlib.pyplot as plt

    required = {"model_id", dimension, metric}
    if missing := required - set(table):
        raise ValueError(f"grouped figure source missing {sorted(missing)}")
    data = table.dropna(subset=[metric]).copy()
    if data.empty:
        raise ValueError(f"grouped figure has no finite {metric}")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model_id, subset in data.groupby("model_id", sort=True):
        subset = subset.sort_values(dimension, kind="stable")
        ax.plot(subset[dimension].astype(str), subset[metric], marker="o",
                ms=3, lw=1, label=str(model_id))
    ax.set(xlabel=dimension.replace("_", " ").title(),
           ylabel=metric.replace("_", " "),
           title=f"{metric.replace('_', ' ').title()} by {dimension}")
    ax.legend(fontsize=6, ncol=2)
    stem = figure_id or f"{metric}_by_{dimension}"
    contract = FigureContract(
        conclusion=f"{metric} varies across {dimension}", evidence=(metric,),
        source_data=str(source), result_status=result_status,
        n_definition="group-local n is reported in the source table",
        statistical_definition="metric recomputed independently within each group")
    paths = save_figure(fig, output / stem, contract)
    return {
        "figure_id": stem, "source_table": str(source),
        "generation_function": "nwp.visualization.figures.grouped_metric_figure",
        "caption": f"{metric} by {dimension}", "status": result_status,
        "files": [str(path) for path in paths],
    }


def tuning_trials_figure(table: pd.DataFrame, *, source: Path, output: Path,
                         result_status: str) -> dict[str, Any]:
    """Show the equal-budget inner-fold candidate evidence for all tuned models."""
    import matplotlib.pyplot as plt

    required = {"model_id", "candidate", "mean_pinball", "status"}
    if missing := required - set(table):
        raise ValueError(f"tuning source missing {sorted(missing)}")
    valid = table.loc[table.status == "ok"].copy()
    if valid.empty:
        raise ValueError("tuning source contains no completed trials")
    summary = valid.groupby(
        ["model_id", "candidate"], as_index=False).mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model_id, subset in summary.groupby("model_id", sort=True):
        ax.plot(subset.candidate, subset.mean_pinball, marker="o", ms=3,
                label=str(model_id))
    ax.set(xlabel="Registered candidate index", ylabel="Mean inner-fold pinball",
           title="Equal-budget tuning evidence")
    ax.legend(fontsize=6)
    contract = FigureContract(
        conclusion="Candidates are compared on the same inner-fold budget",
        evidence=("candidate", "mean_pinball"), source_data=str(source),
        result_status=result_status,
        n_definition="one mean per model, candidate and outer-fold ledger",
        statistical_definition="mean of registered inner scoring-fold pinball losses")
    paths = save_figure(fig, output / "tuning_candidates", contract)
    return {
        "figure_id": "tuning_candidates", "source_table": str(source),
        "generation_function": "nwp.visualization.figures.tuning_trials_figure",
        "caption": "Equal-budget tuning candidate performance",
        "status": result_status, "files": [str(path) for path in paths],
    }


def prediction_case_figure(
        predictions: pd.DataFrame, *, protocol_config: dict[str, Any],
        source: Path, output: Path, result_status: str,
) -> dict[str, Any]:
    """Render one deterministic, traceable forecast day for a quantile model."""
    import matplotlib.pyplot as plt

    q_columns = quantile_columns(protocol_config)
    quantile = predictions.loc[predictions.prediction_type == "quantile"].copy()
    if quantile.empty:
        raise ValueError("prediction source contains no quantile model")
    quantile["_day"] = pd.to_datetime(
        quantile.target_time_utc, utc=True).dt.floor("D")
    key_counts = quantile.groupby(
        ["model_id", "location_id", "lead_time", "_day"]).size()
    model_id, location_id, lead_time, day = key_counts.sort_values(
        ascending=False, kind="stable").index[0]
    case = quantile.loc[
        (quantile.model_id == model_id)
        & (quantile.location_id == location_id)
        & (quantile.lead_time == lead_time)
        & (quantile._day == day)].sort_values("target_time_utc")
    time = pd.to_datetime(case.target_time_utc, utc=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    interval_label = f"{q_columns[0]}–{q_columns[-1]}"
    ax.fill_between(time, case[q_columns[0]], case[q_columns[-1]],
                    color=PALETTE["signal"], alpha=.2, label=interval_label)
    ax.plot(time, case[q_columns[len(q_columns) // 2]],
            color=PALETTE["method"], lw=1.2, label="median")
    ax.plot(time, case.y, color=PALETTE["dark"], lw=1, label="observed")
    ax.set(ylabel="GHI (W m$^{-2}$)", xlabel="Target time (UTC)",
           title=f"Traceable forecast case: {model_id}, {location_id}, lead {lead_time:g} h")
    ax.legend(fontsize=6, ncol=3)
    contract = FigureContract(
        conclusion="One forecast case shows interval, median and observations",
        evidence=(q_columns[0], q_columns[-1], "y"), source_data=str(source),
        result_status=result_status,
        n_definition=f"all available rows on {pd.Timestamp(day).date().isoformat()}",
        statistical_definition="registered outer quantiles without post-hoc smoothing")
    paths = save_figure(fig, output / "prediction_case", contract)
    return {
        "figure_id": "prediction_case", "source_table": str(source),
        "generation_function": "nwp.visualization.figures.prediction_case_figure",
        "caption": "Traceable one-day quantile forecast case",
        "status": result_status, "files": [str(path) for path in paths],
    }


def gap_sensitivity_figure(
        tables: dict[int, pd.DataFrame], *, sources: dict[int, Path],
        output: Path, result_status: str, metric: str = "mean_pinball",
) -> dict[str, Any]:
    """Compare separately resolved gap runs; no gap is recomputed in plotting."""
    import matplotlib.pyplot as plt

    rows = []
    for gap, table in sorted(tables.items()):
        if {"model_id", metric} - set(table):
            raise ValueError(f"gap {gap} source lacks model_id/{metric}")
        rows.append(table[["model_id", metric]].assign(gap_days=int(gap)))
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for model_id, subset in data.groupby("model_id", sort=True):
        subset = subset.sort_values("gap_days")
        ax.plot(subset.gap_days, subset[metric], marker="o", ms=3,
                label=str(model_id))
    ax.set(xlabel="Purge gap (days)", ylabel=metric.replace("_", " "),
           title="Registered gap sensitivity")
    ax.legend(fontsize=6, ncol=2)
    source_text = ";".join(f"{gap}:{sources[gap]}" for gap in sorted(sources))
    contract = FigureContract(
        conclusion="Metric stability is compared across separately resolved gap runs",
        evidence=(metric,), source_data=source_text,
        result_status=result_status,
        n_definition="each point is one complete run's overview metric",
        statistical_definition="no pooling across gaps; each run follows its resolved config")
    paths = save_figure(fig, output / "gap_sensitivity", contract)
    return {
        "figure_id": "gap_sensitivity", "source_table": source_text,
        "generation_function": "nwp.visualization.figures.gap_sensitivity_figure",
        "caption": "Gap sensitivity across resolved runs", "status": result_status,
        "files": [str(path) for path in paths],
    }


def residual_relationship_figure(
        frame: pd.DataFrame, *, feature: str, prediction: str, source: Path,
        output: Path, result_status: str,
) -> dict[str, Any]:
    """Generic dense feature–residual diagnostic; never implies causality."""
    import matplotlib.pyplot as plt

    if {feature, prediction, "y"} - set(frame):
        raise ValueError("residual relationship source lacks requested columns")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    residual = pd.to_numeric(frame.y) - pd.to_numeric(frame[prediction])
    relationship(ax, frame[feature], residual, xlabel=feature,
                 ylabel=f"Observed - {prediction}")
    ax.axhline(0, color=PALETTE["neutral"], lw=.8)
    contract = FigureContract(
        conclusion="Residual structure is diagnosed, not assigned a causal effect",
        evidence=(feature, prediction, "y"), source_data=str(source),
        result_status=result_status,
        n_definition="all finite joined source rows",
        statistical_definition="hexbin/scatter of observed minus prediction")
    stem = f"residual_by_{feature}"
    paths = save_figure(fig, output / stem, contract)
    return {
        "figure_id": stem, "source_table": str(source),
        "generation_function": "nwp.visualization.figures.residual_relationship_figure",
        "caption": f"Residual diagnostic by {feature}", "status": result_status,
        "files": [str(path) for path in paths],
    }


def group_effect_figure(
        ablation: pd.DataFrame, permutation: pd.DataFrame, *,
        ablation_source: Path, permutation_source: Path,
        output: Path, result_status: str) -> dict[str, Any]:
    """Plot downstream group evidence without treating it as feature selection."""
    import matplotlib.pyplot as plt

    required = {"group", "delta_vs_full"}
    if required - set(ablation) or required - set(permutation):
        raise ValueError("group evidence tables lack group/delta_vs_full")
    left = ablation.groupby("group").delta_vs_full.mean().rename("ablation")
    right = permutation.groupby("group").delta_vs_full.mean().rename("permutation")
    table = pd.concat([left, right], axis=1).sort_values("ablation")
    if table.empty:
        raise ValueError("group evidence is empty")
    positions = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(7.2, max(2.8, .4 * len(table))))
    ax.barh(positions - .17, table.ablation, height=.32,
            color=PALETTE["method"], label="group ablation")
    ax.barh(positions + .17, table.permutation, height=.32,
            color=PALETTE["signal"], label="grouped permutation")
    ax.set_yticks(positions, labels=table.index)
    ax.axvline(0, color=PALETTE["neutral"], lw=.8)
    ax.set(xlabel="Mean pinball increase versus full model",
           ylabel="Feature group", title="Development-only group evidence")
    ax.legend(fontsize=6)
    source = f"{ablation_source};{permutation_source}"
    contract = FigureContract(
        conclusion="Group evidence diagnoses predictive dependence only",
        evidence=("delta_vs_full",), source_data=source,
        result_status=result_status,
        n_definition="means across declared development inner folds",
        statistical_definition="refit ablation and within-lead grouped permutation")
    paths = save_figure(fig, output / "group_mechanism", contract)
    return {
        "figure_id": "group_mechanism", "source_table": source,
        "generation_function": "nwp.visualization.figures.group_effect_figure",
        "caption": "Development-only feature-group mechanism evidence",
        "status": result_status, "files": [str(path) for path in paths],
    }


def generate_run_figures(metrics_index: Path, analysis_index: Path,
                         output: Path, *, result_status: str,
                         protocol_config: dict[str, Any] | None = None,
                         prediction_index: Path | None = None) -> list[dict[str, Any]]:
    """Render only from run-owned source tables; never copy CSV into figures."""
    apply_publication_style()
    metrics = json.loads(metrics_index.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_index.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []
    probability_path = Path(metrics["overview"]["probability_primary"])
    point_path = Path(metrics["overview"]["point_secondary"])
    reliability_path = Path(metrics["overview"]["reliability"])
    probability, point = pd.read_csv(probability_path), pd.read_csv(point_path)
    reliability = pd.read_csv(reliability_path)
    if not probability.empty:
        figures.append(_bar_figure(
            probability, value="mean_pinball", ylabel="Mean pinball loss",
            title="Probability-model performance", source=probability_path,
            output=output, figure_id="probability_ranking",
            result_status=result_status))
    if not point.empty:
        figures.append(_bar_figure(
            point, value="rmse", ylabel="RMSE (W m$^{-2}$)",
            title="Auxiliary point-forecast performance", source=point_path,
            output=output, figure_id="point_performance",
            result_status=result_status))
    if not reliability.empty:
        figures.append(_reliability_figure(
            reliability, source=reliability_path, output=output,
            result_status=result_status))
    for dimension, tables in sorted(metrics.get("grouped", {}).items()):
        probability_group = Path(tables["probability_primary"])
        point_group = Path(tables["point_secondary"])
        q_table, p_table = pd.read_csv(probability_group), pd.read_csv(point_group)
        if not q_table.empty and dimension in {"lead_time", "outer_fold", "season"}:
            figures.append(grouped_metric_figure(
                q_table, dimension=dimension, metric="mean_pinball",
                source=probability_group, output=output,
                result_status=result_status))
        if (not p_table.empty and dimension in {"lead_time", "outer_fold"}
                and "rmse" in p_table):
            figures.append(grouped_metric_figure(
                p_table, dimension=dimension, metric="rmse", source=point_group,
                output=output, result_status=result_status))
    summary = Path(analysis["model_summary"])
    if not summary.is_file():
        raise ValueError("analysis index points to a missing model summary")
    tuning_path = analysis.get("tuning_trials")
    if tuning_path and Path(tuning_path).is_file():
        trials = pd.read_csv(tuning_path)
        if not trials.empty:
            figures.append(tuning_trials_figure(
                trials, source=Path(tuning_path), output=output,
                result_status=result_status))
    ablation_path = analysis.get("group_ablation")
    permutation_path = analysis.get("grouped_permutation")
    if (ablation_path and permutation_path and Path(ablation_path).is_file()
            and Path(permutation_path).is_file()):
        figures.append(group_effect_figure(
            pd.read_csv(ablation_path), pd.read_csv(permutation_path),
            ablation_source=Path(ablation_path),
            permutation_source=Path(permutation_path), output=output,
            result_status=result_status))
    if prediction_index is not None and protocol_config is not None:
        payload = json.loads(prediction_index.read_text(encoding="utf-8"))
        paths = [Path(item["path"])
                 for item in payload.get("outer_predictions", [])]
        if paths:
            predictions = pd.concat([pd.read_parquet(path) for path in paths],
                                    ignore_index=True)
            figures.append(prediction_case_figure(
                predictions, protocol_config=protocol_config,
                source=prediction_index, output=output,
                result_status=result_status))
    return figures
