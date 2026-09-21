"""Upgrade Nanjing outer-fold benchmark figures after all CPU refits finish.

Ten models must share exactly the same five-outer-fold forecast samples. The
three quantile models retain their observed pre-calibration crossing rates.
All outputs are diagnostic one-site evidence, never official 20-site results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s13_visualization.s01_style.publication import apply_publication_style  # noqa: E402

ORDER = ("climatology", "persistence", "smart_persistence", "optimal_convex",
         "raw_gfs", "bias_correction", "linear_mos", "ridge_mos", "lgbm", "xgboost")
QUANTILE = ("ridge_mos", "lgbm", "xgboost")
BLUE, AMBER, GRAY, RED, PALE = "#245680", "#D4863B", "#8C959A", "#B6534B", "#D7E2E8"
SCOPE = "Nanjing only | 15-variable diagnostic | not official 20-site evidence"


def save(fig, directory: Path, name: str) -> list[str]:
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color=GRAY)
    files = []
    for extension in ("svg", "pdf", "png"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def load_predictions() -> pd.DataFrame:
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    source = ROOT / manifest["data_layout"]["root"] / "06_cpu_single_site"
    fixed = pd.read_parquet(source / "outer_fixed_predictions.parquet")
    tuned = pd.read_parquet(source / "outer_tuned_predictions.parquet")
    frame = pd.concat([fixed, tuned], ignore_index=True)
    if (set(frame.model_id) != set(ORDER) or set(frame.result_status) != {"diagnostic"}
            or set(frame.outer_fold) != {f"outer_{i}" for i in range(1, 6)}
            or len(set(frame.location_id)) != 1):
        raise ValueError("all ten diagnostic CPU models and five outer folds required")
    sample = ["location_id", "target_time_utc", "forecast_issue_time_utc",
              "lead_time", "outer_fold"]
    reference = None
    for model in ORDER:
        part = frame.loc[frame.model_id == model]
        if len(part) != 9564 or part.duplicated(sample).any():
            raise ValueError(f"{model}: sample count/uniqueness differs")
        present = part.sort_values(sample)[sample + ["y"]].reset_index(drop=True)
        if reference is not None and (not present[sample].equals(reference[sample])
                                      or not np.allclose(present.y, reference.y, rtol=0, atol=1e-9)):
            raise ValueError(f"{model}: outer scoring rows not paired with reference")
        reference = present
    return frame


def rmse(frame: pd.DataFrame) -> float:
    delta = frame.y.to_numpy(dtype=float) - frame.point_prediction.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(delta ** 2)))


def tables(frame: pd.DataFrame, directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([
        {"model_id": model, "n": len(part), "rmse": rmse(part)}
        for model, part in frame.groupby("model_id")])
    by_fold = pd.DataFrame([
        {"model_id": model, "outer_fold": fold, "n": len(part), "rmse": rmse(part)}
        for (model, fold), part in frame.groupby(["model_id", "outer_fold"])])
    by_lead = pd.DataFrame([
        {"model_id": model, "lead_time": int(lead), "n": len(part), "rmse": rmse(part)}
        for (model, lead), part in frame.groupby(["model_id", "lead_time"])])
    overall.to_csv(directory / "all_cpu_rmse_overall.csv", index=False)
    by_fold.to_csv(directory / "all_cpu_rmse_by_fold.csv", index=False)
    by_lead.to_csv(directory / "all_cpu_rmse_by_lead.csv", index=False)
    return overall, by_fold, by_lead


def value_ladder(overall: pd.DataFrame, by_fold: pd.DataFrame, directory: Path) -> dict:
    lookup = overall.set_index("model_id")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 99 / 25.4))
    y = np.arange(len(ORDER))
    colors = [GRAY] * 4 + [AMBER] + [BLUE] * 5
    ax.barh(y, [lookup.loc[name, "rmse"] for name in ORDER], color=colors, height=0.55)
    for index, name in enumerate(ORDER):
        values = by_fold.loc[by_fold.model_id == name, "rmse"].to_numpy(dtype=float)
        ax.scatter(values, np.full(len(values), index), s=8, facecolor="white",
                   edgecolor="#263942", linewidth=0.5, zorder=3)
    ax.axvline(float(lookup.loc["raw_gfs", "rmse"]), color=AMBER, lw=0.8, ls="--")
    ax.set_yticks(y, ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (W m$^{-2}$), pooled original five outer validations")
    ax.set_title("Ten-model CPU value ladder; dots are fold scores, not confidence intervals",
                 loc="left", pad=12)
    ax.grid(axis="x", color=PALE, lw=0.5)
    fig.tight_layout()
    return {"files": save(fig, directory, "fig_value_ladder"),
            "source_data": ["all_cpu_rmse_overall.csv", "all_cpu_rmse_by_fold.csv"],
            "n_definition": "9,564 identical Nanjing site×target-hour×lead outer-score rows per model"}


def lead_performance(by_lead: pd.DataFrame, directory: Path) -> dict:
    by_lead.to_csv(directory / "fig_cpu_lead_performance_source.csv", index=False)
    matrix = by_lead.pivot(index="model_id", columns="lead_time", values="rmse").loc[
        list(ORDER), [24, 48, 72]]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 101 / 25.4))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto", vmin=100, vmax=300)
    ax.set_xticks([0, 1, 2], ["D+1", "D+2", "D+3"])
    ax.set_yticks(np.arange(len(ORDER)), ORDER)
    for i in range(len(ORDER)):
        for j in range(3):
            value = float(matrix.iloc[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 225 else "#20303B", fontsize=7)
    fig.colorbar(image, ax=ax, label="RMSE (W m$^{-2}$)", fraction=0.04, pad=0.03)
    ax.set_title("Lead-wise RMSE on paired Nanjing outer-score samples", loc="left", pad=12)
    fig.tight_layout()
    return {"files": save(fig, directory, "fig_cpu_lead_performance"),
            "source_data": "fig_cpu_lead_performance_source.csv",
            "n_definition": "3,188 identical outer-score site×target-hour rows per model per lead"}


def improvement(by_fold: pd.DataFrame, directory: Path) -> dict:
    raw = by_fold.loc[by_fold.model_id == "raw_gfs", ["outer_fold", "rmse"]].rename(
        columns={"rmse": "raw_rmse"})
    values = by_fold.merge(raw, on="outer_fold", validate="many_to_one")
    values = values.loc[values.model_id != "raw_gfs"].copy()
    values["rmse_improvement_pct"] = 100 * (1 - values.rmse / values.raw_rmse)
    values.to_csv(directory / "fig_cpu_improvement_vs_raw_source.csv", index=False)
    order = [model for model in ORDER if model != "raw_gfs"]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 95 / 25.4))
    for index, model in enumerate(order):
        sample = values.loc[values.model_id == model, "rmse_improvement_pct"].to_numpy()
        center = float(sample.mean())
        ax.plot([sample.min(), sample.max()], [index, index], color=GRAY, lw=1)
        ax.scatter(sample, np.full(len(sample), index), color=GRAY, s=9, zorder=3)
        ax.scatter([center], [index], color=BLUE if center >= 0 else RED,
                   s=30, marker="D", zorder=4)
    ax.axvline(0, color=AMBER, lw=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE improvement vs matched raw GFS (%)")
    ax.set_title("Five-fold improvement spread; diamond = arithmetic fold mean",
                 loc="left", pad=12)
    ax.grid(axis="x", color=PALE, lw=0.5)
    fig.tight_layout()
    return {"files": save(fig, directory, "fig_cpu_improvement_vs_raw"),
            "source_data": "fig_cpu_improvement_vs_raw_source.csv",
            "n_definition": "five chronological outer-fold contrasts per model; no confidence interval claimed"}


def pinball(frame: pd.DataFrame, directory: Path) -> dict:
    rows = []
    quantile_columns = ["q0.05", "q0.10", "q0.25", "q0.50", "q0.75", "q0.90", "q0.95"]
    taus = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    for model in QUANTILE:
        for fold, part in [("pooled", frame.loc[frame.model_id == model]), *list(
                frame.loc[frame.model_id == model].groupby("outer_fold"))]:
            y = part.y.to_numpy(dtype=float)[:, None]
            q = part[quantile_columns].to_numpy(dtype=float)
            errors = y - q
            loss = np.maximum(taus * errors, (taus - 1) * errors)
            rows.append({"model_id": model, "outer_fold": fold, "n": len(part),
                         "mean_pinball": float(loss.mean()),
                         "crossing_rate": float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))})
    data = pd.DataFrame(rows)
    data.to_csv(directory / "fig_cpu_overall_pinball_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 74 / 25.4))
    for index, model in enumerate(QUANTILE):
        subset = data.loc[data.model_id == model]
        pooled = subset.loc[subset.outer_fold == "pooled"].iloc[0]
        folds = subset.loc[subset.outer_fold != "pooled"].mean_pinball.to_numpy()
        ax.plot([folds.min(), folds.max()], [index, index], color=GRAY, lw=1)
        ax.scatter(folds, np.full(len(folds), index), color=GRAY, s=13, zorder=3)
        ax.scatter([pooled.mean_pinball], [index], color=BLUE if model != "xgboost" else AMBER,
                   marker="D", s=40, zorder=4)
        ax.text(float(max(folds.max(), pooled.mean_pinball)) + 0.4, index,
                f"crossing {100 * pooled.crossing_rate:.1f}%", va="center", fontsize=6)
    ax.set_yticks(np.arange(len(QUANTILE)), QUANTILE)
    ax.invert_yaxis()
    ax.set_xlabel("Mean pinball (W m$^{-2}$); lower is better")
    ax.set_title("Quantile models only; uncalibrated five-fold outer validation",
                 loc="left", pad=12)
    ax.grid(axis="x", color=PALE, lw=0.5)
    ax.margins(x=0.22)
    fig.tight_layout()
    return {"files": save(fig, directory, "fig_cpu_overall_pinball"),
            "source_data": "fig_cpu_overall_pinball_source.csv",
            "n_definition": "9,564 identical outer-score rows per model; five fold dots and pooled diamond",
            "statistic": "mean of seven pinball losses; raw quantile crossing rate shown, not repaired"}


def main() -> None:
    apply_publication_style()
    frame = load_predictions()
    directory = ROOT / "figs" / "nanjing_15var_diagnostic" / "03_model_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    overall, by_fold, by_lead = tables(frame, directory)
    figures = {
        "fig_value_ladder": value_ladder(overall, by_fold, directory),
        "fig_cpu_overall_pinball": pinball(frame, directory),
        "fig_cpu_lead_performance": lead_performance(by_lead, directory),
        "fig_cpu_improvement_vs_raw": improvement(by_fold, directory),
    }
    path = directory.parent / "figure_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["figures"].update(figures)
    manifest["figure_count"] = len(manifest["figures"])
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": list(figures), "total_figures": manifest["figure_count"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
