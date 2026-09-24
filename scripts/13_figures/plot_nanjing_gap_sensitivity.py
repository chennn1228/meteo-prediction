"""Plot preregistered 7/10/14-day Nanjing nested-gap sensitivity.

Each gap must have its own complete six-candidate inner ledgers, fixed models,
selected-candidate outer refits, and identical five outer scoring samples.
The 10-day result remains the main protocol; 7/14 are diagnostics only.
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

ALL_MODELS = ("climatology", "persistence", "smart_persistence", "optimal_convex",
              "raw_gfs", "bias_correction", "linear_mos", "ridge_mos", "lgbm", "xgboost")
QUANTILE = ("ridge_mos", "lgbm", "xgboost")
POINT_DISPLAY = ("raw_gfs", "linear_mos", "ridge_mos", "lgbm", "xgboost")
COLORS = {"raw_gfs": "#8C959A", "linear_mos": "#357E84", "ridge_mos": "#8798A8",
          "lgbm": "#245680", "xgboost": "#D4863B"}
SCOPE = "Nanjing only | 7/14-day diagnostic; 10-day main | not official 20-site evidence"
TAUS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
QCOLS = [f"q{tau:.2f}" for tau in TAUS]


def save(fig, directory: Path, name: str) -> list[str]:
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    paths = []
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def load_gap(root: Path, gap: int) -> pd.DataFrame:
    suffix = "" if gap == 10 else f"_gap{gap}"
    fixed = pd.read_parquet(root / f"outer_fixed{suffix}_predictions.parquet")
    tuned = pd.read_parquet(root / f"outer_tuned{suffix}_predictions.parquet")
    summary = json.loads((root / f"outer_all_cpu{suffix}_summary.json").read_text(encoding="utf-8"))
    if (summary.get("official_eligible") is not False
            or int(summary.get("gap_days", 10)) != gap):
        raise ValueError(f"gap {gap}: missing complete diagnostic summary")
    frame = pd.concat([fixed, tuned], ignore_index=True)
    if (set(frame.model_id) != set(ALL_MODELS)
            or set(frame.result_status) != {"diagnostic"}
            or len(set(frame.location_id)) != 1):
        raise ValueError(f"gap {gap}: incomplete model or site set")
    return frame


def paired_samples(frame: pd.DataFrame, reference: pd.DataFrame | None) -> pd.DataFrame:
    keys = ["location_id", "target_time_utc", "forecast_issue_time_utc",
            "lead_time", "outer_fold"]
    target = None
    for model in ALL_MODELS:
        part = frame.loc[frame.model_id == model]
        if len(part) != 9564 or part.duplicated(keys).any():
            raise ValueError(f"{model}: missing/duplicate outer-score samples")
        sample = part.sort_values(keys)[keys + ["y"]].reset_index(drop=True)
        if target is not None and (not sample[keys].equals(target[keys])
                                   or not np.allclose(sample.y, target.y, atol=1e-9, rtol=0)):
            raise ValueError("models do not share matched scoring samples")
        target = sample
    if reference is not None and (not target[keys].equals(reference[keys])
                                  or not np.allclose(target.y, reference.y, atol=1e-9, rtol=0)):
        raise ValueError("gap variants changed outer scoring samples")
    return target


def scores(frame: pd.DataFrame, gap: int) -> pd.DataFrame:
    rows = []
    for model in ALL_MODELS:
        for fold, part in [("pooled", frame.loc[frame.model_id == model]), *list(
                frame.loc[frame.model_id == model].groupby("outer_fold"))]:
            y, point = part.y.to_numpy(dtype=float), part.point_prediction.to_numpy(dtype=float)
            item = {"gap_days": gap, "model_id": model, "outer_fold": fold, "n": len(part),
                    "rmse": float(np.sqrt(np.mean((y - point) ** 2))),
                    "mean_pinball": np.nan, "crossing_rate": np.nan}
            if model in QUANTILE:
                q = part[QCOLS].to_numpy(dtype=float)
                errors = y[:, None] - q
                item["mean_pinball"] = float(np.maximum(TAUS * errors, (TAUS - 1) * errors).mean())
                item["crossing_rate"] = float(np.mean(np.any(np.diff(q, axis=1) < 0, axis=1)))
            rows.append(item)
    return pd.DataFrame(rows)


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    root = ROOT / manifest["data_layout"]["root"] / "06_cpu_single_site"
    reference, tables, choices = None, [], []
    for gap in (7, 10, 14):
        frame = load_gap(root, gap)
        reference = paired_samples(frame, reference)
        tables.append(scores(frame, gap))
        folder = root / ("outer_tuned_parts" if gap == 10 else f"outer_tuned_gap{gap}_parts")
        for outer_index in range(1, 6):
            for model in QUANTILE:
                receipt = json.loads((folder / f"outer_{outer_index}_{model}.json").read_text(encoding="utf-8"))
                if int(receipt.get("gap_days", 10)) != gap:
                    raise ValueError("candidate receipt gap mismatch")
                choices.append({"gap_days": gap, "outer_fold": f"outer_{outer_index}",
                                "model_id": model, "selected_candidate": receipt["selected_candidate"]})
    data = pd.concat(tables, ignore_index=True)
    output = ROOT / "figs" / "nanjing_15var_diagnostic" / "07_robustness"
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "fig_gap_sensitivity_source.csv", index=False)
    pd.DataFrame(choices).to_csv(output / "fig_gap_sensitivity_candidates.csv", index=False)
    pooled = data.loc[data.outer_fold == "pooled"]
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 79 / 25.4))
    for model in QUANTILE:
        part = pooled.loc[pooled.model_id == model].sort_values("gap_days")
        axes[0].plot(part.gap_days, part.mean_pinball, marker="o", ms=3.5,
                     color=COLORS[model], lw=1.2, label=model)
    for model in POINT_DISPLAY:
        part = pooled.loc[pooled.model_id == model].sort_values("gap_days")
        axes[1].plot(part.gap_days, part.rmse, marker="o", ms=3.5,
                     color=COLORS[model], lw=1.2, label=model)
    axes[0].set(title="Probabilistic primary score", ylabel="Mean pinball (W m$^{-2}$)")
    axes[1].set(title="Auxiliary point score", ylabel="RMSE (W m$^{-2}$)")
    for ax in axes:
        ax.axvline(10, color="#AEB9C1", lw=0.8, ls="--")
        ax.set(xlabel="Purge between blocks (days)", xticks=[7, 10, 14])
        ax.grid(axis="y", color="#D7E2E8", lw=0.5)
        ax.legend(fontsize=5.5, loc="best")
    fig.suptitle("Nested gap sensitivity on identical five-outer-fold score hours",
                 x=0.04, ha="left", y=1.02)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.19, top=0.78, wspace=0.36)
    figure = {"files": save(fig, output, "fig_gap_sensitivity"),
              "source_data": ["fig_gap_sensitivity_source.csv", "fig_gap_sensitivity_candidates.csv"],
              "n_definition": "9,564 paired Nanjing outer-score rows per model and gap; same target hours at 7/10/14 days",
              "protocol": "six candidates × three inner folds retuned per model, outer fold, and gap; then full outer refit",
              "caution": "7/14-day gaps are preregistered sensitivity only; 10 days remains main protocol"}
    path = output.parent / "figure_manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["figures"]["fig_gap_sensitivity"] = figure
    metadata["figure_count"] = len(metadata["figures"])
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": "fig_gap_sensitivity", "total_figures": metadata["figure_count"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
