"""Four source-backed probability figures for the Nanjing diagnostic test year.

The same original final-test window is used throughout. Missing truth is
retained in prediction artifacts but excluded from score numerators and
denominators. No figure is official 20-site evidence.
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

MODELS = ("ridge_mos", "lgbm", "xgboost")
COLORS = {"ridge_mos": "#8798A8", "lgbm": "#245680", "xgboost": "#D4863B"}
LABELS = {"ridge_mos": "Ridge MOS", "lgbm": "LightGBM", "xgboost": "XGBoost"}
LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
SCOPE = "Nanjing only | 15-variable diagnostic | not official 20-site evidence"


def save_figure(fig, output: Path, name: str) -> list[str]:
    files = []
    for extension in ("svg", "pdf", "png"):
        destination = output / f"{name}.{extension}"
        fig.savefig(destination, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(str(destination.relative_to(ROOT)))
    plt.close(fig)
    return files


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    root = ROOT / manifest["data_layout"]["root"]
    raw = pd.read_parquet(root / "06_cpu_single_site" / "final_test_uncalibrated.parquet")
    calibrated = pd.read_parquet(root / "06_cpu_single_site" / "final_test_calibrated.parquet")
    features = pd.read_parquet(root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet")
    for label, table in (("raw", raw), ("calibrated", calibrated)):
        if (set(table.result_status) != {"diagnostic"}
                or len(set(table.location_id)) != 1
                or set(table.outer_fold) != {"final_test"}):
            raise ValueError(f"{label}: only diagnostic Nanjing final-test rows allowed")
    if set(calibrated.model_id) != set(MODELS):
        raise ValueError("calibrated table must contain exactly three quantile models")
    keys = ["model_id", "target_time_utc", "forecast_issue_time_utc", "lead_time"]
    if calibrated.duplicated(keys).any() or raw.duplicated(keys).any():
        raise ValueError("duplicate final-test forecast samples")
    if any(raw.loc[raw.model_id == model].shape[0] != calibrated.loc[
            calibrated.model_id == model].shape[0] for model in MODELS):
        raise ValueError("calibration changed test sample counts")
    return raw, calibrated, features


def figure_case(raw: pd.DataFrame, calibrated: pd.DataFrame, output: Path) -> dict:
    reference = raw.loc[(raw.model_id == "raw_gfs") & (raw.lead_time == 24)
                        & np.isfinite(raw.y), ["target_time_utc", "y", "point_prediction"]].copy()
    reference["local_day"] = pd.to_datetime(reference.target_time_utc, utc=True).dt.tz_convert(
        "Asia/Shanghai").dt.date
    daily = reference.groupby("local_day").agg(
        n=("y", "size"), rmse=("y", lambda _: np.nan))
    daily["rmse"] = reference.groupby("local_day").apply(
        lambda frame: float(np.sqrt(np.mean((frame.y - frame.point_prediction) ** 2))),
        include_groups=False)
    eligible = daily.loc[daily.n >= 8].copy()
    if eligible.empty:
        raise ValueError("no day with at least eight observed daylight D+1 hours")
    median_rmse = float(eligible.rmse.median())
    chosen_day = min(eligible.index, key=lambda day: (abs(float(eligible.loc[day, "rmse"]) - median_rmse), day))
    case = reference.loc[reference.local_day == chosen_day].copy()
    q = calibrated.loc[(calibrated.model_id == "xgboost") & (calibrated.lead_time == 24)].copy()
    q = q.merge(case[["target_time_utc"]], on="target_time_utc", how="inner", validate="one_to_one")
    if len(q) != len(case):
        raise ValueError("selected case hours have missing quantile predictions")
    case = case.merge(q[["target_time_utc", *[f"q{tau:.2f}" for tau in LEVELS]]],
                      on="target_time_utc", validate="one_to_one")
    case["local_hour"] = pd.to_datetime(case.target_time_utc, utc=True).dt.tz_convert(
        "Asia/Shanghai").dt.hour
    case = case.sort_values("local_hour")
    case.to_csv(output / "fig_quantile_case_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 78 / 25.4))
    x = case.local_hour.to_numpy(dtype=float)
    ax.fill_between(x, case["q0.05"], case["q0.95"], color=COLORS["xgboost"],
                    alpha=0.16, label="90% interval")
    ax.fill_between(x, case["q0.25"], case["q0.75"], color=COLORS["xgboost"],
                    alpha=0.35, label="50% interval")
    ax.plot(x, case["q0.50"], color=COLORS["xgboost"], lw=1.4, label="XGBoost median")
    ax.plot(x, case.y, color="#27343D", lw=1.6, marker="o", ms=2.5, label="Observed GHI")
    ax.plot(x, case.point_prediction, color=COLORS["lgbm"], lw=1,
            ls="--", label="Raw GFS")
    ax.set(xlabel="Local hour (China Standard Time)", ylabel="GHI (W m$^{-2}$)")
    ax.set_xticks(x)
    ax.set_title(f"D+1 example: {chosen_day} (median daily raw-GFS RMSE rule)", loc="left")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6)
    ax.grid(axis="y", color="#D7E2E8", lw=0.5)
    fig.text(0.99, 0.98, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.28)
    return {"files": save_figure(fig, output, "fig_quantile_case"),
            "source_data": "fig_quantile_case_source.csv",
            "n_definition": f"{len(case)} observed D+1 daylight hours on selected local day",
            "case_selection_rule": "eligible days >=8 observed D+1 hours; closest daily raw-GFS RMSE to median, earlier date breaks tie"}


def figure_reliability(raw: pd.DataFrame, calibrated: pd.DataFrame, output: Path) -> dict:
    records = []
    for stage, table in (("uncalibrated", raw), ("causally calibrated", calibrated)):
        for model in MODELS:
            subset = table.loc[(table.model_id == model) & np.isfinite(table.y)]
            if subset.empty:
                raise ValueError(f"{stage}/{model}: no observed final-test rows")
            for tau in LEVELS:
                column = f"q{tau:.2f}"
                records.append({"stage": stage, "model_id": model, "nominal_quantile": tau,
                                "empirical_cdf": float(np.mean(subset.y.to_numpy() <= subset[column].to_numpy())),
                                "n": len(subset)})
    data = pd.DataFrame(records)
    data.to_csv(output / "fig_coverage_reliability_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4), sharex=True, sharey=True)
    for ax, stage in zip(axes, ("uncalibrated", "causally calibrated")):
        ax.plot([0, 1], [0, 1], color="#B5BEC5", ls="--", lw=1, label="Ideal")
        for model in MODELS:
            part = data.loc[(data.stage == stage) & (data.model_id == model)]
            ax.plot(part.nominal_quantile, part.empirical_cdf, marker="o", ms=3,
                    lw=1.3, color=COLORS[model], label=LABELS[model])
        ax.set(title=stage.capitalize(), xlabel="Nominal quantile")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color="#D7E2E8", lw=0.5)
    axes[0].set_ylabel("Empirical P(observation ≤ predicted quantile)")
    axes[1].legend(loc="lower right", fontsize=6)
    fig.suptitle("Seven-quantile reliability on the held-out test year", x=0.05,
                 ha="left", y=1.02)
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.17, top=0.80, wspace=0.14)
    return {"files": save_figure(fig, output, "fig_coverage_reliability"),
            "source_data": "fig_coverage_reliability_source.csv",
            "n_definition": "observed Nanjing test site×target-hour×lead rows per model; repeated leads are correlated"}


def figure_width(calibrated: pd.DataFrame, output: Path) -> dict:
    intervals = ((50, "q0.25", "q0.75"), (80, "q0.10", "q0.90"),
                 (90, "q0.05", "q0.95"))
    records = []
    for model in MODELS:
        part = calibrated.loc[(calibrated.model_id == model) & np.isfinite(calibrated.y)]
        for nominal, lower, upper in intervals:
            lo, hi, y = (part[column].to_numpy(dtype=float) for column in (lower, upper, "y"))
            records.append({"model_id": model, "nominal_coverage": nominal,
                            "empirical_coverage": float(np.mean((y >= lo) & (y <= hi)) * 100),
                            "mean_width_w_m2": float(np.mean(hi - lo)), "n": len(part)})
    data = pd.DataFrame(records)
    data.to_csv(output / "fig_coverage_width_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4))
    axes[0].plot([45, 95], [45, 95], color="#B5BEC5", ls="--", lw=1, label="Nominal")
    for model in MODELS:
        part = data.loc[data.model_id == model].sort_values("nominal_coverage")
        axes[0].plot(part.nominal_coverage, part.empirical_coverage, color=COLORS[model],
                     marker="o", ms=3.5, lw=1.3, label=LABELS[model])
        axes[1].plot(part.nominal_coverage, part.mean_width_w_m2, color=COLORS[model],
                     marker="o", ms=3.5, lw=1.3)
    axes[0].set(ylabel="Empirical coverage (%)", ylim=(0, 100), title="Coverage")
    axes[1].set(ylabel="Mean interval width (W m$^{-2}$)", title="Width")
    for ax in axes:
        ax.set(xlabel="Nominal interval (%)", xticks=[50, 80, 90])
        ax.grid(color="#D7E2E8", lw=0.5)
    axes[0].legend(loc="upper left", fontsize=6)
    fig.suptitle("Coverage must be interpreted together with interval width",
                 x=0.05, ha="left", y=1.02)
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.80, wspace=0.38)
    return {"files": save_figure(fig, output, "fig_coverage_width"),
            "source_data": "fig_coverage_width_source.csv",
            "n_definition": "observed Nanjing test site×target-hour×lead rows per model and interval"}


def figure_weather(calibrated: pd.DataFrame, features: pd.DataFrame, output: Path) -> dict:
    keys = ["target_time_utc", "forecast_issue_time_utc", "lead_time"]
    covariates = features.loc[:, [*keys, "cloud_cover_fcst"]].copy()
    if covariates.duplicated(keys).any():
        raise ValueError("forecast cloud join is not one-to-one")
    merged = calibrated.merge(covariates, on=keys, how="left", validate="many_to_one",
                              indicator=True)
    if (merged._merge != "both").any():
        raise ValueError("test forecast absent from immutable feature source")
    cloud = pd.to_numeric(merged.cloud_cover_fcst, errors="coerce")
    if ((cloud.notna()) & ((cloud < 0) | (cloud > 100))).any():
        raise ValueError("forecast total cloud outside 0–100%")
    merged["cloud_stratum"] = np.select(
        [cloud < 30, (cloud >= 30) & (cloud < 70), cloud >= 70],
        ["<30%", "30–<70%", "≥70%"], default="missing forecast cloud")
    merged = merged.loc[np.isfinite(merged.y)].copy()
    merged["covered_80"] = ((merged.y >= merged["q0.10"]) &
                             (merged.y <= merged["q0.90"])).astype(float)
    order = ("<30%", "30–<70%", "≥70%")
    data = merged.groupby(["lead_time", "cloud_stratum", "model_id"], as_index=False).agg(
        n=("covered_80", "size"), coverage_80=("covered_80", "mean"))
    data.to_csv(output / "fig_weather_conditional_coverage_source.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 83 / 25.4), sharey=True)
    for ax, lead in zip(axes, (24, 48, 72)):
        lead_data = data.loc[data.lead_time == lead]
        categories = [name for name in order if name in set(lead_data.cloud_stratum)]
        x = np.arange(len(categories), dtype=float)
        for i, model in enumerate(MODELS):
            part = lead_data.loc[lead_data.model_id == model].set_index("cloud_stratum")
            values = np.array([float(part.loc[name, "coverage_80"]) * 100 for name in categories])
            ax.bar(x + (i - 1) * 0.23, values, width=0.22, color=COLORS[model],
                   label=LABELS[model])
        ax.axhline(80, color="#657783", ls="--", lw=0.8)
        counts = lead_data.loc[lead_data.model_id == MODELS[0]].set_index("cloud_stratum")
        ax.set_xticks(x, [f"{name}\n(n={int(counts.loc[name, 'n']):,})" for name in categories],
                      fontsize=5.5)
        ax.set(title=f"D+{lead // 24}", ylim=(0, 105))
        ax.grid(axis="y", color="#D7E2E8", lw=0.5)
    axes[0].set_ylabel("80% interval empirical coverage (%)")
    axes[2].legend(loc="upper right", fontsize=6)
    fig.suptitle("Coverage by issue-time forecast total cloud", x=0.05, ha="left", y=1.02)
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    missing = data.loc[(data.model_id == MODELS[0]) &
                       (data.cloud_stratum == "missing forecast cloud")]
    if not missing.empty:
        detail = ", ".join(f"D+{int(row.lead_time) // 24} n={int(row.n)}"
                           for row in missing.itertuples())
        fig.text(0.5, 0.05, f"Missing forecast cloud excluded from bars ({detail}); "
                 "all values retained in source data", ha="center", fontsize=6,
                 color="#657783")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.78, wspace=0.12)
    return {"files": save_figure(fig, output, "fig_weather_conditional_coverage"),
            "source_data": "fig_weather_conditional_coverage_source.csv",
            "n_definition": "observed test rows per lead and forecast-cloud stratum; model counts match",
            "strata": "<30%, 30–<70%, ≥70%, and explicit missing forecast cloud"}


def main() -> None:
    apply_publication_style()
    raw, calibrated, features = load_data()
    output = ROOT / "figs" / "nanjing_15var_diagnostic" / "04_probability"
    output.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig_quantile_case": figure_case(raw, calibrated, output),
        "fig_coverage_reliability": figure_reliability(raw, calibrated, output),
        "fig_coverage_width": figure_width(calibrated, output),
        "fig_weather_conditional_coverage": figure_weather(calibrated, features, output),
    }
    path = output.parent / "figure_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["figures"].update(figures)
    manifest["figure_count"] = len(manifest["figures"])
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"new_figures": list(figures), "total_figures": manifest["figure_count"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
