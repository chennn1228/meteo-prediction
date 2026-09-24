"""Evidence-labelled single-site diagnostic figures from real local artifacts.

No generated figure is an official 20-site or spatial-generalization result.
Uses only Python/matplotlib and never calls a data provider.
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

BLUE = "#245680"
TEAL = "#357E84"
AMBER = "#D4863B"
RED = "#B6534B"
GRAY = "#8C959A"
PALE = "#D7E2E8"
SCOPE = "Nanjing only | 15-variable diagnostic | not official 20-site evidence"
FORECAST_FIELDS = (
    "ghi_fcst", "dhi_fcst", "dni_fcst", "gti_fcst", "cloud_cover_fcst",
    "temp_fcst", "rh_fcst", "dewpoint_fcst", "wind_speed_fcst", "wind_dir_fcst",
    "pressure_fcst", "precip_fcst", "sunshine_fcst", "terrestrial_fcst", "is_day_fcst",
)


def save(fig, folder: Path, stem: str) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color=GRAY)
    paths = []
    for extension in ("svg", "pdf", "png"):
        path = folder / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        paths.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return paths


def rmse(frame: pd.DataFrame) -> float:
    return float(np.sqrt(np.mean(np.square(frame.y.to_numpy(dtype=float)
                                           - frame.point_prediction.to_numpy(dtype=float)))))


def data_timeline(frame: pd.DataFrame, destination: Path) -> dict:
    frame = frame.copy()
    frame["month"] = pd.to_datetime(frame.target_time_utc, utc=True).dt.strftime("%Y-%m")
    day = frame.solar_elevation > 0
    counts = []
    for month, group in frame.loc[day].groupby("month", sort=True):
        hours = group.drop_duplicates("target_time_utc")
        forecast = group.loc[:, FORECAST_FIELDS].notna().all(axis=1)
        counts.append({"month": month,
                       "gfs_missing_pct": float(100 * (1 - forecast.mean())),
                       "himawari_missing_pct": float(100 * hours.ghi_obs_sat.isna().mean()),
                       "era5_missing_pct": float(100 * hours.ghi_obs_era5.isna().mean()),
                       "gfs_rows": len(group), "satellite_hours": len(hours),
                       "satellite_missing_hours": int(hours.ghi_obs_sat.isna().sum())})
    table = pd.DataFrame(counts)
    table.to_csv(destination / "fig_data_source_timeline_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(183 / 25.4, 78 / 25.4))
    x = np.arange(len(table))
    for label, column, color, marker in (
        ("GFS forecast rows", "gfs_missing_pct", BLUE, "o"),
        ("Himawari GHI hours", "himawari_missing_pct", AMBER, "s"),
        ("ERA5 GHI hours", "era5_missing_pct", TEAL, "^"),
    ):
        ax.plot(x, table[column], color=color, lw=1.2, marker=marker, ms=2.7, label=label)
    ax.set_xlim(-0.5, len(table) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xticks(x[::3], table.month.iloc[::3], rotation=45, ha="right")
    ax.set_ylabel("Missing daytime values (%)")
    ax.set_title("All 31 monthly payloads are present; value availability is not perfect", loc="left", pad=10)
    ax.legend(ncol=3, loc="upper left", fontsize=6)
    ax.grid(axis="y", color=PALE, lw=0.5)
    fig.tight_layout()
    return {"files": save(fig, destination, "fig_data_source_timeline"),
            "source_data": "fig_data_source_timeline_source.csv",
            "n_definition": "daytime forecast rows for GFS, unique UTC hours for truth sources"}


def hexbin_cloud_error(frame: pd.DataFrame, destination: Path) -> dict:
    selected = frame.loc[(frame.solar_elevation > 0)
                         & frame[["cloud_cover_fcst", "cloud_cover_obs", "ghi_fcst", "ghi_obs_sat"]]
                         .notna().all(axis=1),
                         ["lead_time", "cloud_cover_fcst", "cloud_cover_obs", "ghi_fcst", "ghi_obs_sat"]].copy()
    selected["cloud_error_pp"] = selected.cloud_cover_fcst - selected.cloud_cover_obs
    selected["ghi_error_wm2"] = selected.ghi_fcst - selected.ghi_obs_sat
    selected.loc[:, ["lead_time", "cloud_error_pp", "ghi_error_wm2"]].to_csv(
        destination / "fig_cloud_error_ghi_error_hexbin_source.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 70 / 25.4), sharex=True, sharey=True)
    artist = None
    for axis, lead in zip(axes, (24, 48, 72)):
        group = selected.loc[selected.lead_time == lead]
        artist = axis.hexbin(group.cloud_error_pp, group.ghi_error_wm2, gridsize=38,
                             extent=(-100, 100, -1000, 1000), mincnt=1,
                             bins="log", cmap="Blues")
        axis.axhline(0, color=GRAY, lw=0.6)
        axis.axvline(0, color=GRAY, lw=0.6)
        axis.set_title(f"D+{lead // 24}  (n={len(group):,})")
        axis.set_xlabel("Forecast − ERA5 cloud cover (pp)")
    axes[0].set_ylabel("GFS − Himawari GHI (W m$^{-2}$)")
    fig.suptitle("Cloud-cover disagreement co-occurs with GHI forecast error", x=0.04,
                 ha="left", y=1.01)
    fig.colorbar(artist, ax=axes, label="log hexagon count", fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.23, top=0.79, wspace=0.08)
    return {"files": save(fig, destination, "fig_cloud_error_ghi_error_hexbin"),
            "source_data": "fig_cloud_error_ghi_error_hexbin_source.csv",
            "n_definition": "daytime site×valid-hour×lead rows with both GFS and ERA5 cloud values; ERA5 is supplementary reference"}


def feature_distribution(frame: pd.DataFrame, destination: Path) -> dict:
    selected = frame.loc[(frame.solar_elevation > 0)
                         & frame[["cloud_cover_fcst", "kt_raw", "ghi_fcst", "ghi_clear_sky"]]
                         .notna().all(axis=1),
                         ["lead_time", "cloud_cover_fcst", "kt_raw", "ghi_fcst", "ghi_clear_sky"]].copy()
    selected.to_csv(destination / "fig_feature_distribution_hexbin_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 73 / 25.4))
    h0 = axes[0].hexbin(selected.cloud_cover_fcst, selected.kt_raw, gridsize=42,
                        extent=(0, 100, 0, 3), mincnt=1, bins="log", cmap="Blues")
    axes[0].set(xlabel="GFS total cloud cover (%)", ylabel="Raw unclipped GFS clear-sky index")
    h1 = axes[1].hexbin(selected.ghi_clear_sky, selected.ghi_fcst, gridsize=42,
                        extent=(0, 1200, 0, 1200), mincnt=1, bins="log", cmap="Blues")
    axes[1].plot([0, 1200], [0, 1200], color=AMBER, lw=0.7, ls="--")
    axes[1].set(xlabel="Clear-sky GHI (W m$^{-2}$)", ylabel="GFS GHI (W m$^{-2}$)")
    for axis, label in zip(axes, ("a", "b")):
        axis.text(-0.1, 1.02, label, transform=axis.transAxes, fontweight="bold", fontsize=8)
    fig.suptitle(f"Formal forecast-feature support (n={len(selected):,} site-hour-lead rows)",
                 x=0.04, ha="left", y=1.01)
    fig.colorbar(h1, ax=axes, label="log hexagon count", fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.09, right=0.87, bottom=0.24, top=0.79, wspace=0.36)
    return {"files": save(fig, destination, "fig_feature_distribution_hexbin"),
            "source_data": "fig_feature_distribution_hexbin_source.csv",
            "n_definition": "daytime site×valid-hour×lead rows with finite formal forecast features"}


def fixed_tables(predictions: pd.DataFrame, destination: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = []
    for keys, group in predictions.groupby(["model_id", "outer_fold", "lead_time"], sort=True):
        grouped.append({"model_id": keys[0], "outer_fold": keys[1], "lead_time": int(keys[2]),
                        "n": len(group), "rmse": rmse(group)})
    by_lead_fold = pd.DataFrame(grouped)
    by_lead_fold.to_csv(destination / "fixed_rmse_by_fold_lead.csv", index=False)
    pooled = []
    for model, group in predictions.groupby("model_id"):
        pooled.append({"model_id": model, "n": len(group), "rmse": rmse(group)})
    overall = pd.DataFrame(pooled)
    overall.to_csv(destination / "fixed_rmse_overall.csv", index=False)
    fold = []
    for (model, outer), group in predictions.groupby(["model_id", "outer_fold"]):
        fold.append({"model_id": model, "outer_fold": outer, "n": len(group), "rmse": rmse(group)})
    by_fold = pd.DataFrame(fold)
    by_fold.to_csv(destination / "fixed_rmse_by_fold.csv", index=False)
    return overall, by_fold, by_lead_fold


def value_ladder(overall: pd.DataFrame, by_fold: pd.DataFrame, destination: Path) -> dict:
    order = ["climatology", "persistence", "smart_persistence", "optimal_convex",
             "raw_gfs", "bias_correction", "linear_mos"]
    lookup = overall.set_index("model_id")
    fig, ax = plt.subplots(figsize=(183 / 25.4, 90 / 25.4))
    y = np.arange(len(order))
    colors = [GRAY] * 4 + [AMBER] + [BLUE] * 2
    ax.barh(y, [lookup.loc[name, "rmse"] for name in order], color=colors, height=0.55)
    for index, name in enumerate(order):
        values = by_fold.loc[by_fold.model_id == name, "rmse"].to_numpy(dtype=float)
        ax.scatter(values, np.full(len(values), index), s=8, facecolor="white",
                   edgecolor="#263942", linewidth=0.5, zorder=3)
    ax.axvline(float(lookup.loc["raw_gfs", "rmse"]), color=AMBER, lw=0.8, ls="--")
    ax.set_yticks(y, order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (W m$^{-2}$), pooled five original outer validations")
    ax.set_title("Fixed-model value ladder; dots are fold scores, not confidence intervals",
                 loc="left", pad=12)
    ax.grid(axis="x", color=PALE, lw=0.5)
    fig.tight_layout()
    return {"files": save(fig, destination, "fig_value_ladder"),
            "source_data": ["fixed_rmse_overall.csv", "fixed_rmse_by_fold.csv"],
            "n_definition": "9,564 matched daytime Nanjing site-hour-lead rows per model across five outer folds"}


def lead_performance(by_lead_fold: pd.DataFrame, predictions: pd.DataFrame, destination: Path) -> dict:
    order = ["climatology", "persistence", "smart_persistence", "optimal_convex",
             "raw_gfs", "bias_correction", "linear_mos"]
    table = []
    for (model, lead), group in predictions.groupby(["model_id", "lead_time"]):
        table.append({"model_id": model, "lead_time": int(lead), "n": len(group), "rmse": rmse(group)})
    summary = pd.DataFrame(table)
    summary.to_csv(destination / "fig_cpu_lead_performance_source.csv", index=False)
    matrix = summary.pivot(index="model_id", columns="lead_time", values="rmse").loc[order, [24, 48, 72]]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 89 / 25.4))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto", vmin=100, vmax=300)
    ax.set_xticks([0, 1, 2], ["D+1", "D+2", "D+3"])
    ax.set_yticks(np.arange(len(order)), order)
    for i in range(len(order)):
        for j in range(3):
            value = float(matrix.iloc[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 225 else "#20303B", fontsize=7)
    fig.colorbar(image, ax=ax, label="RMSE (W m$^{-2}$)", fraction=0.04, pad=0.03)
    ax.set_title("Lead-wise performance uses identical Nanjing outer-score timestamps",
                 loc="left", pad=12)
    fig.tight_layout()
    return {"files": save(fig, destination, "fig_cpu_lead_performance"),
            "source_data": "fig_cpu_lead_performance_source.csv",
            "n_definition": "matched daytime outer-validation rows per model and lead"}


def improvement_by_fold(by_fold: pd.DataFrame, destination: Path) -> dict:
    raw = by_fold.loc[by_fold.model_id == "raw_gfs", ["outer_fold", "rmse"]].rename(columns={"rmse": "raw_rmse"})
    improvement = by_fold.merge(raw, on="outer_fold", validate="many_to_one")
    improvement = improvement.loc[improvement.model_id != "raw_gfs"].copy()
    improvement["rmse_improvement_pct"] = 100 * (1 - improvement.rmse / improvement.raw_rmse)
    improvement.to_csv(destination / "fig_cpu_improvement_vs_raw_source.csv", index=False)
    order = ["climatology", "persistence", "smart_persistence", "optimal_convex",
             "bias_correction", "linear_mos"]
    fig, ax = plt.subplots(figsize=(183 / 25.4, 82 / 25.4))
    for index, model in enumerate(order):
        values = improvement.loc[improvement.model_id == model, "rmse_improvement_pct"].to_numpy()
        center = float(values.mean())
        ax.plot([values.min(), values.max()], [index, index], color=GRAY, lw=1)
        ax.scatter(values, np.full(len(values), index), color=GRAY, s=9, zorder=3)
        ax.scatter([center], [index], color=BLUE if center >= 0 else RED,
                   s=30, marker="D", zorder=4)
    ax.axvline(0, color=AMBER, lw=0.8)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE improvement relative to matched raw GFS (%)")
    ax.set_title("Five-fold spread in fixed-model improvement; diamond = fold mean",
                 loc="left", pad=12)
    ax.grid(axis="x", color=PALE, lw=0.5)
    fig.tight_layout()
    return {"files": save(fig, destination, "fig_cpu_improvement_vs_raw"),
            "source_data": "fig_cpu_improvement_vs_raw_source.csv",
            "n_definition": "five chronological outer-fold contrasts per model, one site; no confidence interval claimed"}


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    data_root = ROOT / manifest["data_layout"]["root"]
    feature_path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    prediction_path = data_root / "06_cpu_single_site" / "outer_fixed_predictions.parquet"
    feature_receipt = json.loads(feature_path.with_name(feature_path.name + ".receipt.json").read_text(encoding="utf-8"))
    if feature_receipt["data_version"] != manifest["data_version"]:
        raise ValueError("Nanjing feature version does not match manifest")
    frame = pd.read_parquet(feature_path)
    predictions = pd.read_parquet(prediction_path)
    if set(predictions.result_status) != {"diagnostic"} or set(predictions.location_id) != set(frame.location_id):
        raise ValueError("prediction result status or location mismatch")
    folder = ROOT / "figs" / "nanjing_15var_diagnostic"
    data_dir, model_dir = folder / "01_data", folder / "03_model_benchmark"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fig_data_source_timeline": data_timeline(frame, data_dir),
        "fig_cloud_error_ghi_error_hexbin": hexbin_cloud_error(frame, data_dir),
        "fig_feature_distribution_hexbin": feature_distribution(frame, data_dir),
    }
    overall, by_fold, by_lead_fold = fixed_tables(predictions, model_dir)
    outputs.update({
        "fig_value_ladder": value_ladder(overall, by_fold, model_dir),
        "fig_cpu_lead_performance": lead_performance(by_lead_fold, predictions, model_dir),
        "fig_cpu_improvement_vs_raw": improvement_by_fold(by_fold, model_dir),
    })
    metadata = {"result_status": "diagnostic", "official_eligible": False,
                "site": "nanjing_1", "data_version": manifest["data_version"],
                "figure_count": len(outputs), "figures": outputs}
    (folder / "figure_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"figure_count": len(outputs), "figure_names": list(outputs),
                      "result_status": "diagnostic"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
