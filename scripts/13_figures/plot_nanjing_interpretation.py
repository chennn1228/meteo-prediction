"""Development-only Nanjing model explanation and residual diagnostics.

The selected outer-5 LightGBM median is reproduced on its original outer fit
and score before SHAP. Every explanation, residual, and sample date is inside
the development period; final-test features and labels are not read here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s03_features.preprocessing import fit_fold_preprocessing  # noqa: E402
from s04_splits.rolling import outer_folds  # noqa: E402
from s11_interpretation.development_gate import require_development_rows  # noqa: E402
from s13_visualization.s01_style.publication import apply_publication_style  # noqa: E402

SCOPE = "Nanjing only | development-fold diagnostic | not official 20-site evidence"


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(frame) & np.isfinite(pd.to_numeric(frame.ghi_obs_sat, errors="coerce"))
    result = frame.loc[mask].copy()
    result["y"] = result.ghi_obs_sat.astype(float)
    require_development_rows(result)
    return result


def save(fig, directory: Path, name: str) -> list[str]:
    fig.text(0.995, 0.995, SCOPE, ha="right", va="top", fontsize=6, color="#8C959A")
    files = []
    for extension in ("svg", "png"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(str(path.relative_to(ROOT)))
    plt.close(fig)
    return files


def reproduce_outer_median(frame: pd.DataFrame, predictions: pd.DataFrame,
                           receipt: dict):
    outer = next(fold for fold in outer_folds(frame) if fold.fold_id == "outer_5")
    fit, score = eligible(outer.fit), eligible(outer.score)
    (x_fit, _, x_score), prep = fit_fold_preprocessing(fit, fit.iloc[:0], score, seed=0)
    rounds = int(receipt["fit_receipt"]["refit_rounds_by_quantile"]["0.5"])
    model = lgb.LGBMRegressor(objective="quantile", alpha=0.50, metric="quantile",
                              n_estimators=rounds, random_state=0, verbose=-1,
                              **receipt["selected_parameters"])
    model.fit(x_fit, fit.y.to_numpy(dtype=float))
    calculated = model.predict(x_score)
    saved = predictions.loc[(predictions.model_id == "lgbm") &
                            (predictions.outer_fold == "outer_5")].copy()
    require_development_rows(saved)
    keys = ["target_time_utc", "forecast_issue_time_utc", "lead_time"]
    comparison = score[keys].copy()
    comparison["calculated"] = calculated
    comparison = comparison.merge(saved[keys + ["q0.50"]], on=keys,
                                  validate="one_to_one")
    if len(comparison) != len(score):
        raise ValueError("reproduced model and saved outer-5 predictions differ in sample count")
    difference = float(np.max(np.abs(comparison.calculated - comparison["q0.50"])))
    if difference > 1e-6:
        raise ValueError(f"outer-5 median model did not reproduce: max diff {difference}")
    return model, score, x_score, prep, rounds, difference


def figure_shap(model, score: pd.DataFrame, x_score: pd.DataFrame,
                reproduction_diff: float, directory: Path) -> dict:
    require_development_rows(score)
    # Equal-stride within the outer-5 development score, independent of y.
    indices = np.unique(np.linspace(0, len(x_score) - 1, min(768, len(x_score)), dtype=int))
    selected = x_score.iloc[indices].copy()
    explanation = shap.TreeExplainer(model)(selected)
    if explanation.values.shape != selected.shape or not np.isfinite(explanation.values).all():
        raise ValueError("SHAP values have invalid shape or values")
    importance = pd.DataFrame({"feature": selected.columns,
                               "mean_abs_shap_w_m2": np.mean(np.abs(explanation.values), axis=0)})
    importance = importance.sort_values("mean_abs_shap_w_m2", ascending=False)
    importance.to_csv(directory / "fig_shap_beeswarm_importance.csv", index=False)
    top = importance.feature.head(15).tolist()
    source = score.iloc[indices][["target_time_utc", "forecast_issue_time_utc", "lead_time"]].reset_index(drop=True)
    for feature in top:
        position = selected.columns.get_loc(feature)
        source[f"value__{feature}"] = selected[feature].to_numpy(dtype=float)
        source[f"shap__{feature}"] = explanation.values[:, position]
    source.to_csv(directory / "fig_shap_beeswarm_source.csv", index=False)
    shap.plots.beeswarm(explanation, max_display=16, show=False, plot_size=(7.2, 4.6))
    fig = plt.gcf()
    main = fig.axes[0]
    main.set_title("Outer-5 LightGBM median: attribution, not causal effect",
                   loc="left", pad=9, fontsize=8)
    main.set_xlabel("SHAP impact on predicted median GHI (W m$^{-2}$)", fontsize=7)
    main.tick_params(axis="both", labelsize=7)
    if len(fig.axes) > 1:
        colour = fig.axes[1]
        colour.set_ylabel("Fold-scaled feature value", fontsize=7)
        colour.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.32, right=0.88, top=0.82, bottom=0.14)
    return {"files": save(fig, directory, "fig_shap_beeswarm"),
            "source_data": ["fig_shap_beeswarm_source.csv", "fig_shap_beeswarm_importance.csv"],
            "n_definition": f"{len(indices)} equal-stride outer-5 development forecasts; no final-test rows",
            "model_reproduction_max_abs_diff_w_m2": reproduction_diff,
            "caution": "SHAP explains one frozen fitted model; it does not select features or establish causality"}


def figure_hexbin(predictions: pd.DataFrame, frame: pd.DataFrame,
                  directory: Path) -> dict:
    keys = ["target_time_utc", "forecast_issue_time_utc", "lead_time"]
    part = predictions.loc[predictions.model_id == "lgbm", keys + ["y", "q0.50"]].copy()
    require_development_rows(part)
    source = frame.loc[:, keys + ["ghi_fcst", "cloud_cover_fcst"]]
    if source.duplicated(keys).any():
        raise ValueError("feature rows duplicate forecast keys")
    part = part.merge(source, on=keys, validate="one_to_one", how="left", indicator=True)
    if (part._merge != "both").any():
        raise ValueError("outer-score feature join incomplete")
    part["residual_w_m2"] = part.y - part["q0.50"]
    part.drop(columns=["_merge"]).to_csv(directory / "fig_key_feature_hexbin_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 79 / 25.4), sharey=True)
    radiation = part.loc[np.isfinite(part.ghi_fcst)]
    axes[0].hexbin(radiation.ghi_fcst, radiation.residual_w_m2, gridsize=42,
                   mincnt=1, bins="log", cmap="Blues")
    axes[0].set(xlabel="Issue-time GFS GHI (W m$^{-2}$)",
                ylabel="Observed − LightGBM median (W m$^{-2}$)",
                title=f"GFS radiation  (n={len(radiation):,})")
    cloud = part.loc[np.isfinite(part.cloud_cover_fcst)]
    artist = axes[1].hexbin(cloud.cloud_cover_fcst, cloud.residual_w_m2,
                             gridsize=42, mincnt=1, bins="log", cmap="Blues")
    axes[1].set(xlabel="Issue-time GFS total cloud (%)",
                title=f"Forecast cloud  (n={len(cloud):,})")
    for ax in axes:
        ax.axhline(0, color="#8C959A", lw=0.7)
    fig.colorbar(artist, ax=axes, label="Hexagon count (log colour)", fraction=0.025, pad=0.02)
    fig.suptitle("Development-only residual structure across forecast inputs",
                 x=0.04, ha="left", y=1.02)
    fig.subplots_adjust(left=0.11, right=0.87, bottom=0.20, top=0.78, wspace=0.16)
    return {"files": save(fig, directory, "fig_key_feature_hexbin"),
            "source_data": "fig_key_feature_hexbin_source.csv",
            "n_definition": "observed development outer-score site×target-hour×lead rows across five folds",
            "caution": "feature-residual association is not group ablation or causal contribution"}


def main() -> None:
    apply_publication_style()
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    root = ROOT / manifest["data_layout"]["root"]
    frame = pd.read_parquet(root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet")
    predictions = pd.read_parquet(root / "06_cpu_single_site" / "outer_tuned_predictions.parquet")
    receipt = json.loads((root / "06_cpu_single_site" / "outer_tuned_parts" /
                          "outer_5_lgbm.json").read_text(encoding="utf-8"))
    if (receipt.get("official_eligible") is not False
            or set(predictions.result_status) != {"diagnostic"}
            or len(set(predictions.location_id)) != 1):
        raise ValueError("only diagnostic one-site outer artifacts allowed")
    directory = ROOT / "figs" / "nanjing_15var_diagnostic" / "05_interpretation"
    directory.mkdir(parents=True, exist_ok=True)
    model, score, x_score, prep, rounds, difference = reproduce_outer_median(
        frame, predictions, receipt)
    figures = {
        "fig_shap_beeswarm": figure_shap(model, score, x_score, difference, directory),
        "fig_key_feature_hexbin": figure_hexbin(predictions, frame, directory),
    }
    audit = {"scope": "nanjing_outer_5_development_only_interpretation_diagnostic",
             "official_eligible": False, "model_id": "lgbm", "quantile": 0.5,
             "preprocessing": prep, "frozen_inner_median_rounds": rounds,
             "reproduction_max_abs_diff_w_m2": difference}
    (directory / "model_reproduction_receipt.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path = directory.parent / "figure_manifest.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["figures"].update(figures)
    metadata["figure_count"] = len(metadata["figures"])
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": list(figures), "total_figures": metadata["figure_count"],
                      "reproduction_max_abs_diff_w_m2": difference}, ensure_ascii=False))


if __name__ == "__main__":
    main()
