"""Five original outer folds for fixed CPU baselines on Nanjing only.

This is a one-site diagnostic checkpoint, not an official 20-site benchmark.
No external data requests or changes to the configured temporal windows occur.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s04_splits.rolling import outer_folds  # noqa: E402
from s06_models.cpu_fixed import FIXED_MODEL_IDS, predict_fixed_cpu  # noqa: E402
from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402
from s07_prediction.writer import write_predictions  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402


def usable(block: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(block) & np.isfinite(pd.to_numeric(block.ghi_obs_sat, errors="coerce"))
    out = block.loc[mask].copy()
    out["y"] = out.ghi_obs_sat.astype(float)
    return out


def prediction_rows(score: pd.DataFrame, model_id: str, point: np.ndarray,
                    outer_id: str, manifest: dict, gap_days: int = 10) -> pd.DataFrame:
    out = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": "prototype",
        "execution_level": "development",
        "prediction_type": "point",
        "location_id": score.location_id.to_numpy(),
        "requested_coordinates": list(zip(score.requested_latitude, score.requested_longitude)),
        "gfs_service_coordinates": list(zip(score.gfs_service_latitude, score.gfs_service_longitude)),
        "truth_service_coordinates": list(zip(score.himawari_service_latitude, score.himawari_service_longitude)),
        "target_time_utc": score.target_time_utc.to_numpy(),
        "forecast_issue_time_utc": score.forecast_issue_time_utc.to_numpy(),
        "lead_time": score.lead_time.to_numpy(),
        "outer_fold": outer_id,
        "inner_fold": "outer_refit",
        "seed": 0,
        "y": score.y.to_numpy(dtype=float),
        "point_prediction": point,
        "data_version": manifest["data_version"],
        "feature_version": manifest["feature_version"],
        "protocol_revision": manifest["protocol_version"],
        "experiment_id": ("nanjing_15var_outer_diagnostic" if gap_days == 10
                          else f"nanjing_15var_outer_gap{gap_days}_diagnostic"),
        "result_status": "diagnostic",
    })
    for column in QUANTILE_COLUMNS:
        out[column] = np.nan
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gap-days", type=int, choices=(7, 10, 14), default=10)
    gap = parser.parse_args().gap_days
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    data_root = ROOT / manifest["data_layout"]["root"]
    path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    metadata = json.loads(path.with_name(path.name + ".receipt.json").read_text(encoding="utf-8"))
    if metadata["data_version"] != manifest["data_version"]:
        raise ValueError("wrong feature data version")
    frame = pd.read_parquet(path)
    model_ids = [model_id for model_id in manifest["cpu_experiment"]["model_ids"]
                 if model_id in FIXED_MODEL_IDS]
    predictions = []
    sample_counts = []
    for outer in outer_folds(frame, gap_days=gap):
        fit, score = usable(outer.fit), usable(outer.score)
        if fit.empty or score.empty:
            raise ValueError(f"{outer.fold_id}: no eligible fit/score samples")
        counts = {"outer_fold": outer.fold_id, "fit_raw_rows": len(outer.fit),
                  "fit_eligible_rows": len(fit), "score_raw_rows": len(outer.score),
                  "score_eligible_rows": len(score)}
        for model_id in model_ids:
            point, _ = predict_fixed_cpu(model_id, fit, fit.iloc[:0], score)
            point = np.asarray(point, dtype=float)
            if point.shape != (len(score),) or not np.isfinite(point).all():
                raise ValueError(f"{outer.fold_id}/{model_id}: non-finite or incomplete output")
            predictions.append(prediction_rows(score, model_id, point, outer.fold_id, manifest,
                                               gap_days=gap))
        sample_counts.append(counts)
    combined = pd.concat(predictions, ignore_index=True)
    output = data_root / "06_cpu_single_site"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if gap == 10 else f"_gap{gap}"
    saved = write_predictions(combined, output / f"outer_fixed{suffix}_predictions.parquet")
    report = evaluate_predictions(saved, formal=False)
    summary = {
        "scope": "nanjing_one_site_five_original_outer_folds_fixed_cpu_models_only",
        "official_eligible": False,
        "gap_days": gap,
        "data_version": manifest["data_version"],
        "models": model_ids,
        "sample_counts": sample_counts,
        "prediction_rows": len(combined),
        "point_overview": report.overview.point_secondary.loc[:,
            ["model_id", "n", "mae", "rmse", "bias", "r2", "rmse_skill", "rmse_skill_reference"]
        ].to_dict("records"),
    }
    def json_safe(value):
        if isinstance(value, dict):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        return value

    cleaned = json_safe(summary)
    (output / f"outer_fixed{suffix}_summary.json").write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(cleaned, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
