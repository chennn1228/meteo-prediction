"""Offline real-data CPU interface test, never an official benchmark.

Uses the already cached current-contract Nanjing months. One inner fold and
one pre-registered Ridge candidate exercise the complete data-to-metrics path;
they cannot be used to select a model or report an official score.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s04_splits.rolling import inner_folds, outer_folds  # noqa: E402
from s05_tuning.estimators import ridge_fit_predict, tree_fit_predict  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402
from s06_models.cpu_fixed import FIXED_MODEL_IDS, predict_fixed_cpu  # noqa: E402
from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402
from s07_prediction.writer import write_predictions  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(block: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(block) & np.isfinite(pd.to_numeric(block["ghi_obs_sat"], errors="coerce"))
    result = block.loc[mask].copy()
    result["y"] = result["ghi_obs_sat"].astype(float)
    if result.empty:
        raise ValueError("no eligible real daytime Himawari observations")
    return result


def prediction_rows(score: pd.DataFrame, model_id: str, values: np.ndarray,
                    *, quantile: bool, manifest: dict) -> pd.DataFrame:
    out = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": "prototype",
        "execution_level": "smoke",
        "prediction_type": "quantile" if quantile else "point",
        "location_id": score["location_id"].to_numpy(),
        "requested_coordinates": list(zip(score.requested_latitude, score.requested_longitude)),
        "gfs_service_coordinates": list(zip(score.gfs_service_latitude, score.gfs_service_longitude)),
        "truth_service_coordinates": list(zip(score.himawari_service_latitude, score.himawari_service_longitude)),
        "target_time_utc": score.target_time_utc.to_numpy(),
        "forecast_issue_time_utc": score.forecast_issue_time_utc.to_numpy(),
        "lead_time": score.lead_time.to_numpy(),
        "outer_fold": "outer_1",
        "inner_fold": "inner_1",
        "seed": 0,
        "y": score.y.to_numpy(dtype=float),
        "point_prediction": values[:, 3] if quantile else values,
        "data_version": manifest["data_version"],
        "feature_version": manifest["feature_version"],
        "protocol_revision": manifest["protocol_version"],
        "experiment_id": "nanjing_current_cache_cpu_interface_smoke",
        "result_status": "diagnostic",
    })
    for index, column in enumerate(QUANTILE_COLUMNS):
        out[column] = values[:, index] if quantile else np.nan
    return out


def main() -> None:
    manifest = load_manifest(ROOT / "project_manifest.yaml")
    data_root = ROOT / manifest["data_layout"]["root"]
    source = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    source_receipt = source.with_name(source.name + ".receipt.json")
    clean = data_root / "02_clean" / "nanjing_1_clean_2024-02_2026-08.parquet"
    feature_metadata = json.loads(source_receipt.read_text(encoding="utf-8"))
    if feature_metadata["data_version"] != manifest["data_version"] or not clean.is_file():
        raise ValueError("current-contract real feature/clean inputs not verified")
    frame = pd.read_parquet(source)
    if len(frame) != feature_metadata["rows"] or set(frame.station_id) != {"nanjing_1"}:
        raise ValueError("Nanjing feature receipt or source content mismatch")
    outer = outer_folds(frame)[0]
    inner = inner_folds(outer)[0]
    fit, early, score = (eligible(block) for block in (inner.fit, inner.early_stop, inner.score))
    fixed = tuple(model_id for model_id in manifest["cpu_experiment"]["model_ids"]
                  if model_id in FIXED_MODEL_IDS)
    predictions = []
    fixed_metadata = {}
    for model_id in fixed:
        values, metadata = predict_fixed_cpu(model_id, fit, early, score)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(score),) or not np.isfinite(values).all():
            raise ValueError(f"{model_id} produced invalid real-data predictions")
        predictions.append(prediction_rows(score, model_id, values,
                                           quantile=False, manifest=manifest))
        fixed_metadata[model_id] = metadata["definition"]
    taus = tuple(float(value) for value in manifest["quantiles"])
    quantile_models = ("ridge_mos", "lgbm", "xgboost")
    for model_id in quantile_models:
        adapter = ridge_fit_predict if model_id == "ridge_mos" else tree_fit_predict(model_id)
        values, _, _ = adapter(candidates(model_id)[0], fit, early, score, 0, taus)
        values = np.asarray(values, dtype=float)
        if values.shape != (len(score), len(taus)) or not np.isfinite(values).all():
            raise ValueError(f"{model_id} produced invalid real-data seven-quantile predictions")
        predictions.append(prediction_rows(score, model_id, values,
                                           quantile=True, manifest=manifest))
    combined = pd.concat(predictions, ignore_index=True)
    evaluation = evaluate_predictions(combined, formal=False)
    if len(evaluation.overview.probability_primary) != len(quantile_models) or len(evaluation.overview.point_secondary) != len(fixed) + len(quantile_models):
        raise ValueError("evaluation did not retain separate probability/point tables")
    output = data_root / "05_cpu_mini_e2e"
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = write_predictions(combined, output / "predictions.parquet")
    metrics = {
        "scope": "one_real_site_one_inner_fold_diagnostic_only",
        "mean_pinball": {str(row.model_id): float(row.mean_pinball)
                         for row in evaluation.overview.probability_primary.itertuples()},
        "crossing_rate": {str(row.model_id): float(row.crossing_rate)
                          for row in evaluation.overview.probability_primary.itertuples()},
        "fixed_point_models": fixed_metadata,
        "score_rows_per_model": len(score),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    receipt = {
        "status": "pass",
        "result_status": "diagnostic",
        "execution_level": "smoke",
        "official_eligible": False,
        "data_version": manifest["data_version"],
        "feature_version": manifest["feature_version"],
        "protocol_version": manifest["protocol_version"],
        "site": "nanjing_1",
        "outer_fold": outer.fold_id,
        "inner_fold": inner.fold_id,
        "fit_rows": len(fit),
        "early_stop_rows": len(early),
        "scoring_rows": len(score),
        "daylight_definition": manifest["daylight_definition"]["formal"],
        "fixed_models_tested": list(fixed),
        "quantile_models_tested": list(quantile_models),
        "quantile_count": len(taus),
        "source_feature_sha256": sha256(source),
        "source_clean_sha256": sha256(clean),
        "prediction_sha256": sha256(prediction_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"receipt": receipt, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
