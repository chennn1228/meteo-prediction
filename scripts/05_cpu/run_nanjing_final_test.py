"""Offline Nanjing-only final chronology and causal probability calibration.

Requires all five original outer-prefix inner-tuning ledgers. The test year is
never consulted for candidate selection. Outputs remain diagnostic because
the registered CPU cohort is 20 sites and the old 19-site cache lacks exact
request-provenance sidecars.
"""
from __future__ import annotations

import hashlib
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
from s04_splits.rolling import assert_gap  # noqa: E402
from s05_tuning.estimators import ridge_fit_predict, tree_fit_predict  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402
from s06_models.cpu_fixed import FIXED_MODEL_IDS, predict_fixed_cpu  # noqa: E402
from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402
from s07_prediction.writer import write_predictions  # noqa: E402
from s08_calibration.quantile import CausalIssueQuantileCalibrator  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402

TUNED = ("ridge_mos", "lgbm", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window(frame: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    time = pd.to_datetime(frame.target_time_utc, utc=True)
    start = pd.Timestamp(first, tz="UTC")
    stop = pd.Timestamp(last, tz="UTC") + pd.Timedelta(days=1)
    return frame.loc[(time >= start) & (time < stop)].copy()


def daylight(block: pd.DataFrame, *, require_truth: bool) -> pd.DataFrame:
    mask = formal_daylight_mask(block)
    if require_truth:
        mask &= np.isfinite(pd.to_numeric(block.ghi_obs_sat, errors="coerce"))
    result = block.loc[mask].copy()
    result["y"] = pd.to_numeric(result.ghi_obs_sat, errors="coerce")
    if result.empty:
        raise ValueError("configured final block has no eligible daylight rows")
    return result


def selected_from_inner_ledgers(folder: Path, model_id: str,
                                source_sha: str, manifest_sha: str) -> tuple[int, dict]:
    records = []
    ledger_hashes = {}
    for index in range(1, 6):
        path = folder / f"outer_{index}_{model_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "pass" or payload.get("site") != "nanjing_1"
                or payload.get("model_id") != model_id
                or payload.get("source_feature_sha256") != source_sha
                or payload.get("manifest_sha256") != manifest_sha
                or len(payload.get("trials", [])) != 18):
            raise ValueError(f"invalid complete inner-only tuning ledger: {path}")
        ledger_hashes[path.name] = sha256(path)
        records.extend(payload["trials"])
    totals = {}
    for candidate in range(6):
        matched = [row for row in records if row["candidate"] == candidate]
        if len(matched) != 15 or any(row["status"] != "ok" or row["mean_pinball"] is None
                                     for row in matched):
            raise ValueError(f"{model_id}: not all 15 inner scores available for candidate {candidate}")
        totals[candidate] = float(np.mean([row["mean_pinball"] for row in matched]))
    chosen = min(totals, key=lambda index: (totals[index], index))
    return chosen, {"inner_mean_pinball_by_candidate": totals,
                    "five_outer_inner_ledger_sha256": ledger_hashes}


def prediction_rows(block: pd.DataFrame, model_id: str, predictions: np.ndarray,
                    *, quantile: bool, manifest: dict, experiment_id: str) -> pd.DataFrame:
    values = np.asarray(predictions, dtype=float)
    expected = (len(block), len(QUANTILE_COLUMNS)) if quantile else (len(block),)
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"{model_id}: final prediction shape/finite contract failed")
    result = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": "prototype",
        "execution_level": "development",
        "prediction_type": "quantile" if quantile else "point",
        "location_id": block.location_id.to_numpy(),
        "requested_coordinates": list(zip(block.requested_latitude, block.requested_longitude)),
        "gfs_service_coordinates": list(zip(block.gfs_service_latitude, block.gfs_service_longitude)),
        "truth_service_coordinates": list(zip(block.himawari_service_latitude, block.himawari_service_longitude)),
        "target_time_utc": block.target_time_utc.to_numpy(),
        "forecast_issue_time_utc": block.forecast_issue_time_utc.to_numpy(),
        "lead_time": block.lead_time.to_numpy(),
        "outer_fold": "final_test",
        "inner_fold": "final_fit",
        "seed": 0,
        "y": block.y.to_numpy(dtype=float),
        "point_prediction": values[:, 3] if quantile else values,
        "data_version": manifest["data_version"],
        "feature_version": manifest["feature_version"],
        "protocol_revision": manifest["protocol_version"],
        "experiment_id": experiment_id,
        "result_status": "diagnostic",
    })
    for index, column in enumerate(QUANTILE_COLUMNS):
        result[column] = values[:, index] if quantile else np.nan
    return result


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


def main() -> None:
    manifest_path = ROOT / "project_manifest.yaml"
    manifest = load_manifest(manifest_path)
    data_root = ROOT / manifest["data_layout"]["root"]
    feature_path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    receipt = json.loads(feature_path.with_name(feature_path.name + ".receipt.json").read_text(encoding="utf-8"))
    if receipt["data_version"] != manifest["data_version"]:
        raise ValueError("feature data version mismatch")
    source_sha, manifest_sha = sha256(feature_path), sha256(manifest_path)
    frame = pd.read_parquet(feature_path)
    schedule = manifest["validation_protocol"]["final_fit"]
    fit_raw = window(frame, manifest["development_period"]["start"][:10], schedule["fit_end"])
    early_raw = window(frame, schedule["early_stop_start"], schedule["early_stop_end"])
    calib_raw = window(frame, schedule["calibration_start"], schedule["calibration_end"])
    test_raw = window(frame, manifest["test_period"]["start"][:10],
                      manifest["test_period"]["end"][:10])
    assert_gap(fit_raw, early_raw, 10)
    assert_gap(early_raw, calib_raw, 10)
    fit, early, calibration = (daylight(block, require_truth=True)
                               for block in (fit_raw, early_raw, calib_raw))
    test = daylight(test_raw, require_truth=False)
    fit_end = pd.Timestamp(schedule["fit_end"], tz="UTC") + pd.Timedelta(hours=23)
    early_end = pd.Timestamp(schedule["early_stop_end"], tz="UTC") + pd.Timedelta(hours=23)
    quantile_levels = tuple(float(value) for value in manifest["quantiles"])
    experiment_id = "nanjing_15var_final_test_diagnostic"
    raw_rows, calibrated_rows = [], []
    history = pd.concat([early, calibration], ignore_index=True)
    fixed_ids = [model_id for model_id in manifest["cpu_experiment"]["model_ids"]
                 if model_id in FIXED_MODEL_IDS]
    for model_id in fixed_ids:
        point, _ = predict_fixed_cpu(model_id, fit, history, test)
        raw_rows.append(prediction_rows(test, model_id, point, quantile=False,
                                        manifest=manifest, experiment_id=experiment_id))
    tuning = {}
    combined_score = pd.concat([calibration, test], ignore_index=True)
    for model_id in TUNED:
        chosen, evidence = selected_from_inner_ledgers(
            data_root / "06_cpu_single_site" / "tuning", model_id, source_sha, manifest_sha
        )
        adapter = ridge_fit_predict if model_id == "ridge_mos" else tree_fit_predict(model_id)
        predictions, epoch, _ = adapter(candidates(model_id)[chosen], fit, early,
                                        combined_score, 0, quantile_levels)
        predictions = np.asarray(predictions, dtype=float)
        calibration_values, test_values = predictions[:len(calibration)], predictions[len(calibration):]
        calib_rows = prediction_rows(calibration, model_id, calibration_values,
                                     quantile=True, manifest=manifest, experiment_id=experiment_id)
        test_rows = prediction_rows(test, model_id, test_values,
                                    quantile=True, manifest=manifest, experiment_id=experiment_id)
        calibrator = CausalIssueQuantileCalibrator().fit(
            calib_rows, fit_end=fit_end, early_stop_end=early_end
        )
        calibrated = calibrator.apply(test_rows)
        raw_rows.append(test_rows)
        calibrated_rows.append(calibrated)
        tuning[model_id] = {"chosen_candidate": chosen,
                            "chosen_parameters": candidates(model_id)[chosen],
                            "final_early_stop_epoch": epoch, **evidence}
    raw_all = pd.concat(raw_rows, ignore_index=True)
    calibrated_all = pd.concat(calibrated_rows, ignore_index=True)
    output = data_root / "06_cpu_single_site"
    raw_path = write_predictions(raw_all, output / "final_test_uncalibrated.parquet",
                                 require_truth=False)
    calibrated_path = write_predictions(calibrated_all, output / "final_test_calibrated.parquet",
                                        require_truth=False)
    raw_evaluable = raw_all.loc[np.isfinite(raw_all.y)].copy()
    calibrated_evaluable = calibrated_all.loc[np.isfinite(calibrated_all.y)].copy()
    raw_report = evaluate_predictions(raw_evaluable, formal=False)
    calibrated_report = evaluate_predictions(calibrated_evaluable, formal=False)
    summary = {
        "scope": "nanjing_one_site_final_chronology_diagnostic_only",
        "official_eligible": False,
        "data_version": manifest["data_version"],
        "source_feature_sha256": source_sha,
        "chronology": schedule,
        "fit_eligible_rows": len(fit),
        "early_stop_eligible_rows": len(early),
        "calibration_eligible_rows": len(calibration),
        "test_daylight_rows": len(test),
        "test_missing_truth_rows": int(test.y.isna().sum()),
        "test_evaluable_rows_per_model": int(np.isfinite(test.y).sum()),
        "fixed_models": fixed_ids,
        "tuned_model_selection": tuning,
        "test_raw_prediction_sha256": sha256(raw_path),
        "test_calibrated_prediction_sha256": sha256(calibrated_path),
        "uncalibrated_probability": raw_report.overview.probability_primary.to_dict("records"),
        "calibrated_probability": calibrated_report.overview.probability_primary.to_dict("records"),
        "point_auxiliary": raw_report.overview.point_secondary.to_dict("records"),
        "calibration_available_rows_min": int(calibrated_all.calibration_available_rows.min()),
        "calibration_available_rows_max": int(calibrated_all.calibration_available_rows.max()),
    }
    (output / "final_test_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print(json.dumps(json_safe({key: value for key, value in summary.items()
                                if key not in {"uncalibrated_probability", "calibrated_probability", "point_auxiliary"}}),
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
