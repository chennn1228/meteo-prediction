"""Refit selected quantile models on all five original Nanjing outer folds.

Only the already-complete six-candidate/three-inner ledgers select each model.
Tree durations are recovered by rerunning just the selected candidate on the
same three inner folds, then taking the per-quantile median best round and
refitting on *all* eligible outer-training rows. The outer score is never an
early-stop or selection block. All outputs are one-site diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s03_features.preprocessing import fit_fold_preprocessing  # noqa: E402
from s04_splits.rolling import assert_gap, inner_folds, outer_folds  # noqa: E402
from s05_tuning.estimators import refit_tree_quantiles, tree_fit_predict  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402
from s07_prediction.schema import QUANTILE_COLUMNS  # noqa: E402
from s07_prediction.writer import write_predictions  # noqa: E402
from s09_metrics.probabilistic import probability_metrics  # noqa: E402
from s10_evaluation.runner import evaluate_predictions  # noqa: E402

MODELS = ("ridge_mos", "lgbm", "xgboost")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(block: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(block) & np.isfinite(pd.to_numeric(block.ghi_obs_sat, errors="coerce"))
    out = block.loc[mask].copy()
    out["y"] = out.ghi_obs_sat.astype(float)
    if out.empty:
        raise ValueError("empty eligible outer/inner block")
    return out


def verify_ledger(path: Path, *, model_id: str, outer_id: str,
                  source_sha: str, manifest_sha: str, gap_days: int = 10) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (payload.get("status") != "pass" or payload.get("official_eligible") is not False
            or payload.get("site") != "nanjing_1" or payload.get("model_id") != model_id
            or payload.get("outer_fold") != outer_id
            or payload.get("source_feature_sha256") != source_sha
            or payload.get("manifest_sha256") != manifest_sha
            or int(payload.get("gap_days", 10)) != gap_days
            or len(payload.get("trials", [])) != 18):
        raise ValueError(f"incomplete or stale inner ledger: {path}")
    averages = {}
    for candidate in range(6):
        records = [row for row in payload["trials"] if row["candidate"] == candidate]
        if (len(records) != 3 or len({row["inner"] for row in records}) != 3
                or any(row["status"] != "ok" or row["mean_pinball"] is None for row in records)):
            raise ValueError(f"{path}: incomplete candidate {candidate}")
        averages[candidate] = float(np.mean([row["mean_pinball"] for row in records]))
    selected = min(averages, key=lambda candidate: (averages[candidate], candidate))
    if selected != payload["selected_candidate"]:
        raise ValueError(f"{path}: stored selection differs from inner scores")
    return payload


def ridge_outer_refit(parameters: dict, folds: list, fit: pd.DataFrame,
                      score: pd.DataFrame, taus: tuple[float, ...]) -> tuple[np.ndarray, dict]:
    """Pool earlier inner early-stop residuals; refit point relation on full fit."""
    residuals = []
    for fold in folds:
        (x_fit, x_early, _), _ = fit_fold_preprocessing(
            fold.fit, fold.early_stop, fold.score)
        estimator = Ridge(alpha=parameters["alpha"])
        estimator.fit(x_fit, fold.fit.y.to_numpy(dtype=float))
        residuals.append(fold.early_stop.y.to_numpy(dtype=float) - estimator.predict(x_early))
    pooled = np.concatenate(residuals)
    if len(pooled) == 0 or not np.isfinite(pooled).all():
        raise ValueError("Ridge inner residual pool is missing or nonfinite")
    offsets = np.quantile(pooled, taus)
    (x_fit, _, x_score), prep = fit_fold_preprocessing(fit, fit.iloc[:0], score)
    estimator = Ridge(alpha=parameters["alpha"]).fit(x_fit, fit.y.to_numpy(dtype=float))
    values = estimator.predict(x_score)[:, None] + offsets[None, :]
    return values, {
        "round_selection_rule": "not_applicable_ridge",
        "residual_rule": "pooled_three_inner_early_stop_residuals",
        "inner_residual_rows": len(pooled),
        "residual_offsets_by_quantile": dict(zip((float(t) for t in taus), offsets.tolist())),
        "outer_fit_rows": len(fit), "outer_score_rows": len(score),
        "preprocessing": prep,
    }


def tree_outer_refit(model_id: str, parameters: dict, folds: list,
                     fit: pd.DataFrame, score: pd.DataFrame,
                     taus: tuple[float, ...], ledger: dict,
                     gap_days: int = 10) -> tuple[np.ndarray, dict]:
    adapter = tree_fit_predict(model_id)
    selected = ledger["selected_candidate"]
    for fold in folds:
        predicted, _, _ = adapter(parameters, fold.fit, fold.early_stop,
                                  fold.score, 0, taus)
        reproduced = float(probability_metrics(
            fold.score.y.to_numpy(dtype=float), np.asarray(predicted, dtype=float)
        )["mean_pinball"])
        original = next(row for row in ledger["trials"]
                        if row["candidate"] == selected and row["inner"] == fold.fold_id)
        if not np.isclose(reproduced, original["mean_pinball"], rtol=1e-6, atol=1e-6):
            raise ValueError(f"{model_id}/{fold.fold_id}: selected inner trial did not reproduce")
    predictions, receipt = refit_tree_quantiles(
        model_id, parameters, fit, score, 0, taus, adapter.receipts,
        gap_days=gap_days)
    receipt["reproduced_selected_inner_pinball"] = {
        fold.fold_id: float(next(row["mean_pinball"] for row in ledger["trials"]
                                if row["candidate"] == selected and row["inner"] == fold.fold_id))
        for fold in folds
    }
    receipt["selected_inner_round_receipts"] = adapter.receipts
    return predictions, receipt


def prediction_rows(score: pd.DataFrame, model_id: str, values: np.ndarray,
                    outer_id: str, manifest: dict, gap_days: int = 10) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    if values.shape != (len(score), len(QUANTILE_COLUMNS)) or not np.isfinite(values).all():
        raise ValueError(f"{outer_id}/{model_id}: incomplete quantile predictions")
    out = pd.DataFrame({
        "model_id": model_id,
        "implementation_level": "prototype",
        "execution_level": "development",
        "prediction_type": "quantile",
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
        "point_prediction": values[:, 3],
        "data_version": manifest["data_version"],
        "feature_version": manifest["feature_version"],
        "protocol_revision": manifest["protocol_version"],
        "experiment_id": ("nanjing_15var_outer_diagnostic" if gap_days == 10
                          else f"nanjing_15var_outer_gap{gap_days}_diagnostic"),
        "result_status": "diagnostic",
    })
    for column, values_at_tau in zip(QUANTILE_COLUMNS, values.T):
        out[column] = values_at_tau
    return out


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", choices=[f"outer_{i}" for i in range(1, 6)])
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--gap-days", type=int, choices=(7, 10, 14), default=10)
    args = parser.parse_args()
    gap = args.gap_days
    manifest_path = ROOT / "project_manifest.yaml"
    manifest = load_manifest(manifest_path)
    data_root = ROOT / manifest["data_layout"]["root"]
    feature_path = data_root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    receipt = json.loads(feature_path.with_name(feature_path.name + ".receipt.json").read_text(encoding="utf-8"))
    if receipt["data_version"] != manifest["data_version"]:
        raise ValueError("feature receipt version differs from manifest")
    source_sha, manifest_sha = sha256(feature_path), sha256(manifest_path)
    frame = pd.read_parquet(feature_path)
    output = data_root / "06_cpu_single_site"
    suffix = "" if gap == 10 else f"_gap{gap}"
    parts = output / f"outer_tuned{suffix}_parts"
    parts.mkdir(parents=True, exist_ok=True)
    taus = tuple(float(value) for value in manifest["quantiles"])
    for outer in outer_folds(frame, gap_days=gap):
        if args.outer and outer.fold_id != args.outer:
            continue
        fit, score = eligible(outer.fit), eligible(outer.score)
        assert_gap(fit, score, gap)
        folds = []
        for fold in inner_folds(outer, gap_days=gap):
            folds.append(type(fold)(fold.fold_id, eligible(fold.fit), eligible(fold.early_stop),
                                    eligible(fold.score), fold.fit_end, fold.early_stop_start,
                                    fold.early_stop_end, fold.score_start))
        for model_id in MODELS:
            if args.model and model_id != args.model:
                continue
            tuning_folder = "tuning" if gap == 10 else f"tuning_gap{gap}"
            ledger_path = output / tuning_folder / f"{outer.fold_id}_{model_id}.json"
            ledger = verify_ledger(ledger_path, model_id=model_id,
                                   outer_id=outer.fold_id, source_sha=source_sha,
                                   manifest_sha=manifest_sha, gap_days=gap)
            part = parts / f"{outer.fold_id}_{model_id}.parquet"
            part_receipt = parts / f"{outer.fold_id}_{model_id}.json"
            if part.is_file() and part_receipt.is_file():
                old = json.loads(part_receipt.read_text(encoding="utf-8"))
                if (old.get("source_feature_sha256") == source_sha
                        and old.get("manifest_sha256") == manifest_sha
                        and int(old.get("gap_days", 10)) == gap
                        and old.get("inner_ledger_sha256") == sha256(ledger_path)
                        and old.get("prediction_sha256") == sha256(part)):
                    print(f"REUSE {outer.fold_id}/{model_id}", flush=True)
                    continue
            parameters = candidates(model_id)[ledger["selected_candidate"]]
            if model_id == "ridge_mos":
                values, fit_receipt = ridge_outer_refit(parameters, folds, fit, score, taus)
            else:
                values, fit_receipt = tree_outer_refit(model_id, parameters, folds,
                                                       fit, score, taus, ledger,
                                                       gap_days=gap)
            saved = write_predictions(prediction_rows(score, model_id, values,
                                                       outer.fold_id, manifest,
                                                       gap_days=gap), part)
            metadata = {
                "scope": "nanjing_one_site_outer_refit_diagnostic_only",
                "official_eligible": False,
                "gap_days": gap,
                "model_id": model_id,
                "outer_fold": outer.fold_id,
                "selected_candidate": ledger["selected_candidate"],
                "selected_parameters": parameters,
                "source_feature_sha256": source_sha,
                "manifest_sha256": manifest_sha,
                "inner_ledger_sha256": sha256(ledger_path),
                "prediction_sha256": sha256(saved),
                "fit_receipt": fit_receipt,
            }
            part_receipt.write_text(
                json.dumps(json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8")
            print(f"PASS {outer.fold_id}/{model_id}: selected={ledger['selected_candidate']} "
                  f"rows={len(score)}", flush=True)
    all_parts = [parts / f"outer_{index}_{model_id}.parquet"
                 for index in range(1, 6) for model_id in MODELS]
    if all(path.is_file() and path.with_suffix(".json").is_file() for path in all_parts):
        combined = pd.concat([pd.read_parquet(path) for path in all_parts], ignore_index=True)
        saved = write_predictions(combined, output / f"outer_tuned{suffix}_predictions.parquet")
        fixed = pd.read_parquet(output / f"outer_fixed{suffix}_predictions.parquet")
        report = evaluate_predictions(pd.concat([fixed, combined], ignore_index=True), formal=False)
        summary = {
            "scope": "nanjing_one_site_five_original_outer_folds_all_cpu_models_diagnostic_only",
            "official_eligible": False,
            "gap_days": gap,
            "source_feature_sha256": source_sha,
            "outer_tuned_prediction_sha256": sha256(saved),
            "models": list(MODELS),
            "prediction_rows": len(combined),
            "probability_primary": report.overview.probability_primary.to_dict("records"),
            "point_auxiliary": report.overview.point_secondary.to_dict("records"),
        }
        (output / f"outer_all_cpu{suffix}_summary.json").write_text(
            json.dumps(json_safe(summary), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")
        print("PASS all 5 outer folds and 10 CPU models", flush=True)


if __name__ == "__main__":
    main()
