"""Development-inner group ablation and grouped permutation for one site.

This is diagnostic mechanism evidence, not a complete feature-selection gate:
the separately preregistered Base+cloud/kt controls and confidence intervals
still remain. No final-test rows or truth enter either analysis.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "project_manifest.yaml").is_file())
sys.path.insert(0, str(ROOT / "src"))

from s01_core.config_loader import load_manifest  # noqa: E402
from s03_features.engineering import formal_daylight_mask  # noqa: E402
from s03_features.preprocessing import fit_fold_preprocessing  # noqa: E402
from s04_splits.rolling import inner_folds, outer_folds  # noqa: E402
from s05_tuning.search_space import candidates  # noqa: E402
from s09_metrics.probabilistic import probability_metrics  # noqa: E402
from s11_interpretation.development_gate import require_development_rows  # noqa: E402

TAUS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
REPEATS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(frame: pd.DataFrame) -> pd.DataFrame:
    mask = formal_daylight_mask(frame) & np.isfinite(pd.to_numeric(frame.ghi_obs_sat, errors="coerce"))
    out = frame.loc[mask].copy()
    out["y"] = out.ghi_obs_sat.astype(float)
    require_development_rows(out)
    if out.empty:
        raise ValueError("development inner block has no eligible rows")
    return out


def fit_quantiles(parameters: dict, x_fit: pd.DataFrame, x_early: pd.DataFrame,
                  x_score: pd.DataFrame, y_fit: np.ndarray,
                  y_early: np.ndarray) -> tuple[np.ndarray, list]:
    predictions, models = [], []
    for tau in TAUS:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=tau, metric="quantile", n_estimators=3000,
            random_state=0, verbose=-1, **parameters)
        model.fit(x_fit, y_fit, eval_set=[(x_early, y_early)],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
        predictions.append(model.predict(x_score, num_iteration=model.best_iteration_))
        models.append(model)
    return np.column_stack(predictions), models


def mean_pinball(y: np.ndarray, q: np.ndarray) -> float:
    return float(probability_metrics(y, q)["mean_pinball"])


def grouped_permutation(models: list, x_score: pd.DataFrame,
                        score: pd.DataFrame, group_columns: list[str],
                        seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    permuted = x_score.copy()
    column_index = [permuted.columns.get_loc(column) for column in group_columns]
    for lead in (24, 48, 72):
        indices = np.flatnonzero(score.lead_time.to_numpy() == lead)
        source = rng.permutation(indices)
        permuted.iloc[indices, column_index] = x_score.iloc[source, column_index].to_numpy()
    return np.column_stack([
        model.predict(permuted, num_iteration=model.best_iteration_)
        for model in models
    ])


def main() -> None:
    manifest_path = ROOT / "project_manifest.yaml"
    manifest = load_manifest(manifest_path)
    root = ROOT / manifest["data_layout"]["root"]
    feature_path = root / "03_featured" / "nanjing_1_featured_2024-02_2026-08.parquet"
    feature_sha, manifest_sha = sha256(feature_path), sha256(manifest_path)
    frame = pd.read_parquet(feature_path)
    groups = {group: columns for group, columns in manifest["feature_groups"].items()
              if group not in {"selection_evidence", "shap_role", "spatial_static"}}
    if len(groups) != 6:
        raise ValueError("expected six nonconstant preregistered feature groups")
    output = root / "06_cpu_single_site" / "group_analysis"
    output.mkdir(parents=True, exist_ok=True)
    for outer_index, outer in enumerate(outer_folds(frame), start=1):
        ledger_path = root / "06_cpu_single_site" / "tuning" / f"{outer.fold_id}_lgbm.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if (ledger.get("status") != "pass" or ledger.get("source_feature_sha256") != feature_sha
                or ledger.get("manifest_sha256") != manifest_sha
                or len(ledger.get("trials", [])) != 18):
            raise ValueError(f"stale or incomplete LightGBM ledger: {ledger_path}")
        selected = int(ledger["selected_candidate"])
        parameters = candidates("lgbm")[selected]
        for inner_index, fold in enumerate(inner_folds(outer), start=1):
            path = output / f"{outer.fold_id}_{fold.fold_id}.json"
            if path.is_file():
                old = json.loads(path.read_text(encoding="utf-8"))
                if (old.get("status") == "pass" and old.get("feature_sha256") == feature_sha
                        and old.get("manifest_sha256") == manifest_sha
                        and old.get("tuning_ledger_sha256") == sha256(ledger_path)
                        and len(old.get("ablation", [])) == len(groups)
                        and len(old.get("permutation", [])) == len(groups) * REPEATS):
                    print(f"REUSE {outer.fold_id}/{fold.fold_id}", flush=True)
                    continue
            fit, early, score = (eligible(block) for block in
                                 (fold.fit, fold.early_stop, fold.score))
            # The development window has no missing forecast cloud. Otherwise
            # a full-feature cloud imputer could leak an ablated group.
            if any(block.cloud_cover_fcst.isna().any() for block in (fit, early, score)):
                raise ValueError("group ablation requires group-specific cloud imputer when cloud is missing")
            (x_fit, x_early, x_score), prep = fit_fold_preprocessing(
                fit, early, score, seed=0)
            y_fit, y_early, y_score = (block.y.to_numpy(dtype=float)
                                       for block in (fit, early, score))
            base_q, base_models = fit_quantiles(parameters, x_fit, x_early,
                                                x_score, y_fit, y_early)
            baseline = mean_pinball(y_score, base_q)
            original = next(row for row in ledger["trials"]
                            if row["candidate"] == selected and row["inner"] == fold.fold_id)
            if not np.isclose(baseline, original["mean_pinball"], rtol=1e-6, atol=1e-6):
                raise ValueError(f"{outer.fold_id}/{fold.fold_id}: baseline did not reproduce tuning")
            ablation, permutation = [], []
            for group_index, (group, columns) in enumerate(groups.items()):
                if any(column not in x_fit for column in columns):
                    raise ValueError(f"{group}: declared feature absent from frozen matrix")
                keep = [column for column in x_fit if column not in columns]
                ablated_q, _ = fit_quantiles(parameters, x_fit[keep], x_early[keep],
                                             x_score[keep], y_fit, y_early)
                ablated_loss = mean_pinball(y_score, ablated_q)
                ablation.append({"group": group, "dropped_features": columns,
                                 "mean_pinball": ablated_loss,
                                 "delta_vs_full": ablated_loss - baseline})
                for repeat in range(REPEATS):
                    q = grouped_permutation(base_models, x_score, score, columns,
                                            seed=10000 * outer_index + 1000 * inner_index
                                                 + 10 * group_index + repeat)
                    loss = mean_pinball(y_score, q)
                    permutation.append({"group": group, "repeat": repeat,
                                        "mean_pinball": loss,
                                        "delta_vs_full": loss - baseline})
            payload = {"status": "pass", "official_eligible": False,
                       "scope": "nanjing_one_site_development_inner_only_group_diagnostic",
                       "outer_fold": outer.fold_id, "inner_fold": fold.fold_id,
                       "selected_candidate": selected, "parameters": parameters,
                       "feature_sha256": feature_sha,
                       "manifest_sha256": manifest_sha,
                       "tuning_ledger_sha256": sha256(ledger_path),
                       "fit_rows": len(fit), "early_stop_rows": len(early),
                       "score_rows": len(score), "baseline_mean_pinball": baseline,
                       "baseline_rounds_by_quantile": [int(model.best_iteration_)
                                                       for model in base_models],
                       "preprocessing": prep,
                       "ablation": ablation, "permutation": permutation,
                       "excluded_group": "spatial_static constant at one site",
                       "controls_not_included": "Base+cloud and Base+kt add-back controls"}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")
            print(f"PASS {outer.fold_id}/{fold.fold_id}: full={baseline:.3f}; "
                  f"groups={len(groups)}", flush=True)
    print("PASS 15 development inner blocks complete", flush=True)


if __name__ == "__main__":
    main()
